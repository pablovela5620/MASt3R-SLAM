from dataclasses import dataclass
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Frame, Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
from mast3r_slam.lietorch_utils import as_SE3
import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager
from beartype.door import infer_hint
from timeit import default_timer as timer
import rerun as rr
from jaxtyping import UInt8, Float32, Int
import numpy as np

# from simplecv.rerun_log_utils import RerunTyroConfig
from pathlib import Path


def format_time(seconds):
    """Format time in minutes:seconds format (mm:ss)."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class InferenceConfig:
    # rr_config: RerunTyroConfig
    dataset: str = (
        "/home/pablo/0Dev/personal/MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_room"
    )
    config: str = "/home/pablo/0Dev/personal/MASt3R-SLAM/config/base.yaml"
    save_as: str = "default"
    no_viz: bool = False


def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal_knowing_depth(
    pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(
        -1, 1, 2
    )  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (
        2 * np.tan(np.deg2rad(60) / 2)
    )  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
    return focal


@rr.shutdown_at_exit
def log_rr_frame_mp(keyframes: SharedKeyframes, states: SharedStates):
    rr.init(application_id="mast3r_slam")
    rr.connect_tcp()

    idx = 0
    # Check mode to know when to exit
    while states.get_mode() != Mode.TERMINATED:
        current_frame: Frame = states.get_frame()
        # Add your rerun logging logic here
        rgb_img: Float32[torch.Tensor, "H W 3"] = current_frame.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img * 255).numpy().astype(np.uint8)
        rr.set_time_sequence(timeline="frame", sequence=idx)
        rr.log(
            "image",
            rr.Image(image=rgb_img, color_model=rr.ColorModel.RGB),
        )

        idx += 1

    # Clean exit when TERMINATED
    print("ReRun visualization process exiting")


class RerunLogger:
    def __init__(self, parent_log_path: Path):
        self.parent_log_path: Path = parent_log_path
        rr.init(application_id="mast3r_slam")
        rr.connect_tcp()
        rr.log(f"{self.parent_log_path}", rr.ViewCoordinates.RDF, static=True)

        self.path_list = []
        self.keyframe_logged_list = []

    def log_frame(
        self, current_frame: Frame, keyframes: SharedKeyframes, states: SharedStates
    ):
        # Add your rerun logging logic here
        H = current_frame.img_shape.squeeze()[0].item()
        W = current_frame.img_shape.squeeze()[1].item()

        pp = torch.tensor((W / 2, H / 2))
        pts3d: torch.Tensor = current_frame.X_canon.clone().cpu().reshape(H, W, 3)
        focal = float(
            estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld")
        )

        rgb_img: Float32[torch.Tensor, "H W 3"] = current_frame.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img * 255).numpy().astype(np.uint8)

        se3_pose: lietorch.SE3 = as_SE3(current_frame.T_WC.cpu())
        matb4x4: Float32[np.ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        mat4x4: Float32[np.ndarray, "4 4"] = matb4x4[
            0
        ]  # Extract the first batch element

        # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
        rotation_matrix: Float32[np.ndarray, "3 3"] = mat4x4[
            :3, :3
        ]  # Top-left 3x3 block
        translation_vector: Float32[np.ndarray, "3"] = mat4x4[
            :3, 3
        ]  # Right column, first 3 elements
        image_plane_distance = 0.1
        cam_log_path = self.parent_log_path / "camera"
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                focal_length=focal,
                principal_point=pp.numpy(),
                height=H,
                width=W,
                image_plane_distance=image_plane_distance,
            ),
        )
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(image=rgb_img, color_model=rr.ColorModel.RGB).compress(),
        )
        self.path_list.append(translation_vector.tolist())
        rr.log(
            f"{self.parent_log_path}/path",
            rr.LineStrips3D(
                strips=self.path_list,
                colors=(255, 0, 0),
                labels=("Camera Path"),
            ),
        )

        with keyframes.lock:
            N_keyframes = len(keyframes)
            # dirty_idx = keyframes.get_dirty_idx()

        for kf_idx in range(N_keyframes):
            keyframe: Frame = keyframes[kf_idx]
            se3_pose: lietorch.SE3 = as_SE3(keyframe.T_WC.cpu())
            matb4x4: Float32[np.ndarray, "1 4 4"] = (
                se3_pose.matrix().numpy().astype(dtype=np.float32)
            )
            mat4x4: Float32[np.ndarray, "4 4"] = matb4x4[
                0
            ]  # Extract the first batch element

            # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
            rotation_matrix: Float32[np.ndarray, "3 3"] = mat4x4[
                :3, :3
            ]  # Top-left 3x3 block
            translation_vector: Float32[np.ndarray, "3"] = mat4x4[
                :3, 3
            ]  # Right column, first 3 elements
            cam_log_path = self.parent_log_path / f"keyframe-{kf_idx}"
            if kf_idx not in self.keyframe_logged_list:
                kf_img: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
                kf_img: UInt8[np.ndarray, "H W 3"] = (
                    (kf_img * 255).numpy().astype(np.uint8)
                )
                rr.log(
                    f"{cam_log_path}/pinhole/image",
                    rr.Image(image=kf_img, color_model=rr.ColorModel.RGB).compress(),
                )
                # create a mask based on the confidence values
                mask = keyframe.C.cpu().numpy() > 5

                # Convert the mask from shape (h*w, 1) to shape (h*w,)
                mask = (
                    mask.squeeze()
                )  # Remove the trailing dimension to get a 1D boolean array

                # Now apply the mask to both positions and colors
                positions: Float32[np.ndarray, "num_points 3"] = (
                    keyframe.X_canon.cpu().numpy()
                )
                colors: UInt8[np.ndarray, "num_points 3"] = kf_img.reshape(-1, 3)

                masked_positions = positions[
                    mask
                ]  # Now selects entire rows where mask is True
                masked_colors = colors[mask]
                rr.log(
                    f"{cam_log_path}/pointcloud",
                    rr.Points3D(
                        positions=masked_positions,
                        colors=masked_colors,
                    ),
                )
                self.keyframe_logged_list.append(kf_idx)
            rr.log(
                f"{cam_log_path}",
                rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
            )
            rr.log(
                f"{cam_log_path}/pinhole",
                rr.Pinhole(
                    focal_length=focal,
                    principal_point=pp.numpy(),
                    height=H,
                    width=W,
                    image_plane_distance=image_plane_distance,
                ),
            )

        with states.lock:
            ii: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_ii, dtype=torch.long
            )
            jj: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_jj, dtype=torch.long
            )
            if ii.numel() > 0 and jj.numel() > 0:
                T_WCi = lietorch.Sim3(keyframes.T_WC[ii, 0])
                T_WCj = lietorch.Sim3(keyframes.T_WC[jj, 0])
        if ii.numel() > 0 and jj.numel() > 0:
            t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
            t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
            line_strips = []
            for t_i, t_j in zip(t_WCi, t_WCj):
                line_strips.append(t_i.tolist())
                line_strips.append(t_j.tolist())
            rr.log(
                f"{self.parent_log_path}/edges",
                rr.LineStrips3D(
                    strips=line_strips, colors=(0, 255, 0), labels=("Factor Graph")
                ),
            )


def mast3r_slam_inference(inf_config: InferenceConfig):
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now: str = str(datetime.datetime.now()).replace(" ", "_")

    ## rerun setup
    parent_log_path = Path("world")
    rr_logger = RerunLogger(parent_log_path)

    load_config(inf_config.config)
    print(inf_config.dataset)
    print(config)

    manager: SyncManager = mp.Manager()
    main2viz = new_queue(manager, inf_config.no_viz)
    viz2main = new_queue(manager, inf_config.no_viz)

    dataset = load_dataset(inf_config.dataset)
    dataset.subsample(config["dataset"]["subsample"])

    h, w = dataset.get_img_shape()[0]
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not inf_config.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(states, keyframes, main2viz, viz2main, inf_config.config),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib: bool = dataset.has_calib()
    use_calib: bool = config["use_calib"]
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)
        print(f"Saving results to {save_dir}")
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.pt"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(
        target=run_backend, args=(inf_config.config, model, states, keyframes, K)
    )
    backend.start()

    i = 0
    fps_timer: float = time.time()
    start_time = timer()

    frames = []

    while True:
        rr.set_time_sequence(timeline="frame", sequence=i)
        mode: Mode = states.get_mode()
        msg: WindowMsg | None = try_get_msg(viz2main)
        last_msg: WindowMsg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC: lietorch.Sim3 = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame: Frame = create_frame(
            i, img, T_WC, img_size=dataset.img_size, device=device
        )

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features needed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        ## rerun log stuff
        rr_logger.log_frame(frame, keyframes, states)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(inf_config, dataset)
        eval.save_ATE(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir, f"{seq_name}.pt", dataset.timestamps, keyframes
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        print(len(frames))
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    print(f"Inference time: {format_time(timer() - start_time)}")
    print(f"Processed {len(keyframes)}")
    backend.join()
    if not inf_config.no_viz:
        # First wait for visualization to finish
        viz.join()
        # Then wait for ReRun process to finish
        # viz_rr.join()
        print("All visualization processes terminated")


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(config_path, model, states, keyframes, K):
    load_config(config_path)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success: bool = relocalization(
                frame, keyframes, factor_graph, retrieval_database
            )
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
