import tyro

from mast3r_slam.api.inferece import InferenceConfig, mast3r_slam_inference

if __name__ == "__main__":
    mast3r_slam_inference(tyro.cli(InferenceConfig))
