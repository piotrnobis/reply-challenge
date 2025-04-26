import os
import subprocess

import pycolmap

class Colmap:
    def __init__(self, project_dir):
        self.image_dir = os.path.join(project_dir, "data/dev_data")
        self.workspace_dir = os.path.join(project_dir, "colmap_workspace")
        self.database_path = os.path.join(self.workspace_dir, "database.db")
        self.sparse_dir = os.path.join(self.workspace_dir, "sparse")

        os.makedirs(self.sparse_dir, exist_ok=True)

    def run_feature_extraction(self):
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", self.database_path,
            "--image_path", self.image_dir,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
        ])

    def run_feature_matching(self):
        subprocess.run([
            "colmap", "sequential_matcher",
            "--database_path", self.database_path,
            "--SiftMatching.use_gpu", "0",
            "--SequentialMatching.overlap", "5"
        ])

    def run_sparse_reconstruction(self):
        ([
            "colmap", "mapper",
            "--database_path", self.database_path,
            "--image_path", self.image_dir,
            "--output_path", self.sparse_dir
        ])

    def estimate_relative_poses(self):
        self.run_feature_extraction()
        self.run_feature_matching()
        self.run_sparse_reconstruction()

    def iterate_images(self):
        model = pycolmap.Reconstruction(f"{self.sparse_dir}/0")

        # Access cameras, images, and 3D points
        for image_id, image in model.images.items():
            name = image.name
            R = image.cam_from_world.rotation          # 3x3 rotation
            t = image.cam_from_world.translation               # translation
            K = image.camera.calibration_matrix()  # intrinsic matrix (3x3)

            print(f"Image: {name}, Pose:\nR=\n{R}, t={t}, K=\n{K}")