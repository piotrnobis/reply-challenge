import os
import cv2
import subprocess

import pycolmap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
        subprocess.run([
            "colmap", "mapper",
            "--database_path", self.database_path,
            "--image_path", self.image_dir,
            "--output_path", self.sparse_dir
        ])
        print("constructed!!")

    def estimate_relative_poses(self):
        self.run_feature_extraction()
        self.run_feature_matching()
        self.run_sparse_reconstruction()

    def iterate_images(self):
        model = pycolmap.Reconstruction(os.path.join(self.sparse_dir,"0"))

        # Access cameras, images, and 3D points
        for image_id, image in model.images.items():
            name = image.name
            img_ctr = int(name.split('_')[2])
            if img_ctr >= 50 and img_ctr <= 50:
                R = image.cam_from_world.rotation          # 3x3 rotation
                t = image.cam_from_world.translation               # translation
                K = image.camera.calibration_matrix()  # intrinsic matrix (3x3)

                # print(f"Image: {name}, Pose:\nR=\n{R}, t={t}, K=\n{K}")

                # Load image from disk
                image_path = os.path.join(self.image_dir, name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not load image: {image_path}")
                    continue

                # Convert BGR (OpenCV default) to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get 2D keypoints (Nx2 array: x, y)
                keypoints = image.points2D

                # Plot
                plt.figure(figsize=(10, 8))
                plt.imshow(img_rgb)
                plt.title(f"Keypoints for: {name}")

                # Plot keypoints
                for pt in keypoints:
                    x, y = pt.xy
                    plt.gca().add_patch(Circle((x, y), radius=1.5, color='lime', linewidth=0.5))

                plt.axis("off")
                plt.show()