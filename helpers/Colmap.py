import os
import cv2
import subprocess
import random

import pycolmap
import numpy as np
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

    def run_model_converter(self):
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", os.path.join(self.sparse_dir, "0"),
            "--output_path", self.workspace_dir,
            "--output_type", "TXT"
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
            plt.show(block=True)

    def print_3D_points(self, scale = 1.0, keep_ratio=0.40):
        points = []
        colors = []
        file_path = os.path.join(self.workspace_dir, "points3D.txt")
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue  # Skip comments and empty lines
                """ if random.random() > keep_ratio:
                    continue """  # Skip this point with probability 1 - keep_ratio
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                """ x *= scale
                y *= scale
                z *= scale """
                points.append((x, y, z))
                colors.append((r / 255.0, g / 255.0, b / 255.0))
        return points, colors
    
    def is_green(self, r, g, b):
        """
        Determines if a color is green based on its RGB values.

        Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)

        Returns:
        bool: True if the color is green, False otherwise.
        """
        # Conditions for a color to be considered "green"
        if g > 80 and g > r + 20 and g > b + 20:
            return True
        return False
    
    def set_axes_equal(self, ax):
        ''' Make axes of 3D plot have equal scale so the plot isn't distorted '''
        limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        spans = [abs(lim[1] - lim[0]) for lim in limits]
        centers = [(lim[0] + lim[1]) / 2 for lim in limits]
        max_span = max(spans)
        new_limits = [(c - max_span/2, c + max_span/2) for c in centers]
        ax.set_xlim(new_limits[0])
        ax.set_ylim(new_limits[1])
        ax.set_zlim(new_limits[2])


    def plot_pointcloud(self, points, colors):
        xs, ys, zs = zip(*points)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, s=5, c=colors, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Colored and Scaled 3D Point Cloud')
        # self.set_axes_equal(ax)
        # ax.view_init(elev=20, azim=45)  # Adjust these angles if you want
        plt.show(block=True)