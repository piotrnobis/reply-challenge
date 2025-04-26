import os
import cv2
import subprocess
import random
import math

import pycolmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

class Colmap:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, "data")
        self.image_dir = os.path.join(project_dir, self.data_dir, "dev_data")
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

    
    def plot_pointcloud_better(self):

        # --- Load points3D.txt with colors ---
        def read_points3D_txt(path):
            points3D = {}
            colors = {}
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('#'):
                        continue
                    elems = line.strip().split()
                    point3D_id = int(elems[0])
                    xyz = np.array(list(map(float, elems[1:4])))
                    rgb = np.array(list(map(int, elems[4:7])))
                    points3D[point3D_id] = xyz
                    colors[point3D_id] = rgb / 255.0  # Normalize color to [0, 1] for matplotlib
            return points3D, colors

        # --- Load images.txt ---
        def read_images_txt(path):
            poses = {}
            with open(path, 'r') as f:
                lines = f.readlines()
                idx = 0
                while idx < len(lines):
                    line = lines[idx]
                    if line.startswith('#'):
                        idx += 1
                        continue
                    elems = line.strip().split()
                    image_id = int(elems[0])
                    qw, qx, qy, qz = map(float, elems[1:5])
                    tx, ty, tz = map(float, elems[5:8])
                    poses[image_id] = (qw, qx, qy, qz, tx, ty, tz)
                    idx += 2  # images.txt: image data every 2 lines
            return poses

        # --- Paths to your reconstruction files ---
        points3D_path = os.path.join(self.workspace_dir, "points3D.txt")
        images_path = os.path.join(self.workspace_dir, "images.txt")

        # --- Load data ---
        points3D, colors = read_points3D_txt(points3D_path)
        poses = read_images_txt(images_path)

        # --- Extract arrays ---
        all_points = np.array(list(points3D.values()))
        all_colors = np.array(list(colors.values()))
        all_cameras = np.array([pose[4:] for pose in poses.values()])  # (tx, ty, tz)

        # --- Flip Z axis (if reconstruction is upside down) ---
        all_points[:, 2] *= -1  # Flip 3D points
        all_cameras[:, 2] *= -1  # Flip camera positions too

        # --- Plot ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points with colors
        ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], c=all_colors, s=0.5, label='3D Points (colored)')

        # Plot camera positions
        ax.scatter(all_cameras[:,0], all_cameras[:,1], all_cameras[:,2], c='red', s=30, label='Camera Positions')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Sparse Reconstruction (Fixed Orientation)')

        # Improve visualization
        ax.view_init(elev=30, azim=-60)  # Adjust viewing angle if you want
        ax.grid(True)

        plt.show(block=True)

        # Load points, colors, and poses as before
        # (re-use read_points3D_txt and read_images_txt functions from previous code)

        # --- Pick a specific camera ---
        selected_image_id = 1  # Choose one image ID from your dataset
        selected_pose = poses[selected_image_id]
        qw, qx, qy, qz, tx, ty, tz = selected_pose

        # Flip Z (if you did it earlier)
        tz *= -1

        # --- Quaternion to rotation matrix ---
        def quaternion_to_rotation_matrix(qw, qx, qy, qz):
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            return R

        # Calculate rotation matrix
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

        # --- Set up plot ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points
        ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], c=all_colors, s=0.5, label='3D Points')

        # Plot all camera positions
        ax.scatter(all_cameras[:,0], all_cameras[:,1], all_cameras[:,2], c='red', s=30, label='Camera Positions')

        # Label
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'View from Camera {selected_image_id}')

        # --- Set viewing position ---
        # Move the view so that (tx, ty, tz) is the center
        ax.set_xlim(tx-5, tx+5)
        ax.set_ylim(ty-5, ty+5)
        ax.set_zlim(tz-5, tz+5)

        # --- Point the virtual camera ---
        # Calculate angles for ax.view_init (matplotlib needs elevation and azimuth)

        # The forward vector (camera's look direction)
        forward = R @ np.array([0, 0, 1])  # In camera frame, "forward" is usually +Z

        # Convert the forward vector to spherical coordinates (elevation, azimuth)
        def cartesian_to_spherical(x, y, z):
            r = np.linalg.norm([x, y, z])
            elev = math.degrees(math.asin(z/r))
            azim = math.degrees(math.atan2(y, x))
            return elev, azim

        elev, azim = cartesian_to_spherical(*forward)

        ax.view_init(elev=elev, azim=azim)

        plt.show(block=True)
