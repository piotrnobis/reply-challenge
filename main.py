import os
import subprocess

import pycolmap
import numpy as np
import cv2

from pathlib import Path
import pandas as pd

from helpers import Colmap, Roboflow, Triangulation

current_file_dir = Path(__file__).resolve().parent
project_dir = current_file_dir
colmap = Colmap(project_dir)
# colmap.estimate_relative_poses()
# colmap.iterate_images()
# colmap.run_model_converter()
# points, colors = colmap.print_3D_points(scale = 3)
# colmap.plot_pointcloud(points, colors)

model_id1 = "pile-of-crate-detection/10"  # Replace with your model ID
image_dir = "dev_data_demo"  # Your images folder

roboflow = Roboflow()
# all_detections = roboflow.run_inference_on_images(model_id1, colmap.image_dir, show_img=False)

# Example: print the final result
""" print("\nFinal Detections:")
for image_name, boxes in all_detections.items():
    print(f"Image: {image_name}")
    print("Boxes:", boxes) """

# roboflow.blur_except_rectangles(all_detections, colmap.image_dir, os.path.join(colmap.data_dir, "dev_data_blurred"))
# Run colmap with blurred images
# colmap.image_dir = os.path.join(colmap.data_dir, "dev_data_blurred")
""" colmap.estimate_relative_poses()
colmap.run_model_converter()
colmap.plot_pointcloud_better()
 """
# Path to your images
img1_path = os.path.join(project_dir, "data", "dev_data_blurred", "DJI_20250424193048_0052_V.jpeg")
img2_path = os.path.join(project_dir, "data", "dev_data_blurred", "DJI_20250424193049_0053_V.jpeg")

# Example camera intrinsic matrix (replace with your actual values)
# Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
intrinsic_matrix = np.array([
    [np.float64(2674.582355102382), 0, np.float64(2016.0)],  # fx, 0, cx
    [0, np.float64(2668.056313282602), np.float64(1512.0)],  # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])

# Example GPS coordinates (latitude, longitude, altitude in meters)
# Replace with your actual drone positions
location1 = (49.0993716667, 12.18093, 483.34)  # San Francisco example
location2 = (49.0993761111, 12.180945, 483.124)  # A short distance away

triangulation = Triangulation()

# Load images
img1, img2, gray1, gray2 = triangulation.load_images(img1_path, img2_path)

# Detect and match features
kp1, kp2, good_matches, pts1, pts2 = triangulation.detect_and_match_features(gray1, gray2)

# Visualize matches
triangulation.visualize_matches(img1, img2, kp1, kp2, good_matches)

# Create camera matrices
P1, P2 = triangulation.create_camera_matrices(intrinsic_matrix, location1, location2)

# Triangulate 3D points
points_3d = triangulation.triangulate_points(P1, P2, pts1, pts2)

# Plot 3D points
triangulation.plot_3d_points(points_3d, location1, location2)

# Convert 3D points to latitude, longitude, altitude
points_lat_lon_alt = triangulation.convert_to_lat_lon_alt(points_3d, location1)

# Create a DataFrame for better visualization
df_points = pd.DataFrame(points_lat_lon_alt, columns=['Latitude', 'Longitude', 'Altitude'])

# Export the results to CSV if needed
df_points.to_csv("triangulated_points_gps.csv", index=False)

# Print the number of triangulated points
print(f"Successfully triangulated {len(points_3d)} 3D points")