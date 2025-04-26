import os
import subprocess

import pycolmap
import numpy as np
import cv2

from pathlib import Path
import pandas as pd

from helpers import Colmap, Roboflow, Triangulation, Pic

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
triangulation = Triangulation()

os.makedirs(os.path.join(project_dir, "data", "dev_data_test"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "data", "dev_data_test_blurred"), exist_ok=True)

img1_name = "DJI_20250424162351_0130_V.jpeg"
img2_name = "DJI_20250424162352_0131_V.jpeg"

# Path to your images
img1_raw_path = os.path.join(project_dir, "data", "dev_data_test", img1_name)
img2_raw_path = os.path.join(project_dir, "data", "dev_data_test", img2_name)
# img3_raw_path = os.path.join(project_dir, "data", "dev_data_test", "DJI_20250424162354_0133_V.jpeg")

results = triangulation.detect_barcodes(os.path.join(project_dir, "data", "dev_data_test"))
roboflow.blur_except_rectangles(results, 
                                os.path.join(project_dir, "data", "dev_data_test"),
                                os.path.join(project_dir, "data", "dev_data_test_blurred"))

# Path to your images
img1_path = os.path.join(project_dir, "data", "dev_data_test_blurred", img1_name)
img2_path = os.path.join(project_dir, "data", "dev_data_test_blurred", img2_name)
# img3_path = os.path.join(project_dir, "data", "dev_data_test_blurred", "DJI_20250424162354_0133_V.jpeg")

# Example camera intrinsic matrix (replace with your actual values)
# Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
intrinsic_matrix = np.array([
    [np.float64(2674.582355102382), 0, np.float64(2016.0)],  # fx, 0, cx
    [0, np.float64(2668.056313282602), np.float64(1512.0)],  # 0, fy, cy
    [0, 0, 1]        # 0, 0, 1
])

file_metadata_1 = Pic(img1_raw_path)
file_metadata_2 = Pic(img2_raw_path)

# Example GPS coordinates (latitude, longitude, altitude in meters)
# Replace with your actual drone positions
location1 = (float(file_metadata_1['GPSInfo']['latitude']),
             float(file_metadata_1['GPSInfo']['longitude']),
             float(file_metadata_1['GPSInfo']['altitude']))  # San Francisco example
location2 = (float(file_metadata_2['GPSInfo']['latitude']),
             float(file_metadata_2['GPSInfo']['longitude']),
             float(file_metadata_2['GPSInfo']['altitude']))  # A short distance away

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

df_selected = triangulation.extract_one_feature_per_rectangle(results[img1_name]['white_spaces'],
                                                              pts1, points_lat_lon_alt, img1)

df_selected.to_csv("triangulated_selected_points_gps.csv", index=False)

# Print the number of triangulated points
print(f"Successfully triangulated {len(points_3d)} 3D points")