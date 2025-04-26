import os
import subprocess

import pycolmap
import numpy as np
import cv2

from pathlib import Path

from helpers import Colmap, Roboflow

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
colmap.image_dir = os.path.join(colmap.data_dir, "dev_data_blurred")
# colmap.estimate_relative_poses()
# colmap.run_model_converter()
colmap.plot_pointcloud_better()