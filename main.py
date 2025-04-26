import os
import subprocess

import pycolmap
import numpy as np
import cv2

from pathlib import Path

from helpers import Colmap

current_file_dir = Path(__file__).resolve().parent
project_dir = current_file_dir
colmap = Colmap(project_dir)
colmap.estimate_relative_poses()
colmap.iterate_images()
colmap.run_model_converter()
points, colors = colmap.print_3D_points(scale = 3)
colmap.plot_pointcloud(points, colors)
