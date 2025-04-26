import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
points3D_path = r'C:\Users\anasn\OneDrive\Desktop\reply-challenge\colmap_workspace\points3D.txt'
images_path = r'C:\Users\anasn\OneDrive\Desktop\reply-challenge\colmap_workspace\images.txt'

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

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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

plt.show()
