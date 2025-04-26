import numpy as np
from helpers.Pic import Pic

# Function to load 3D points
def load_points3d(file_path):
    points = []
    ids = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[3:]  # Skip the first 3 lines
        for line in lines:
            parts = line.strip().split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            points.append([x, y, z])
            ids.append(point_id)
    return np.array(points), np.array(ids)

# Function to load camera positions (TX, TY, TZ) from images.txt
def load_camera_positions(images_txt_path):
    colmap_positions = []
    image_names = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()[4:]  # Skip the first 3 header lines
        for i in range(0, len(lines), 2):  # Read every two lines, pick only the first
            line = lines[i]
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 9:
                tx, ty, tz = map(float, parts[5:8])
                colmap_positions.append([tx, ty, tz])
                image_name = parts[9]
                image_names.append(image_name)
    return np.array(colmap_positions), image_names

# Load GPS data using your Pic class
def load_gps_positions(image_names, gps_folder_path):
    gps_positions = []
    for name in image_names:
        pic = Pic(f"{gps_folder_path}/{name}")
        gpsinfo = pic["GPSInfo"]
        gps_positions.append([float(gpsinfo['latitude']), float(gpsinfo['longitude']), float(gpsinfo['altitude'])])
    return np.array(gps_positions)

# Estimate transformation
def estimate_transform(colmap_positions, gps_positions):
    colmap_origin = np.mean(colmap_positions, axis=0)
    gps_origin = np.mean(gps_positions, axis=0)

    # Estimate scale (roughly) using distances
    colmap_dist = np.linalg.norm(colmap_positions[0] - colmap_positions[1])
    gps_dist = np.linalg.norm(gps_positions[0] - gps_positions[1])  # In degrees, approx meters if small area
    scale = gps_dist / colmap_dist

    return colmap_origin, gps_origin, scale

# Apply transformation to points
def transform_points(points, colmap_origin, gps_origin, scale):
    shifted_points = points - colmap_origin
    gps_points = gps_origin + scale * shifted_points
    return gps_points

# Main
points, point_ids = load_points3d(r"C:\Users\anasn\OneDrive\Desktop\reply-challenge\colmap_workspace\points3D.txt")
colmap_positions, image_names = load_camera_positions(r"C:\Users\anasn\OneDrive\Desktop\reply-challenge\colmap_workspace\images.txt")
gps_positions = load_gps_positions(image_names, r"C:\Users\anasn\OneDrive\Desktop\reply-challenge\data\dev_data")

colmap_origin, gps_origin, scale = estimate_transform(colmap_positions, gps_positions)
gps_points = transform_points(points, colmap_origin, gps_origin, scale)

# Output example
for pid, gps_point in zip(point_ids, gps_points):
    print(f"Point ID {pid}: Latitude {gps_point[0]}, Longitude {gps_point[1]}, Altitude {gps_point[2]}")



# ecef