import os
import numpy as np
import json
import math
from pyproj import Transformer
import sqlite3
import collections
from .. import Pic

class ColmapToGPS:
    def __init__(self, workspace_dir, project_dir, images_folder):
        """Initialize with the workspace directory containing COLMAP output."""
        self.data = os.path.join(project_dir, "data")
        self.images_files_path = os.path.join(self.data, images_folder)
        self.workspace_dir = workspace_dir
        self.database_path = os.path.join(workspace_dir, "database.db")
        self.images_path = os.path.join(workspace_dir, "images.txt")
        self.points3D_path = os.path.join(workspace_dir, "points3D.txt")
        self.cameras_path = os.path.join(workspace_dir, "cameras.txt")
        
        # WGS84 (latitude/longitude) to ECEF (Earth-Centered, Earth-Fixed) transformer
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        
        # Load data
        self.cameras = self.load_cameras()
        self.images = self.load_images()
        self.points3D = self.load_points3D()
        self.image_gps = self.extract_image_gps()
        
    def load_cameras(self):
        """Load camera information from cameras.txt."""
        cameras = {}
        with open(self.cameras_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                elems = line.strip().split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = [float(param) for param in elems[4:]]
                cameras[camera_id] = {
                    'model': model,
                    'width': width, 
                    'height': height,
                    'params': params  # For PINHOLE: [fx, fy, cx, cy]
                }
        return cameras
    
    def load_images(self):
        """Load image information from images.txt."""
        images = {}
        with open(self.images_path, 'r') as f:
            image_info = None
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                
                elems = line.strip().split()
                if len(elems) == 9:  # First line of image info
                    image_id = int(elems[0])
                    qw, qx, qy, qz = [float(e) for e in elems[1:5]]
                    tx, ty, tz = [float(e) for e in elems[5:8]]
                    camera_id = int(elems[8])
                    image_info = {
                        'image_id': image_id,
                        'qvec': [qw, qx, qy, qz],  # Quaternion rotation
                        'tvec': [tx, ty, tz],      # Translation vector
                        'camera_id': camera_id,
                        'name': None,
                        'points2D': []
                    }
                else:  # Second line with image name
                    image_name = elems[0]
                    image_info['name'] = image_name
                    # The rest of the elements are point2D info (x, y, point3D_id)
                    points2D = []
                    for i in range(1, len(elems), 3):
                        if i+2 < len(elems):
                            x = float(elems[i])
                            y = float(elems[i+1])
                            point3D_id = int(elems[i+2])
                            points2D.append((x, y, point3D_id))
                    image_info['points2D'] = points2D
                    images[image_info['image_id']] = image_info
                    image_info = None
        return images
    
    def load_points3D(self):
        """Load 3D points from points3D.txt."""
        points3D = {}
        with open(self.points3D_path, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                elems = line.strip().split()
                point3D_id = int(elems[0])
                x, y, z = [float(e) for e in elems[1:4]]
                r, g, b = [int(e) for e in elems[4:7]]
                error = float(elems[7])
                # Track IDs and features are in the remaining elements
                track = []
                for i in range(8, len(elems), 2):
                    if i+1 < len(elems):
                        image_id = int(elems[i])
                        point2D_idx = int(elems[i+1])
                        track.append((image_id, point2D_idx))
                
                points3D[point3D_id] = {
                    'id': point3D_id,
                    'xyz': [x, y, z],
                    'rgb': [r, g, b],
                    'error': error,
                    'track': track  # List of (image_id, point2D_idx) pairs
                }
        return points3D
    
    def extract_image_gps(self):
        """Extract GPS coordinates from image metadata stored in the database."""
        image_gps = {}
        
        try:
            poses = {}
            with open(self.images_path, 'r') as f:
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
                    image_name = elems[9]
                    idx += 2  # images.txt: image data every 2 lines
                
                    # Extract GPS data from EXIF tags in the database
                    # This is a simplified approach - in a real scenario, you'd need to
                    # parse EXIF data properly depending on how it's stored in your database
                    
                    # For demonstration, let's assume you have a function to extract GPS from EXIF
                    # Replace this with actual extraction logic based on your database schema
                    lat, lon, alt = self.extract_gps_from_exif(image_name)
                    
                    if lat is not None and lon is not None:
                        image_gps[image_id] = {
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': alt if alt is not None else 0.0
                        }
        except Exception as e:
            print(f"Error extracting GPS from database: {e}")
            # As a fallback, try reading from a JSON file or other source
            
        return image_gps
    
    def extract_gps_from_exif(self, image_name):
        """
        Extract GPS coordinates from image EXIF data.
        In a real implementation, you would parse EXIF data from the database or image files.
        For this example, we'll implement a mock function.
        """
        # TODO: Replace this with actual extraction from your database or files
        # This is a placeholder - in practice, you need to implement this based on
        # how your GPS data is stored (database, separate file, etc.)
        
        # Mock implementation - replace with your actual data access method
        try:
            # Check if there's a JSON file with GPS data
            file_metadata = Pic(os.path.join(self.images_files_path, image_name))
            gpsinfo = file_metadata["GPSInfo"]
            return (
                float(gpsinfo['latitude']),
                float(gpsinfo['longitude']),
                float(gpsinfo['altitude'])
            )
            
            # Alternative: Parse the image file directly for EXIF data using a library like exifread
            # import exifread
            # with open(os.path.join(image_dir, image_name), 'rb') as f:
            #     tags = exifread.process_file(f)
            #     # Extract and convert GPS coordinates
            #     # ...
            
            print(f"Warning: No GPS data found for {image_name}")
            return None, None, None
        except Exception as e:
            print(f"Error extracting GPS from EXIF: {e}")
            return None, None, None
    
    def quaternion_to_rotation_matrix(self, qvec):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = qvec
        R = np.zeros((3, 3))
        
        R[0, 0] = 1 - 2 * y * y - 2 * z * z
        R[0, 1] = 2 * x * y - 2 * w * z
        R[0, 2] = 2 * x * z + 2 * w * y
        
        R[1, 0] = 2 * x * y + 2 * w * z
        R[1, 1] = 1 - 2 * x * x - 2 * z * z
        R[1, 2] = 2 * y * z - 2 * w * x
        
        R[2, 0] = 2 * x * z - 2 * w * y
        R[2, 1] = 2 * y * z + 2 * w * x
        R[2, 2] = 1 - 2 * x * x - 2 * y * y
        
        return R
    
    def compute_colmap_to_gps_transform(self):
        """
        Compute transformation from COLMAP coordinate system to WGS84 (GPS).
        Uses camera positions with known GPS coordinates to establish the transformation.
        """
        # Points in COLMAP space
        colmap_points = []
        # Corresponding points in ECEF (Earth-Centered, Earth-Fixed) coordinates
        ecef_points = []
        
        for image_id, gps_data in self.image_gps.items():
            if image_id in self.images:
                image = self.images[image_id]
                
                # COLMAP camera center: -R.T @ t
                R = self.quaternion_to_rotation_matrix(image['qvec'])
                t = np.array(image['tvec'])
                camera_center = -R.T @ t
                
                # Convert GPS to ECEF coordinates
                lon, lat, alt = gps_data['longitude'], gps_data['latitude'], gps_data['altitude']
                x, y, z = self.transformer.transform(lon, lat, alt)
                
                colmap_points.append(camera_center)
                ecef_points.append(np.array([x, y, z]))
        
        # Need at least 3 points for a decent transform
        if len(colmap_points) < 3:
            raise ValueError(f"Not enough GPS data points: {len(colmap_points)}. Need at least 3.")
        
        # Convert to numpy arrays
        colmap_points = np.array(colmap_points)
        ecef_points = np.array(ecef_points)
        
        # Calculate centroids
        colmap_centroid = np.mean(colmap_points, axis=0)
        ecef_centroid = np.mean(ecef_points, axis=0)
        
        # Center the points
        colmap_centered = colmap_points - colmap_centroid
        ecef_centered = ecef_points - ecef_centroid
        
        # Calculate scales
        colmap_scale = np.mean(np.linalg.norm(colmap_centered, axis=1))
        ecef_scale = np.mean(np.linalg.norm(ecef_centered, axis=1))
        
        # Scale factor
        scale = ecef_scale / colmap_scale
        
        # Calculate rotation using SVD
        H = colmap_centered.T @ ecef_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure R is a proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = ecef_centroid - scale * (R @ colmap_centroid)
        
        return {
            'rotation': R,
            'scale': scale,
            'translation': t,
            'colmap_centroid': colmap_centroid,
            'ecef_centroid': ecef_centroid
        }
    
    def transform_point(self, point, transform):
        """
        Transform a point from COLMAP coordinates to ECEF coordinates.
        """
        point_np = np.array(point)
        R = transform['rotation']
        s = transform['scale']
        t = transform['translation']
        
        # Apply transformation: s * R * point + t
        transformed = s * (R @ point_np) + t
        return transformed
    
    def ecef_to_geodetic(self, x, y, z):
        """
        Convert ECEF coordinates back to geodetic (latitude, longitude, altitude).
        """
        # Convert ECEF to WGS84 geodetic coordinates using pyproj
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
        lon, lat, alt = transformer.transform(x, y, z)
        return lat, lon, alt
    
    def convert_points_to_gps(self):
        """
        Convert all 3D points from COLMAP to GPS coordinates.
        """
        # Compute the transformation
        transform = self.compute_colmap_to_gps_transform()
        
        # Apply transformation to all 3D points
        gps_points = {}
        for point_id, point_data in self.points3D.items():
            # Transform to ECEF
            ecef = self.transform_point(point_data['xyz'], transform)
            
            # Convert ECEF to geodetic (lat, lon, alt)
            lat, lon, alt = self.ecef_to_geodetic(ecef[0], ecef[1], ecef[2])
            
            gps_points[point_id] = {
                'id': point_id,
                'colmap_xyz': point_data['xyz'],
                'ecef_xyz': ecef.tolist(),
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'rgb': point_data['rgb']
            }
        
        return gps_points
    
    def save_gps_points(self, output_file='gps_points.json'):
        """
        Save GPS points to a JSON file.
        """
        gps_points = self.convert_points_to_gps()
        
        # Convert to a format suitable for JSON serialization
        output_data = {str(k): v for k, v in gps_points.items()}
        
        with open(os.path.join(self.workspace_dir, output_file), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(gps_points)} GPS points to {output_file}")
        return output_file
    
    def export_kml(self, output_file='points3D.kml'):
        """
        Export GPS points to KML format for visualization in Google Earth.
        """
        gps_points = self.convert_points_to_gps()
        
        kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>COLMAP 3D Points</name>
  <Style id="redPoint">
    <IconStyle>
      <color>ff0000ff</color>
      <scale>0.5</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
    </IconStyle>
  </Style>
"""
        
        kml_footer = """
</Document>
</kml>"""
        
        with open(os.path.join(self.workspace_dir, output_file), 'w') as f:
            f.write(kml_header)
            
            for point_id, point in gps_points.items():
                # Extract RGB color
                r, g, b = point['rgb']
                # KML color format is aabbggrr (alpha, blue, green, red)
                kml_color = f"ff{b:02x}{g:02x}{r:02x}"
                
                # Create placemark entry
                placemark = f"""
  <Placemark>
    <name>Point {point_id}</name>
    <Style>
      <IconStyle>
        <color>{kml_color}</color>
        <scale>0.3</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
    <Point>
      <coordinates>{point['longitude']},{point['latitude']},{point['altitude']}</coordinates>
    </Point>
  </Placemark>"""
                
                f.write(placemark)
            
            f.write(kml_footer)
        
        print(f"Exported KML file to {output_file}")
        return output_file
        
    def export_csv(self, output_file='points3D.csv'):
        """
        Export GPS points to CSV format.
        """
        gps_points = self.convert_points_to_gps()
        
        with open(os.path.join(self.workspace_dir, output_file), 'w') as f:
            # Write header
            f.write("point_id,latitude,longitude,altitude,r,g,b\n")
            
            # Write data rows
            for point_id, point in gps_points.items():
                r, g, b = point['rgb']
                f.write(f"{point_id},{point['latitude']},{point['longitude']},{point['altitude']},{r},{g},{b}\n")
        
        print(f"Exported CSV file to {output_file}")
        return output_file