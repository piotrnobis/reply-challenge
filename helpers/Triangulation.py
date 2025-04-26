import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import random

from mpl_toolkits.mplot3d import Axes3D
from . import Roboflow

class Triangulation:
    def __init__(self):
        self.roboflow = Roboflow()
    
    def load_images(self, img1_path, img2_path):
        """Load the two drone images."""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        return img1, img2, gray1, gray2


    def detect_and_match_features(self, gray1, gray2):
        """Detect SIFT features and match them between images."""
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # Use FLANN for faster matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Get the matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return kp1, kp2, good_matches, pts1, pts2

    def visualize_matches(self, img1, img2, kp1, kp2, good_matches):
        """Visualize the matched features between the two images."""
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches: {len(good_matches)} matched points')
        plt.axis('off')
        plt.show(block = True)

    def create_camera_matrices(self, intrinsic_matrix, location1, location2, rotation1=None, rotation2=None):
        """
        Create camera matrices from intrinsics and GPS locations.
        
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix
            location1, location2: GPS coordinates of drone (latitude, longitude, altitude)
            rotation1, rotation2: Optional rotation matrices (3x3) if available
        
        Returns:
            Two 3x4 camera projection matrices P1 and P2
        """
        # If rotation matrices are not provided, assume identity (no rotation)
        if rotation1 is None:
            rotation1 = np.eye(3)
        if rotation2 is None:
            rotation2 = np.eye(3)
        
        # Convert GPS coordinates to local Euclidean coordinates
        # This is a simplified conversion - in real-world applications, 
        # you'd use a proper GPS to Euclidean conversion
        earth_radius = 6371000  # in meters
        
        # Extract coordinates
        lat1, lon1, alt1 = location1
        lat2, lon2, alt2 = location2
        
        # Convert to radians
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        
        # Calculate differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Approximate conversion to meters (simplified)
        dx = earth_radius * dlon * np.cos((lat1 + lat2) / 2)
        dy = earth_radius * dlat
        dz = alt2 - alt1
        
        # First camera at origin (reference frame)
        t1 = np.zeros((3, 1))
        
        # Second camera position relative to first
        t2 = np.array([[dx], [dy], [dz]])
        
        # Create projection matrices P = K[R|t]
        P1 = intrinsic_matrix @ np.hstack((rotation1, t1))
        P2 = intrinsic_matrix @ np.hstack((rotation2, t2))
        
        return P1, P2

    def triangulate_points(self, P1, P2, pts1, pts2):
        """
        Triangulate 3D points from matched 2D points and camera matrices.
        
        Args:
            P1, P2: 3x4 camera projection matrices
            pts1, pts2: Nx2 arrays of matched points in the two images
        
        Returns:
            Nx3 array of 3D points
        """
        # Triangulate points using OpenCV's triangulatePoints function
        # Note: This function expects the points in homogeneous coordinates
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T  # Transpose to get Nx3 shape

    def plot_3d_points(self, points_3d, location1, location2):
        """Plot the triangulated 3D points and camera positions."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the 3D points
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o', label='Triangulated Points')
        
        # Plot the camera positions
        ax.scatter([0], [0], [0], c='r', marker='^', s=100, label='Camera 1')
        dx = location2[0] - location1[0]
        dy = location2[1] - location1[1]
        dz = location2[2] - location1[2]
        ax.scatter([dx], [dy], [dz], c='g', marker='^', s=100, label='Camera 2')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Triangulated Points and Camera Positions')
        ax.legend()
        
        plt.show(block = True)

    def convert_to_lat_lon_alt(self, points_3d, reference_location):
        """
        Convert local 3D coordinates back to latitude, longitude, altitude.
        
        Args:
            points_3d: Nx3 array of local 3D points
            reference_location: (latitude, longitude, altitude) of the reference point (first drone position)
        
        Returns:
            Nx3 array of (latitude, longitude, altitude) coordinates
        """
        # Earth parameters
        earth_radius = 6371000  # meters
        
        # Extract reference coordinates
        ref_lat, ref_lon, ref_alt = reference_location
        ref_lat_rad = np.radians(ref_lat)
        ref_lon_rad = np.radians(ref_lon)
        
        # Initialize result array
        n_points = points_3d.shape[0]
        lat_lon_alt = np.zeros((n_points, 3))
        
        for i in range(n_points):
            # Extract local coordinates
            x, y, z = points_3d[i]
            
            # Convert back to GPS coordinates
            # Longitude: x direction (east-west)
            dlon_rad = x / (earth_radius * np.cos(ref_lat_rad))
            lon = ref_lon + np.degrees(dlon_rad)
            
            # Latitude: y direction (north-south)
            dlat_rad = y / earth_radius
            lat = ref_lat + np.degrees(dlat_rad)
            
            # Altitude: z direction (up-down) - relative to reference altitude
            alt = ref_alt + z
            
            lat_lon_alt[i] = [lat, lon, alt]
        
        return lat_lon_alt
    

    def detect_barcodes(self, image_dir):
        model_id1 = "pile-of-crate-detection/12"
        model_id2 = "white-sheet-spotter/2"
        results = self.roboflow.run_inference_on_images(model_id1, model_id2, image_dir, show_img=False)

        return results
    

    def extract_one_feature_per_rectangle(self, important_rectangles, pts1, points_lat_lon_alt, img1):
        """
        Extract one feature from each important rectangle and find its 3D lat/long/alt location.
        
        Args:
            important_rectangles: List of lists in format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            pts1: Matched points from image 1 (Nx2 array)
            points_lat_lon_alt: Nx3 array with [latitude, longitude, altitude] for each matched point
            img1: The first image for visualization
            
        Returns:
            DataFrame with lat/long/alt for one point per rectangle
        """
        # Initialize lists to store results
        selected_pts = []
        selected_rect_indices = []
        selected_match_indices = []
        selected_coords = []
        
        # For each rectangle, find points inside it
        for i, rect in enumerate(important_rectangles):
            x1, y1, x2, y2 = rect  # Format: [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            
            # Find keypoints inside this rectangle
            inside_rect = []
            
            # Check each matched point
            for j, pt in enumerate(pts1):
                if (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2):
                    inside_rect.append(j)
            
            # If points are found inside the rectangle
            if inside_rect:
                # Randomly select one match from this rectangle
                selected_idx = random.choice(inside_rect)
                selected_match_indices.append(selected_idx)
                selected_rect_indices.append(i)
                selected_pts.append(pts1[selected_idx])
                selected_coords.append(points_lat_lon_alt[selected_idx])
        
        # Create DataFrame with results
        df_selected = pd.DataFrame({
            'Rectangle_Index': selected_rect_indices,
            'Match_Index': selected_match_indices,
            'Point_X': [pt[0] for pt in selected_pts],
            'Point_Y': [pt[1] for pt in selected_pts],
            'Latitude': [coord[0] for coord in selected_coords],
            'Longitude': [coord[1] for coord in selected_coords],
            'Altitude': [coord[2] for coord in selected_coords]
        })
        
        # Create a single visualization showing the selected points on the image
        if len(selected_pts) > 0:
            plt.figure(figsize=(12, 8))
            plt.imshow(img1)
            
            # Draw all important rectangles
            for i, rect in enumerate(important_rectangles):
                x1, y1, x2, y2 = rect
                width = x2 - x1
                height = y2 - y1
                plt.gca().add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor='blue', linewidth=2))
                plt.text(x1, y1-10, f"Rect {i}", color='blue', fontsize=10)
            
            # Draw selected points
            for i, pt in enumerate(selected_pts):
                plt.scatter(pt[0], pt[1], c='red', s=80, marker='x')
                rect_idx = selected_rect_indices[i]
                plt.text(pt[0]+5, pt[1]+5, f"R{rect_idx}", color='red', fontsize=10)
            
            plt.title("Selected Feature Points (One per Rectangle)")
            plt.show(block = True)
        
        return df_selected
