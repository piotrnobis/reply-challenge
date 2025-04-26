import os
import cv2
import numpy as np
from inference import get_model
import supervision as sv

class Roboflow:
    def run_inference_on_images(self, model_id1, image_dir, show_img=False):
        # Get the API key (if set in environment variable)
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            print("Error: API Key is not set. Please set the ROBOFLOW_API_KEY environment variable.")
            return []

        # Load the pre-trained model
        model1 = get_model(model_id=model_id1)

        # List all image files in the directory
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No image files found in the specified directory.")
            return []

        all_detections = {}

        # Loop through each image in the directory
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"Running inference on {image_file}...")

            detections = self.run_pipeline(model1, image_path, show_img)

            if detections is not None:
                # Append detections with image filename info
                all_detections[image_file] = detections.xyxy.tolist()

        return all_detections


    def run_pipeline(self, model1, image_path, show_img):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}. Skipping...")
            return None
        
        results = model1.infer(image)[0]
        detections = sv.Detections.from_inference(results)
        print(f'Detections for {image_path}: {detections}')

        if show_img:
            sv.plot_image(image)

        return detections
    
    def blur_except_rectangles(self, all_detections, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # Go through images
        for filename in os.listdir(input_folder):
            if filename not in all_detections:
                continue  # No rectangles for this image, skip
            
            rects = all_detections[filename]
            if not rects:
                continue  # Empty list, skip
            
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {filename}")
                continue

            # Create blurred version
            """ blurred_img = cv2.GaussianBlur(img, (151, 151), 0)  # big kernel for strong blur

            # Create mask for the rectangles
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for rect in rects:
                x1, y1, x2, y2 = map(int, rect)
                mask[y1:y2, x1:x2] = 1  # set inside rectangle to 1

            # Combine: take original inside rectangles, blurred outside
            result = img.copy()
            result[mask == 0] = blurred_img[mask == 0]

            # Save output
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, result) """

            # Create a black image
            result = np.zeros_like(img)

            # Copy original image inside the rectangles
            for rect in rects:
                x1, y1, x2, y2 = map(int, rect)  # force integers
                # Clamp to image size to avoid out-of-bounds error
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
                result[y1:y2, x1:x2] = img[y1:y2, x1:x2]

            # Save output
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, result)