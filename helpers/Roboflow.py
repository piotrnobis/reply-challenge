import os
import cv2
import numpy as np
from inference import get_model
import supervision as sv

class Roboflow:
    def run_inference_on_images(self, model_id1, model_id2, image_dir, show_img=False):
        # Get the API key (if set in environment variable)
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            print("Error: API Key is not set. Please set the ROBOFLOW_API_KEY environment variable.")
            return

        model1 = get_model(model_id=model_id1)
        model2 = get_model(model_id=model_id2)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No image files found in the specified directory.")
            return

        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        all_results = {}  # <-- Collect all dictionaries here

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"Running inference on {image_file}...")

            result = self.run_pipeline(model1, model2, image_path, output_folder, show_img)
            all_results[result['file_name']] = {
                'pallets': result['pallets'] if 'pallets' in result else [],
                'white_spaces': result['white_spaces'] if 'white_spaces' in result else [],
            }

        return all_results   # <-- Return the list of results!


    def run_pipeline(self, model1, model2, image_path, output_folder, show_img):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}. Skipping...")
            return

        results_pallets = model1.infer(image)[0]
        detections1 = sv.Detections.from_inference(results_pallets)
        # print(f'Detections for {image_path} (model1):', detections1)

        if show_img:
            sv.plot_image(image)

        # Save pallet boxes
        pallet_boxes = []
        for idx, box in enumerate(detections1.xyxy):
            x1, y1, x2, y2 = map(int, box)
            pallet_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            label = f"Pallet"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)

        cropped_images, crop_boxes = self.crop_image_by_detections(image, detections1)

        all_mapped_detections = []

        for idx, (cropped_img, crop_box) in enumerate(zip(cropped_images, crop_boxes)):
            x1, y1, _, _ = crop_box

            results_segmentation = model2.infer(cropped_img)[0]
            detections2 = sv.Detections.from_inference(results_segmentation)
            # print(f"Segmentation Detections for crop {idx}:", detections2)

            for det_idx in range(len(detections2.xyxy)):
                det_box = detections2.xyxy[det_idx]
                det_x1, det_y1, det_x2, det_y2 = map(int, det_box)
                mapped_box = [det_x1 + x1, det_y1 + y1, det_x2 + x1, det_y2 + y1]

                mapped_detection = {
                    "bbox": mapped_box,
                    "confidence": float(detections2.confidence[det_idx]),
                    "class_name": detections2.data["class_name"][det_idx],
                }
                all_mapped_detections.append(mapped_detection)

        white_space_boxes = []

        for detection in all_mapped_detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

            white_space_boxes.append([x1, y1, x2, y2])

        # Save final image
        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_filename.replace(".jpg", "_final.jpg"))
        cv2.imwrite(output_path, image)
        print(f"Saved final annotated image: {output_path}")

        if show_img:
            sv.plot_image(image)

        # --- Return dictionary for this image
        return {
            "file_name": base_filename,
            "pallets": pallet_boxes,
            "white_spaces": white_space_boxes
        }
    

    def crop_image_by_detections(self, image, detections):
        cropped_images = []
        crop_boxes = []

        for idx, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            crop_boxes.append((x1, y1, x2, y2))

        return cropped_images, crop_boxes
    

    def blur_except_rectangles(self, all_detections, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # Go through images
        for filename in os.listdir(input_folder):
            if filename not in all_detections:
                continue  # No rectangles for this image, skip
            
            rects = all_detections[filename]['pallets']
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