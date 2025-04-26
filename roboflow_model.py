import os
import cv2
import numpy as np
from inference import get_model
import supervision as sv

def run_inference_on_images(model_id1, model_id2, image_dir, show_img=False):
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

    all_results = []  # <-- Collect all dictionaries here

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Running inference on {image_file}...")

        result = run_pipeline(model1, model2, image_path, output_folder, show_img)
        all_results.append(result)

    return all_results   # <-- Return the list of results!

def run_pipeline(model1, model2, image_path, output_folder, show_img):
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

    cropped_images, crop_boxes = crop_image_by_detections(image, detections1)

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

def crop_image_by_detections(image, detections):
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

if __name__ == "__main__":
    model_id1 = "pile-of-crate-detection/10"
    model_id2 = "white-sheet-spotter/2"
    image_dir = "dev_data_demo"
    results = run_inference_on_images(model_id1, model_id2, image_dir, show_img=False)

    print("Summary of results:")
    for res in results:
        print(res)
