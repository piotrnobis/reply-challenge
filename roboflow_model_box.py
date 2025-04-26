import os
import cv2
from inference import get_model
import supervision as sv

def run_inference_on_images(model_id1, image_dir, show_img=False):
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

    all_detections = []

    # Loop through each image in the directory
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"Running inference on {image_file}...")

        detections = run_pipeline(model1, image_path, show_img)

        if detections is not None:
            # Append detections with image filename info
            all_detections.append({
                "image_file": image_file,
                "detections": detections.xyxy.tolist()  # Convert to list for easy handling
            })

    return all_detections

def run_pipeline(model1, image_path, show_img):
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


if __name__ == "__main__":
    model_id1 = "pile-of-crate-detection/10"  # Replace with your model ID
    image_dir = "dev_data_demo"  # Your images folder

    all_detections = run_inference_on_images(model_id1, image_dir, show_img=False)

    # Example: print the final result
    print("\nFinal Detections:")
    for item in all_detections:
        print(f"Image: {item['image_file']}")
        print("Boxes:", item['detections'])


    # Format
    # "image_file" → the image filename (e.g., "img1.jpg")
    # "detections" → a list of bounding boxes, and each bounding box is a list [x1, y1, x2, y2]
