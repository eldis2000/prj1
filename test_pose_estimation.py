import argparse
import urllib.request
import os
import cv2
from ultralytics import YOLO

def download_default_image(filename="default_person.jpg"):
    """Downloads a default image of a person if not available."""
    if not os.path.exists(filename):
        print(f"Downloading default image to {filename}...")
        # A standard test image with people from ultralytics
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
        try:
            # Adding User-Agent to avoid 403 Forbidden
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    return filename

def main():
    parser = argparse.ArgumentParser(description="Test Lightweight Pose Estimation (YOLOv8n-pose)")
    parser.add_argument("--image", type=str, help="Path to the input image. If not provided, a default image will be downloaded.", default=None)
    args = parser.parse_args()

    # 1. Get the image
    image_path = args.image
    if image_path is None:
        image_path = download_default_image()
        if image_path is None:
            print("Failed to get an image to process. Exiting.")
            return
    elif not os.path.exists(image_path):
        print(f"Error: Provided image path '{image_path}' does not exist.")
        return

    print(f"Processing image: {image_path}")

    # 2. Load the model
    # YOLOv8n-pose is the nano version of YOLOv8 for pose estimation. It is very fast and lightweight.
    print("Loading YOLOv8n-pose model...")
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have internet connection for the first run to download the model weights.")
        return

    # 3. Perform inference
    print("Running inference...")
    # conf: Confidence threshold
    results = model(image_path, conf=0.3)

    # 4. Save/Visualize the result
    if len(results) > 0:
        result = results[0]
        # result.plot() returns a numpy array representing the image with annotations
        annotated_image = result.plot()
        
        output_path = "pose_estimation_result.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\nSuccess! Inference complete.")
        print(f"Saved visualization to: {os.path.abspath(output_path)}")
        
        # Print how many people were detected
        if result.keypoints is not None:
             # shape is typically (num_persons, num_keypoints, x_y_confidence)
             num_persons = result.keypoints.shape[0] if len(result.keypoints.shape) > 0 else 0
             print(f"Detected {num_persons} person(s).")
    else:
        print("No results returned from the model.")

if __name__ == "__main__":
    main()
