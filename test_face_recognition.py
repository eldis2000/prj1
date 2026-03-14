import argparse
import os
import requests
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

def download_image(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            headers = {'User-Agent': 'FaceRecognitionBot/1.0 (testuser@localhost.localdomain) python-requests/2.31'}
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None
    return filename

def get_default_images():
    # Brad Pitt
    img1_url = "https://upload.wikimedia.org/wikipedia/commons/4/4c/Brad_Pitt_2019_by_Glenn_Francis.jpg"
    img2_url = "https://upload.wikimedia.org/wikipedia/commons/5/51/Brad_Pitt_Fury_2014.jpg"
    # Tom Cruise
    img3_url = "https://upload.wikimedia.org/wikipedia/commons/3/33/Tom_Cruise_by_Gage_Skidmore_2.jpg"
    
    img1_path = download_image(img1_url, "pitt1.jpg")
    img2_path = download_image(img2_url, "pitt2.jpg")
    img3_path = download_image(img3_url, "cruise1.jpg")
    
    return img1_path, img2_path, img3_path

def compute_similarity(embedding1, embedding2):
    # Calculate cosine similarity using torch
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

def main():
    parser = argparse.ArgumentParser(description="Test Face Recognition using facenet-pytorch")
    parser.add_argument("--img1", type=str, help="Path to the first image")
    parser.add_argument("--img2", type=str, help="Path to the second image")
    # Facenet cosine similarity threshold is often ~ 0.5 - 0.7 for determining same identity
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for recognizing same person")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing MTCNN and InceptionResnetV1...")
    # Initialize MTCNN for face detection/alignment
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
    # Initialize InceptionResnetV1 for recognition
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not args.img1 or not args.img2:
        print("\nImage paths not provided. Running default tests with downloaded celebrity images...")
        imgpaths = get_default_images()
        if None in imgpaths:
            print("Failed to download one or more default images. Exiting.")
            return
            
        img1_path, img2_path, img3_path = imgpaths
        
        test_pairs = [
            ("Same Person (Brad Pitt)", img1_path, img2_path),
            ("Different Persons (Brad Pitt vs Tom Cruise)", img1_path, img3_path)
        ]
    else:
        test_pairs = [("User Input", args.img1, args.img2)]

    for test_name, path1, path2 in test_pairs:
        print(f"\n--- Testing: {test_name} ---")
        print(f"Image 1: {path1}")
        print(f"Image 2: {path2}")
        
        try:
            img1 = Image.open(path1).convert('RGB')
            img2 = Image.open(path2).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            continue

        print("Detecting and aligning faces...")
        # Get cropped and prewhitened image tensor
        img1_cropped = mtcnn(img1)
        img2_cropped = mtcnn(img2)

        if img1_cropped is None:
            print(f"Error: No face detected in {path1}")
            continue
        if img2_cropped is None:
            print(f"Error: No face detected in {path2}")
            continue

        print("Calculating embeddings...")
        # Calculate embeddings (add batch dimension)
        img1_cropped = img1_cropped.unsqueeze(0).to(device)
        img2_cropped = img2_cropped.unsqueeze(0).to(device)
        
        with torch.no_grad():
            img1_embedding = resnet(img1_cropped)
            img2_embedding = resnet(img2_cropped)
            
        similarity = compute_similarity(img1_embedding, img2_embedding)
        
        is_same = similarity > args.threshold
        
        print(f"Calculated Cosine Similarity: {similarity:.4f}")
        print(f"Threshold: {args.threshold}")
        print(f"Same Person Prediction: {is_same}")
        
if __name__ == '__main__':
    main()
