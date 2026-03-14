import os
import ssl
import cv2
import easyocr
import requests
from PIL import ImageFont, ImageDraw, Image

# Disable SSL verification for downloading resources
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # 1. Download a Korean font (NanumGothic) for visualization and image generation
    font_url = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
    font_path = 'NanumGothic.ttf'
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if not os.path.exists(font_path) or os.path.getsize(font_path) < 1000:
        print("Downloading NanumGothic font...")
        response = requests.get(font_url, headers=headers)
        with open(font_path, 'wb') as f:
            f.write(response.content)
        print("Font downloaded successfully.")

    # 2. Download a sample Korean image from EasyOCR official repo
    image_url = 'https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/korean.png'
    image_path = 'korean_test_image.png'
    
    if not os.path.exists(image_path) or os.path.getsize(image_path) < 1000:
        print(f"Downloading sample image from {image_url}...")
        response = requests.get(image_url, headers=headers)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        print("Image downloaded successfully.")

    # 2. Download a Korean font (NanumGothic) for visualization
    font_url = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
    font_path = 'NanumGothic.ttf'
    
    if not os.path.exists(font_path) or os.path.getsize(font_path) < 1000:
        print("Downloading NanumGothic font...")
        response = requests.get(font_url, headers=headers)
        with open(font_path, 'wb') as f:
            f.write(response.content)
        print("Font downloaded successfully.")

    # 3. Initialize EasyOCR
    print("Initializing EasyOCR reader with Korean ('ko') and English ('en')...")
    # Setting gpu=True will use GPU if available, otherwise it falls back to CPU
    reader = easyocr.Reader(['ko', 'en'], gpu=True)

    # 4. Perform OCR
    print("Performing OCR on the image...")
    results = reader.readtext(image_path)
    
    # 5. Visualization Preparation
    # We use OpenCV to read the image, then convert to PIL Image for proper Korean text rendering
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image at {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        # Load the downloaded font (size 32 is arbitrary, can be adjusted)
        font = ImageFont.truetype(font_path, 32)
    except IOError:
        print("Failed to load custom font. Using default.")
        font = ImageFont.load_default()

    print("\n--- OCR Results ---")
    for (bbox, text, prob) in results:
        print(f"Detected Text: [{text}] (Confidence: {prob:.4f})")
        
        # Extract coordinates
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        
        # Draw bounding box
        draw.rectangle([tl, br], outline="red", width=3)
        
        # Draw text slightly above the bounding box
        # Include background for text to make it more readable
        text_bbox = draw.textbbox((tl[0], tl[1] - 35), f"{text} ({prob:.2f})", font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text((tl[0], tl[1] - 35), f"{text} ({prob:.2f})", font=font, fill="yellow")

    # 6. Save visualization
    output_path = 'ocr_result.jpg'
    pil_image.save(output_path)
    print(f"\nSaved visualized result to {output_path}")

if __name__ == "__main__":
    main()
