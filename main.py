import os
import io
import ssl
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

# facenet
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

# easyocr
import easyocr

# yolo
from ultralytics import YOLO

# transformers
from transformers import pipeline

# SSL 인증서 검증 비활성화 (일부 모델 가중치/폰트 다운로드 용도)
ssl._create_default_https_context = ssl._create_unverified_context

# 전역 모델 사전
ml_models = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return [line.decode('utf-8').strip() for line in response.readlines()]
    except Exception as e:
        print(f"Warning: Failed to load ImageNet classes: {e}")
        return [f"class_{i}" for i in range(1000)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models into memory...")
    
    # 1. Classification Model
    print("Loading MobileNet V3 Small...")
    classify_weights = models.MobileNet_V3_Small_Weights.DEFAULT
    classify_model = models.mobilenet_v3_small(weights=classify_weights).to(device)
    classify_model.eval()
    ml_models["classifier"] = classify_model
    ml_models["classifier_preprocess"] = classify_weights.transforms()
    ml_models["imagenet_classes"] = load_imagenet_classes()

    # 2. Face Recognition Model
    print("Loading MTCNN and InceptionResnetV1...")
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    ml_models["mtcnn"] = mtcnn
    ml_models["facenet"] = resnet

    # 3. Object Detection Model
    print("Loading Faster R-CNN (MobileNet V3 Large 320 FPN)...")
    det_weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    det_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=det_weights, box_score_thresh=0.5).to(device)
    det_model.eval()
    ml_models["detector"] = det_model
    ml_models["detector_preprocess"] = det_weights.transforms()
    ml_models["coco_categories"] = det_weights.meta["categories"]

    # 4. OCR Model
    print("Loading EasyOCR...")
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    ml_models["ocr"] = reader

    # 7. Cat/Dog Classification Model
    print("Loading Cat/Dog ViT model...")
    cat_dog_classifier = pipeline(
        "image-classification",
        model="akahana/vit-base-cats-vs-dogs",
        device=0 if torch.cuda.is_available() else -1
    )
    ml_models["cat_dog_classifier"] = cat_dog_classifier
    
    print("All models loaded successfully!")
    yield
    print("Cleaning up models...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="Deep Learning Integration API", description="5가지 딥러닝 모델 통합 서버")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doorbell", response_class=HTMLResponse)
async def doorbell_page(request: Request):
    return templates.TemplateResponse("doorbell.html", {"request": request})

@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/detect-ui")
async def detect_ui(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # 원본 이미지 (drawing용)
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 모델 입력용 텐서
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = ml_models["detector"]
    preprocess = ml_models["detector_preprocess"]
    categories = ml_models["coco_categories"]
    
    batch = [preprocess(img_tensor).to(device)]

    with torch.no_grad():
        prediction = model(batch)[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    detections = []
    for i in range(len(labels)):
        score = float(scores[i])
        if score < 0.5: continue
        
        box = boxes[i].astype(int)
        label = categories[labels[i]]
        detections.append({
            "class": label,
            "score": score,
            "box": box.tolist()
        })
        
        # Draw on image
        cv2.rectangle(img_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img_cv2, label_text, (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result image
    result_path = "static/result.jpg"
    cv2.imwrite(result_path, img_cv2)

    return {
        "filename": file.filename, 
        "detections": detections,
        "image_url": "/static/result.jpg"
    }

@app.get("/api/v1")
def read_api_root():
    return {"message": "Welcome to DL models API. See /docs for Swagger UI"}

@app.post("/api/v1/classify", summary="이미지 분류", description="MobileNet V3 Small 모델을 이용한 이미지 분류")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    preprocess = ml_models["classifier_preprocess"]
    model = ml_models["classifier"]
    classes = ml_models["imagenet_classes"]

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        idx = top_catid[i].item()
        results.append({
            "rank": i + 1,
            "class": classes[idx],
            "probability": round(top_prob[i].item() * 100, 2)
        })

    return {"filename": file.filename, "predictions": results}

@app.post("/api/v1/classify-cat-dog", summary="개/고양이 분류", description="ViT 모델을 이용한 개/고양이 이진 분류")
async def classify_cat_dog(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    classifier = ml_models["cat_dog_classifier"]
    
    # HuggingFace pipeline image-classification output format:
    # [{'label': 'LABEL_0', 'score': 0.99}, {'label': 'LABEL_1', 'score': 0.01}]
    # Based on the model akahana/vit-base-cats-vs-dogs: 
    # Usually 0=cat, 1=dog or similar. The model card says labels are 'cat' and 'dog'.
    
    results = classifier(image)
    
    # Sort by score descending (already sorted by default in pipeline)
    return {
        "filename": file.filename,
        "predictions": results
    }

@app.get("/api/v1/batch-classify", summary="일괄 분류", description="img/cat, img/dog 폴더의 모든 이미지를 일괄 분류")
async def batch_classify():
    import time
    base_dir = "img"
    results = []
    
    classifier = ml_models["cat_dog_classifier"]
    
    for category in ["cat", "dog"]:
        folder_path = os.path.join(base_dir, category)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_path = os.path.join(folder_path, filename)
                try:
                    image = Image.open(file_path).convert("RGB")
                    
                    start_time = time.time()
                    predictions = classifier(image)
                    end_time = time.time()
                    
                    inference_time = round((end_time - start_time) * 1000, 2) # ms
                    
                    # Top prediction
                    top_pred = predictions[0]
                    label = top_pred['label']
                    score = top_pred['score']
                    
                    # Result mapping for table
                    display_label = "고양이 🐱" if label == "cat" else ("강아지 🐶" if label == "dog" else label)
                    
                    results.append({
                        "filename": f"{category}/{filename}",
                        "label": display_label,
                        "score": f"{round(score * 100, 2)}%",
                        "time": f"{inference_time}ms"
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return {"results": results}

@app.post("/api/v1/face-recognize", summary="얼굴 비교 (얼굴 인식)", description="MTCNN & InceptionResnetV1을 이용한 동일 인물 여부 확인")
async def face_recognize(image1: UploadFile = File(...), image2: UploadFile = File(...), threshold: float = Form(0.6)):
    try:
        img1 = Image.open(io.BytesIO(await image1.read())).convert("RGB")
        img2 = Image.open(io.BytesIO(await image2.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file(s)")

    mtcnn = ml_models["mtcnn"]
    resnet = ml_models["facenet"]

    img1_cropped = mtcnn(img1)
    img2_cropped = mtcnn(img2)

    if img1_cropped is None:
        raise HTTPException(status_code=400, detail="No face detected in image1")
    if img2_cropped is None:
        raise HTTPException(status_code=400, detail="No face detected in image2")

    img1_cropped = img1_cropped.unsqueeze(0).to(device)
    img2_cropped = img2_cropped.unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = resnet(img1_cropped)
        emb2 = resnet(img2_cropped)

    similarity = round(F.cosine_similarity(emb1, emb2).item(), 4)
    is_same = similarity > threshold

    return {
        "similarity": similarity,
        "is_same_person": is_same,
        "threshold": threshold
    }

@app.post("/api/v1/detect-objects", summary="객체 탐지", description="Faster R-CNN (MobileNet V3)을 이용한 객체 탐지")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # [C, H, W] uint8 형태의 텐서 구조 생성 (테스트 스크립트 기반)
        img_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = ml_models["detector"]
    preprocess = ml_models["detector_preprocess"]
    categories = ml_models["coco_categories"]
    
    # 전처리 적용 시 부동소수점 타입 변환 등의 작업 수행
    batch = [preprocess(img_tensor).to(device)]

    with torch.no_grad():
        prediction = model(batch)[0]

    num_detected = len(prediction['labels'])
    results = []
    
    boxes = prediction["boxes"].cpu().numpy().tolist()
    labels = prediction["labels"].cpu().numpy().tolist()
    scores = prediction["scores"].cpu().numpy().tolist()

    for i in range(num_detected):
        results.append({
            "class": categories[labels[i]],
            "score": round(scores[i], 4),
            "box": [round(b, 2) for b in boxes[i]] # [xmin, ymin, xmax, ymax]
        })

    return {"filename": file.filename, "objects_detected": num_detected, "detections": results}

@app.post("/api/v1/ocr", summary="문자 인식 (OCR)", description="EasyOCR 모듈을 이용한 한/영 문자 인식")
async def ocr_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    reader = ml_models["ocr"]
    raw_results = reader.readtext(img_np)
    
    results = []
    for (bbox, text, prob) in raw_results:
        # JSON 직렬화를 위해 float 형으로 변환
        json_bbox = [[float(coord[0]), float(coord[1])] for coord in bbox]
        results.append({
            "text": text,
            "confidence": round(float(prob), 4),
            "bounding_box": json_bbox
        })

    return {"filename": file.filename, "detected_texts": results}

@app.post("/api/v1/analyze-sentiment", summary="감정 분석", description="KoELECTRA 모델을 이용한 한국어 문장 감정 분석 (긍정/부정)")
async def analyze_sentiment(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    analyzer = ml_models["sentiment_analyzer"]
    # pipeline 특성상 단일 텍스트도 리스트로 처리 가능
    result = analyzer(text)[0]
    
    # 모델 출력 레이블 변환 (daekeun-ml/koelectra-small-v3-nsmc는 '0': 부정, '1': 긍정)
    label_map = {"0": "Negative", "1": "Positive", "LABEL_0": "Negative", "LABEL_1": "Positive"}
    label = label_map.get(result["label"], result["label"])
    
    return {
        "text": text,
        "sentiment": label,
        "confidence": round(result["score"], 4)
    }

@app.post("/api/v1/doorbell-analyze", summary="지능형 현관 보안 분석", description="방문자 식별(얼굴 인식) 및 택배 감지(객체 탐지) 통합 분석")
async def doorbell_analyze(file: UploadFile = File(...), threshold: float = Form(0.6)):
    try:
        contents = await file.read()
        # 원본 이미지 (분석 및 시각화용)
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 1. 객체 탐지 (사람 및 박스)
    detector = ml_models["detector"]
    preprocess = ml_models["detector_preprocess"]
    categories = ml_models["coco_categories"]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    batch = [preprocess(img_tensor).to(device)]

    with torch.no_grad():
        prediction = detector(batch)[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    has_person = False
    has_package = False
    
    for i in range(len(labels)):
        if scores[i] < 0.5: continue
        label = categories[labels[i]]
        if label == 'person':
            has_person = True
        if label in ['handbag', 'suitcase', 'backpack', 'bottle']: # MobileNet V3 COCO에는 'box'가 명확하지 않을 수 있어 유사 객체 포함
             has_package = True
        # 시각화
        box = boxes[i].astype(int)
        cv2.rectangle(img_cv2, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(img_cv2, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 2. 얼굴 인식 (가족 여부 확인 - family.jpg가 있을 경우)
    access_granted = False
    similarity_score = 0.0
    family_image_path = "static/family.jpg"
    
    if has_person and os.path.exists(family_image_path):
        mtcnn = ml_models["mtcnn"]
        resnet = ml_models["facenet"]
        
        try:
            family_img = Image.open(family_image_path).convert("RGB")
            visitor_face = mtcnn(img_pil)
            family_face = mtcnn(family_img)
            
            if visitor_face is not None and family_face is not None:
                visitor_face = visitor_face.unsqueeze(0).to(device)
                family_face = family_face.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb1 = resnet(visitor_face)
                    emb2 = resnet(family_face)
                
                similarity_score = float(F.cosine_similarity(emb1, emb2).item())
                if similarity_score > threshold:
                    access_granted = True
        except Exception as e:
            print(f"Face recognition error: {e}")

    # 결과 이미지 저장
    result_path = "static/doorbell_result.jpg"
    cv2.imwrite(result_path, img_cv2)

    return {
        "has_person": has_person,
        "has_package": has_package,
        "access_granted": access_granted,
        "similarity": round(similarity_score, 4),
        "image_url": "/static/doorbell_result.jpg"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
