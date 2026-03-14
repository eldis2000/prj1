import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import urllib.request

# 설정
IMAGE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
IMAGE_PATH = "sample_detection.jpg"
OUTPUT_PATH = "detection_output.jpg"

def main():
    print("📥 샘플 이미지 다운로드 중...")
    if not os.path.exists(IMAGE_PATH):
        req = urllib.request.Request(IMAGE_URL, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req) as response, open(IMAGE_PATH, 'wb') as out_file:
            out_file.write(response.read())
    
    print("📷 이미지 로드 중...")
    # torchvision.io.read_image는 uint8 형태의 [C, H, W] 텐서를 반환합니다. bounding box 그리기에 적합합니다.
    img = read_image(IMAGE_PATH)
    
    print("로딩 모델: Faster R-CNN (MobileNet V3 Large 320 FPN)...")
    # 최신 가중치 로드
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    # 모델 초기화 (신뢰도 0.5 이상인 박스만 결과로 출력하도록 설정)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.5)
    
    # GPU 사용 가능시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 기기: {device}")
    model = model.to(device)
    model.eval()
    
    # 전처리 설정
    preprocess = weights.transforms()
    # 모델이 float 타입의 텐서를 입력으로 받습니다
    batch = [preprocess(img).to(device)]
    
    print("⚙️  객체 탐지 (Inference) 실행 중...")
    with torch.no_grad():
        prediction = model(batch)[0]
    
    num_detected = len(prediction['labels'])
    print(f"✅ 발견된 객체 수: {num_detected}개")
    
    if num_detected > 0:
        # COCO 데이터셋의 클래스 이름 가져오기
        categories = weights.meta["categories"]
        labels = [categories[i] for i in prediction["labels"]]
        scores = prediction["scores"]
        
        # 이미지에 그릴 라벨 텍스트 ('클래스: 정확도%')
        box_labels = [f"{label}: {score:.2f}" for label, score in zip(labels, scores)]
        
        for idx in range(num_detected):
            print(f"  - {box_labels[idx]}")
        
        # 바운딩 박스를 그리기 위해 CPU로 박스 이동
        boxes = prediction["boxes"].cpu()
        
        print("🎨 결과 시각화 및 박싱 처리 중...")
        # uint8 이미지(img) 위에 박스 그리기
        result_img_tensor = draw_bounding_boxes(
            img,
            boxes=boxes,
            labels=box_labels,
            colors="red",
            width=3
        )
        
        # Tensor -> PIL Image 변경
        output_img = to_pil_image(result_img_tensor)
        
        # 시스템 기본 이미지 뷰어로 띄우기 (팝업)
        output_img.show(title="Object Detection Result")
        
        # 파일로 저장하기
        output_img.save(OUTPUT_PATH)
        print(f"💾 결과 이미지가 저장되었습니다: {os.path.abspath(OUTPUT_PATH)}")
    else:
        print("객체를 찾을 수 없습니다.")

if __name__ == '__main__':
    main()
