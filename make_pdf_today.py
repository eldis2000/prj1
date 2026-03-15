import os
import datetime
from PIL import Image

def create_pdf_from_today_screenshots():
    # 1. 설정
    screenshot_dir = r'C:\Users\804\Pictures\Screenshots'
    output_pdf = f"screenshots_{datetime.date.today().strftime('%Y%m%d')}.pdf"
    
    today = datetime.date.today()
    print(f"오늘 날짜: {today}")
    
    # 2. 오늘 생성/수정된 이미지 파일 찾기
    image_files = []
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    if not os.path.exists(screenshot_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {screenshot_dir}")
        return

    for filename in os.listdir(screenshot_dir):
        if filename.lower().endswith(extensions):
            filepath = os.path.join(screenshot_dir, filename)
            # 수정 시간 확인
            mtime = datetime.date.fromtimestamp(os.path.getmtime(filepath))
            if mtime == today:
                image_files.append(filepath)
    
    if not image_files:
        print("오늘 생성된 이미지가 없습니다.")
        return
    
    # 시간순으로 정렬
    image_files.sort(key=os.path.getmtime)
    print(f"발견된 이미지 수: {len(image_files)}")
    
    # 3. PDF로 변환
    images = []
    for f in image_files:
        try:
            img = Image.open(f)
            # PDF는 RGB 모드여야 함 (PNG 등의 RGBA 대응)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"파일 열기 실패 ({f}): {e}")

    if images:
        # 첫 번째 이미지를 기준으로 나머지를 append하여 저장
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"성공! PDF가 생성되었습니다: {os.path.abspath(output_pdf)}")
    else:
        print("변환할 수 있는 유효한 이미지가 없습니다.")

if __name__ == "__main__":
    create_pdf_from_today_screenshots()
