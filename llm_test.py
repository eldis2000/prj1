import os
import argparse
from google import genai
from dotenv import load_dotenv

def main():
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    # API 키 가져오기
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
        return

    # 인자값 설정
    parser = argparse.ArgumentParser(description="Gemini LLM(SDK 2.0)을 이용한 텍스트 긍정/부정 분류기")
    parser.add_argument(
        "text", 
        nargs="?", 
        default="오늘 날씨가 정말 화창해서 기분이 너무 좋아요!", 
        help="분석할 텍스트"
    )
    args = parser.parse_args()

    # 최신 SDK 클라이언트 설정
    client = genai.Client(api_key=api_key)
    
    # 모델 설정: '2.0-flash-lite'에서 할당량 문제가 발생하므로, 
    # 더 범용적인 'gemini-2.0-flash' 또는 'gemini-1.5-flash'를 사용합니다.
    model_name = "gemini-3-flash-preview" 
    
    try:
        # 분석 수행
        response = client.models.generate_content(
            model=model_name,
            contents=args.text,
            config={
                "system_instruction": "당신은 텍스트의 감정을 분석하는 전문가입니다. 입력된 텍스트가 '긍정'인지 '부정'인지 분석하여 단어 하나로만 답변하세요. 판단이 어려우면 '중립'이라고 답변하세요.",
                "temperature": 0.1,
            }
        )
        
        print("\n" + "="*50)
        print(f"입력 텍스트: {args.text}")
        print(f"사용 모델: {model_name}")
        print("-" * 50)
        
        result = response.text.strip()
        print(f"감정 분석 결과: {result}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print("\n[해결 방법]")
        print("1. .env 파일의 API 키가 유효한지 확인하세요.")
        print("2. Google AI Studio에서 해당 모델에 대한 할당량(Quota)을 확인하세요.")
        print("3. 'gemini-2.0-flash-lite' 대신 'gemini-1.5-flash'로 시도해볼 수 있습니다.")

if __name__ == "__main__":
    main()
