import requests
import json

def test_sentiment():
    url = "http://localhost:8005/api/v1/analyze-sentiment"
    
    test_cases = [
        "이 영화 정말 재미있어요! 추천합니다.",
        "정말 별로예요. 시간 낭비했습니다.",
        "그냥 그래요. 볼만은 한데 지루함도 있네요.",
        "완전 대박! 너무 감동적이에요."
    ]
    
    print(f"Testing Sentiment Analysis API at {url}\n")
    
    for text in test_cases:
        payload = {"text": text}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"Text: {result['text']}")
                print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
                print("-" * 30)
            else:
                print(f"Error for '{text}': {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection failed: {e}")
            break

if __name__ == "__main__":
    # Note: Make sure the server is running on port 8005 before running this script
    test_sentiment()
