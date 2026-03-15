import ollama
import sys

# Windows 환경에서 한글 깨짐 방지
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def call_ollama_stream(prompt, model="qwen3.5:0.8b"):
    """
    Ollama 라이브러리를 사용하여 모델의 응답을 스트리밍 방식으로 가져옵니다.
    """
    try:
        # stream=True 설정을 통해 제너레이터 반환
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )
        return stream
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    print(f"--- Ollama 스트리밍 대화 모드 (qwen3.5:0.8b) ---")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("-" * 30)
    
    while True:
        try:
            # 사용자로부터 질문 입력 받기
            user_input = input("질문: ")
            
            # 종료 조건
            if user_input.lower() in ['exit', 'quit']:
                print("대화를 종료합니다.")
                break
            
            if not user_input.strip():
                continue
                
            print("모델 답변: ", end="", flush=True)
            
            # 스트리밍 호출
            stream = call_ollama_stream(user_input)
            
            if stream:
                for chunk in stream:
                    # 각 청크의 텍스트 조각 출력
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                print("\n" + "-" * 30)
            else:
                print("응답을 가져올 수 없습니다.")
            
        except KeyboardInterrupt:
            print("\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n실행 중 오류 발생: {e}")
