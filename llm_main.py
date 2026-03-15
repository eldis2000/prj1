import os
import random
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from google import genai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Wife's Trap Question")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# Gemini 클라이언트 설정
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"  # 사용 가능한 2.0 모델로 변경

# 와이프의 질문 리스트 (LLM이 생성하기 전 기본값)
DEFAULT_QUESTIONS = [
    "자기야, 나 오늘 뭐 바뀐 거 없어?",
    "저 연예인이 예뻐, 내가 예뻐? 솔직하게 말해봐.",
    "나 요즘 살찐 거 같지 않아? (사실 1kg 쪘음)",
    "만약에 내가 바퀴벌레로 변하면 어떻게 할 거야?",
    "자기야, 내 친구 지혜 어때? 좀 괜찮지?"
]

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("llm_index.html", {"request": request})

@app.post("/get-question")
async def get_question():
    try:
        prompt = "아내(와이프) 입장에서 남편에게 던지는, 답변하기 매우 곤란하고 애매한 질문을 하나만 만들어줘. 예: '나 오늘 바뀐 거 없어?', '내가 예뻐 쟤가 예뻐?'. 짧고 강렬하게 한국어로 질문만 출력해."
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"temperature": 1.0}
        )
        question = response.text.strip()
    except Exception as e:
        question = random.choice(DEFAULT_QUESTIONS)
    
    return {"question": question}

@app.post("/evaluate")
async def evaluate(question: str = Form(...), answer: str = Form(...)):
    try:
        prompt = f"""
        당신은 까칠하지만 유머러스한 '와이프'입니다. 
        당신의 질문: "{question}"
        남편의 답변: "{answer}"
        
        이 답변을 평가해주세요.
        1. 점수: 0점에서 10점 사이 (정수)
        2. 이유: 왜 그 점수인지 와이프 말투(~해, ~야 등 반말/친근함/까칠함 섞인 말투)로 설명해주세요.
        
        출력 형식(JSON 형태처럼 구분해주세요):
        SCORE: [점수]
        REASON: [설명]
        """
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"temperature": 0.8}
        )
        
        full_text = response.text.strip()
        
        # 파싱
        score = 0
        reason = "말이 안 나와..."
        
        for line in full_text.split('\n'):
            if line.startswith("SCORE:"):
                score = line.replace("SCORE:", "").strip()
            if line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        return {"score": score, "reason": reason}
    except Exception as e:
        return {"score": 0, "reason": f"에러 났어! (오류: {str(e)})"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
