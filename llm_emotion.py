import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from google import genai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(title="재치만점 삼행시 생성기")

# 템플릿 및 정적 파일 경로 설정
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Gemini API 설정
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
model_name = "gemini-2.5-flash"  # 1.5-flash가 무료 티어 할당량이 더 넉넉한 경우가 많습니다.

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_emotion.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_poem(request: Request, word: str = Form(...)):
    if not word or len(word) < 2:
        return templates.TemplateResponse("index_emotion.html", {
            "request": request, 
            "error": "두 글자 이상의 단어를 입력해주세요.",
            "word": word
        })

    try:
        # 삼행시 생성을 위한 프롬프트 구성
        prompt = f"""
        입력된 단어 '{word}'로 삼행시(또는 단어 글자 수에 맞는 n행시)를 지어주세요.
        조건:
        1. 각 행은 반드시 단어의 각 글자로 시작해야 합니다.
        2. 매우 재치 있고, 창의적이며, 사람들이 웃을 수 있는 재미있는 내용이어야 합니다.
        3. 약간의 아재 개그나 반전이 있어도 좋습니다.
        4. 결과 형식은 다음과 같이 한 줄씩 출력해주세요:
        [글자]: [내용]
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "system_instruction": "당신은 세상에서 가장 웃기고 재치 있는 삼행시 장인입니다. 한국어로 답변하세요.",
                "temperature": 0.8,
            }
        )
        
        poem_result = response.text.strip()
        # 줄바꿈 정제
        poem_lines = [line.strip() for line in poem_result.split('\n') if line.strip()]

        return templates.TemplateResponse("index_emotion.html", {
            "request": request, 
            "word": word, 
            "poem_lines": poem_lines
        })

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "Gemini API 사용 한도(Quota)를 초과했습니다. 잠시 후 다시 시도하거나 다른 API 키를 사용해주세요."
        return templates.TemplateResponse("index_emotion.html", {
            "request": request, 
            "word": word, 
            "error": f"오류가 발생했습니다: {error_msg}"
        })

@app.get("/api/poem")
async def api_poem(word: str):
    """API 형태로 삼행시를 제공하는 엔드포인트"""
    try:
        prompt = f"'{word}'로 재치 있고 재미있는 삼행시를 지어줘. [글자]: [내용] 형식으로 답변해."
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "system_instruction": "재치 있는 삼행시 생성기입니다.",
                "temperature": 0.8,
            }
        )
        return {"word": word, "poem": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("llm_emotion:app", host="0.0.0.0", port=8002, reload=True)
