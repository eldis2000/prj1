import os
import json
import uvicorn
import io
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(title="AI 영수증 분석기")

# 템플릿 및 정적 파일 경로 설정
templates = Jinja2Templates(directory="templates")
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Gemini API 설정
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
# 비전 기능이 강력한 gemini-1.5-flash 또는 gemini-2.0-flash 사용
model_name = "gemini-2.5-flash" 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_receipt.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_receipt(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse("index_receipt.html", {
            "request": request, 
            "error": "이미지 파일만 업로드 가능합니다."
        })

    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Gemini 프롬프트 구성
        prompt = """
        이 영수증 이미지를 분석하여 상품명, 수량, 금액 정보를 추출해 주세요.
        또한 영수증의 총 합계 금액(Total)도 찾아주세요.
        
        결과는 반드시 아래의 JSON 형식으로만 답변해 주세요:
        {
          "receipt_items": [
            {"name": "상품명", "quantity": "수량", "amount": "금액"},
            ...
          ],
          "total_amount": "총 합계 금액"
        }
        
        수량이나 금액을 찾을 수 없는 경우 해당 필드는 비워두거나 0으로 표시하세요.
        단위(원, KRW 등)는 제외하고 숫자만 추출해 주세요.
        """

        # Gemini 호출 (이미지 전달)
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=contents, mime_type=file.content_type)
            ],
            config={
                "system_instruction": "당신은 영수증 데이터를 정확하게 추출하는 AI 전문가입니다. JSON 형식으로만 응답하며, 한국어 상품명을 정확히 인식해야 합니다.",
                "response_mime_type": "application/json",
            }
        )
        
        # 결과 파싱
        try:
            result_data = json.loads(response.text)
        except json.JSONDecodeError:
            # JSON 형식이 아닐 경우 텍스트를 정제하여 시도
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            result_data = json.loads(text)

        return templates.TemplateResponse("index_receipt.html", {
            "request": request,
            "filename": file.filename,
            "result": result_data
        })

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "Gemini API 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
        return templates.TemplateResponse("index_receipt.html", {
            "request": request, 
            "error": f"분석 중 오류가 발생했습니다: {error_msg}"
        })

@app.post("/api/analyze")
async def api_analyze_receipt(file: UploadFile = File(...)):
    """API 용 엔드포인트"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        prompt = "영수증 이미지에서 상품명, 수량, 금액, 총합계를 JSON 형식으로 추출해줘."
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
            }
        )
        return JSONResponse(content=json.loads(response.text))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("receipt_analyzer:app", host="0.0.0.0", port=8003, reload=True)
