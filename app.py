from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import requests
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


@app.get("/")
def home():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.post("/chat")
async def chat(file: UploadFile = File(...), message: str = Form(...)):
    try:
        image_bytes = await file.read()

        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes
        )

        result = response.json()

        # 🔥 HANDLE ALL CASES
        if isinstance(result, dict) and "error" in result:
            return {"response": f"HF Error: {result['error']}"}

        if isinstance(result, list):
            return {"response": result[0]["generated_text"]}

        return {"response": "Unexpected response from model"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
