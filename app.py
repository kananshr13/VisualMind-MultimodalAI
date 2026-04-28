from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import pipeline
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ lightweight model (works on Render free tier)
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(file: UploadFile = File(...), message: str = Form(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = captioner(image)[0]["generated_text"]

        if "describe" in message.lower():
            response = result
        else:
            response = f"I see: {result}"

        return {"response": response}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
