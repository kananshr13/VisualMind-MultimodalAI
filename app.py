from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from transformers import pipeline

app = FastAPI(title="VisualMind AI")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model (CORRECT for transformers 4.37) ─────────────
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# ── Health ─────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "mode": "real-ai"}

# ── Generate caption ───────────────────
def generate_caption(image: Image.Image):
    result = captioner(image)
    return result[0]["generated_text"]

# ── Chat endpoint ──────────────────────
@app.post("/chat")
async def chat(
    file: UploadFile = File(...),
    message: str = Form(...),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        caption = generate_caption(image)

        if "describe" in message.lower():
            response = caption
        else:
            response = f"I see: {caption}"

        return {"response": response}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})