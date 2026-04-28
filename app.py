from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace API
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
HF_TOKEN = os.getenv("HF_TOKEN")  # from Render env

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def query_image(image_bytes):
    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=image_bytes
    )
    return response.json()


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = query_image(image_bytes)

        # Handle API response safely
        if isinstance(result, list):
            caption = result[0]["generated_text"]
        else:
            caption = "Error: " + str(result)

        return {"response": caption}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
