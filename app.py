from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

@app.get("/")
def home():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.post("/chat")
async def chat(file: UploadFile = File(...), message: str = Form(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        result = captioner(image)
        caption = result[0]["generated_text"]

        return {"response": caption}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
