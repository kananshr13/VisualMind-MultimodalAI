# VisualMind — Multimodal Chatbot
### Mini Project | Kanan Sharma 

A multimodal chatbot that takes an image + text question and answers using **BLIP-2 (Salesforce)** via HuggingFace Transformers. FastAPI backend + plain HTML/CSS/JS frontend.

---

## Project Structure

```
multimodal-chatbot/
├── app.py            # FastAPI backend (BLIP-2 inference)
├── index.html        # Frontend UI (open directly in browser)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First install may take a few minutes (PyTorch + Transformers).

### 3. Start the backend
```bash
uvicorn app:app --reload --port 8000
```

On first run, BLIP-2 weights (~5 GB) will auto-download from HuggingFace.
This takes a few minutes — subsequent starts are fast.

### 4. Open the frontend
Just open `index.html` in your browser:
```bash
# macOS
open index.html
# Linux
xdg-open index.html
# Windows
start index.html
```

### 5. Use it!
1. Drag-and-drop or click to upload an image
2. Type any question about it (e.g., "What is in this image?")
3. Hit Enter or click Send

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Check if backend is running        |
| POST   | /describe  | Auto-generate image caption        |
| POST   | /ask       | VQA — answer a question about image|
| POST   | /chat      | Smart unified endpoint (used by UI)|

### Example (curl)
```bash
# Auto-describe
curl -X POST http://localhost:8000/describe \
  -F "file=@your_image.jpg"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -F "file=@your_image.jpg" \
  -F "question=What is the color of the car?"
```

---

## Google Colab (No GPU locally?)

If your machine lacks a GPU, run the backend on Colab:

```python
# In a Colab cell:
!pip install fastapi uvicorn[standard] python-multipart transformers accelerate Pillow -q
!ngrok authtoken YOUR_NGROK_TOKEN

# Copy app.py content into a cell and run:
import nest_asyncio, uvicorn
from pyngrok import ngrok
nest_asyncio.apply()
public_url = ngrok.connect(8000)
print("Backend URL:", public_url)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then replace `http://localhost:8000` in `index.html` with the ngrok URL.

---

## Models / Architecture

```
Image ──► ViT Encoder (BLIP-2) ──┐
                                  ├──► Q-Former ──► OPT-2.7B LLM ──► Answer
Text  ──► Tokenizer ─────────────┘
```

BLIP-2 uses a lightweight **Q-Former** bridge to align visual features with a frozen large language model (OPT-2.7B), enabling both captioning and VQA without full retraining.

---

## References (from Phase-I Report)
- Li et al. — BLIP: Bootstrapping Language-Image Pre-training. ICML 2022.
- Radford et al. — CLIP. ICML 2021.
- Vaswani et al. — Attention Is All You Need. NeurIPS 2017.