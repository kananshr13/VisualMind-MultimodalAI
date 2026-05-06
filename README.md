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
