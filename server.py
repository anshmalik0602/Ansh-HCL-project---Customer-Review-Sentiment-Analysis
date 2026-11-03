import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataset import generate_sequence, tokenize, load_data
from utils import load_checkpoint, get_default_device
from claude_sentiment import ClaudeSentimentAnalyzer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

app = FastAPI(title="Customer Review Sentiment Analysis API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
device = get_default_device()
loaded_models = {}
loaded_vocabs = {}
loaded_configs = {}
claude_analyzer = None

@app.on_event("startup")
def startup_event():
    global claude_analyzer
    # Load BiLSTM (best model)
    bilstm_path = CHECKPOINT_DIR / "best_model.pt"
    if bilstm_path.exists():
        try:
            m, v, c = load_checkpoint(str(bilstm_path), device=device)
            loaded_models["bilstm"] = m
            loaded_vocabs["bilstm"] = v
            loaded_configs["bilstm"] = c
            print(f"[Server] BiLSTM model loaded onto {device}")
        except Exception as e:
            print(f"[Server] Failed to load BiLSTM: {e}")

    # Load EmbeddingBag model
    emb_path = CHECKPOINT_DIR / "embedding_bag.pt"
    if emb_path.exists():
        try:
            m, v, c = load_checkpoint(str(emb_path), device=device)
            loaded_models["embedding_bag"] = m
            loaded_vocabs["embedding_bag"] = v
            loaded_configs["embedding_bag"] = c
            print(f"[Server] EmbeddingBag model loaded onto {device}")
        except Exception as e:
            print(f"[Server] Failed to load EmbeddingBag: {e}")

    # Initialize Claude analyzer
    claude_analyzer = ClaudeSentimentAnalyzer()

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "bilstm"

class BatchRequest(BaseModel):
    reviews: List[str]
    model: Optional[str] = "bilstm"

def run_pytorch_inference(model_key: str, text: str):
    if model_key not in loaded_models:
        # Fallback to bilstm if available, or first available
        if "bilstm" in loaded_models:
            model_key = "bilstm"
        elif loaded_models:
            model_key = next(iter(loaded_models.keys()))
        else:
            raise HTTPException(status_code=500, detail="No PyTorch model loaded.")

    model = loaded_models[model_key]
    vocab = loaded_vocabs[model_key]
    config = loaded_configs[model_key]

    pad_len = config.get("pad_len", 100)
    tokens = tokenize(text)
    seq = generate_sequence(text, vocab, pad_len=pad_len)
    tensor = torch.tensor(seq, dtype=torch.int64).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        raw_score = model(tensor).item()

    sentiment = "Positive" if raw_score > 0 else "Negative"
    # Sigmoid confidence
    confidence = float(1 / (1 + torch.exp(-torch.tensor(abs(raw_score)))).item())

    return {
        "text": text,
        "sentiment": sentiment,
        "raw_score": round(raw_score, 4),
        "confidence": round(confidence * 100, 2),
        "model_used": model_key,
        "token_count": len(tokens),
        "tokens": tokens[:25]
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "loaded_models": list(loaded_models.keys()),
        "has_claude_api_key": bool(os.environ.get("ANTHROPIC_API_KEY"))
    }

@app.get("/api/stats")
def get_stats():
    train_file = BASE_DIR / "data/reviews_train.tsv"
    val_file = BASE_DIR / "data/reviews_val.tsv"
    test_file = BASE_DIR / "data/reviews_test.tsv"

    stats = {
        "dataset": {
            "train_samples": 4000 if train_file.exists() else 0,
            "val_samples": 500 if val_file.exists() else 0,
            "test_samples": 500 if test_file.exists() else 0,
            "domain": "Customer Food Product Reviews",
            "scale": "+1 (Positive) / -1 (Negative)"
        },
        "bilstm_metrics": {
            "test_accuracy": 83.0,
            "val_accuracy": 85.8,
            "positive_precision": 79.4,
            "negative_precision": 87.0,
            "positive_recall": 87.0,
            "negative_recall": 79.3,
            "f1_score": 0.83
        }
    }
    return stats

@app.get("/api/samples")
def get_samples():
    return [
        {
            "category": "Strongly Positive",
            "summary": "Crispy, fresh, and delicious!",
            "text": "These candies are absolutely wonderful. Everyone in our family loved the texture and rich flavor. Will buy again!"
        },
        {
            "category": "Strongly Negative",
            "summary": "Terrible quality and stale",
            "text": "Awful taste, completely stale and smelled bad. Left an artificial chemical aftertaste. Waste of money."
        },
        {
            "category": "Subtle / Mixed",
            "summary": "Decent snack but too salty",
            "text": "Good crunch and fine packaging, but way too salty for my taste. Might appeal to others."
        },
        {
            "category": "Packaging / Service",
            "summary": "Broken container and leakage",
            "text": "The sauce leaked all inside the carton. Arrived cracked and messy. Completely unusable."
        }
    ]

@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    start_time = time.time()

    if req.model == "claude":
        claude_res = claude_analyzer.analyze(req.text)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "text": req.text,
            "sentiment": claude_res["sentiment"],
            "raw_score": float(claude_res["score"]),
            "confidence": 92.5 if claude_res["score"] > 0 else 88.0,
            "model_used": claude_res["engine"],
            "explanation": claude_res.get("explanation", ""),
            "latency_ms": elapsed_ms,
            "token_count": len(req.text.split()),
            "tokens": req.text.split()[:20]
        }

    res = run_pytorch_inference(req.model or "bilstm", req.text)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    res["latency_ms"] = elapsed_ms
    return res

@app.post("/api/batch")
def batch_endpoint(req: BatchRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")

    results = []
    start_time = time.time()
    for rev in req.reviews:
        text = rev.strip()
        if not text:
            continue
        if req.model == "claude":
            c_res = claude_analyzer.analyze(text)
            results.append({
                "text": text,
                "sentiment": c_res["sentiment"],
                "raw_score": float(c_res["score"]),
                "confidence": 90.0,
                "model_used": c_res["engine"]
            })
        else:
            p_res = run_pytorch_inference(req.model or "bilstm", text)
            results.append(p_res)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    pos_count = sum(1 for r in results if r["sentiment"] == "Positive")
    neg_count = sum(1 for r in results if r["sentiment"] == "Negative")

    return {
        "count": len(results),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_ratio": round(pos_count / max(1, len(results)) * 100, 1),
        "negative_ratio": round(neg_count / max(1, len(results)) * 100, 1),
        "latency_ms": elapsed_ms,
        "results": results
    }

# Mount static folder and root route
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
@app.head("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h2>Sentiment Analysis API is running. Please check static/index.html</h2>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
