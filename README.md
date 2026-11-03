# Customer-Review-Sentiment-Analysis

A full-stack sentiment analysis application for customer food product reviews using deep learning (PyTorch BiLSTM with Apple Silicon MPS GPU acceleration), FastAPI backend, interactive web frontend, and Claude LLM benchmarking.

---

## 🌟 What's Included & Configured

1. **Interactive Web Application (Frontend + Backend)**:
   - **Backend**: High-performance FastAPI REST API (`server.py`) serving inference in under 10ms.
   - **Frontend**: Modern glassmorphic dashboard (`static/index.html`) with real-time sentiment gauge, confidence meter, model selector, batch processor, and confusion matrix diagnostics.
   - **Swagger / OpenAPI**: Interactive documentation at `http://127.0.0.1:8000/docs`.

2. **Pre-configured Virtual Environment (`.venv`)**:
   - Modern **PyTorch 2.x** with Metal Performance Shaders (`mps`) support for Apple Silicon GPU acceleration.
   - Self-contained tokenizer and vocabulary pipeline (no obsolete `torchtext` dependencies).

3. **Pre-trained Deep Learning Models**:
   - **BiLSTM (Bidirectional LSTM)**: Pre-trained and saved at `checkpoints/best_model.pt` (**85.8% validation accuracy**).
   - **Embedding Bag Model**: Pre-trained and saved at `checkpoints/embedding_bag.pt`.

4. **CLI & Jupyter Notebook Tools**:
   - `start_server.sh`: One-click startup script for frontend and backend.
   - `predict.py`: CLI and interactive terminal sentiment classifier.
   - `train.py`: Model trainer supporting BiLSTM, Average Embedding, and Embedding Bag.
   - `claude_sentiment.py`: Claude LLM comparative evaluator and benchmark.
   - `Sentiment_Analysis.ipynb`: Fully working Jupyter Notebook with registered kernel `Python (Sentiment Analysis)`.

---

## 🚀 Running the Web Application (Frontend + Backend)

### Launch with One Command
```bash
cd /Users/anshmalik/.gemini/antigravity/scratch/Customer-Review-Sentiment-Analysis
./start_server.sh
```

Then open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Web Dashboard Features:
1. **Single Review Predictor**:
   - Type or paste custom product reviews.
   - One-click sample review buttons (Strongly Positive, Strongly Negative, Subtle/Mixed, Packaging issues).
   - Instant confidence progress meter, continuous raw score, token chips, and latency badge.
   - Model switcher: **BiLSTM (PyTorch)**, **Embedding Bag**, or **Claude LLM**.
2. **Batch Analyzer**:
   - Paste multiple reviews (one per line) or load sample test batches.
   - Generates summary statistics (% Positive vs % Negative) and a detailed review classification table.
3. **Model Performance Tab**:
   - Interactive confusion matrix (True Positives, False Positives, True Negatives, False Negatives).
   - Precision, Recall, and F1-score benchmarks.

---

## 💻 Running via Command Line (CLI)

Activate the virtual environment:
```bash
cd /Users/anshmalik/.gemini/antigravity/scratch/Customer-Review-Sentiment-Analysis
source .venv/bin/activate
```

- **Single review prediction:**
  ```bash
  python predict.py --text "These chocolates are silky smooth and delicious!"
  ```

- **Interactive prompt loop:**
  ```bash
  python predict.py --interactive
  ```

- **Compare with Claude LLM on test dataset:**
  ```bash
  python claude_sentiment.py --compare --samples 5
  ```

- **Train models:**
  ```bash
  python train.py --model bilstm --epochs 10 --optimizer adam --lr 0.001
  ```
