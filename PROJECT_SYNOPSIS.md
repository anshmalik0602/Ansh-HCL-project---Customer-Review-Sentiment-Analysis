# HCLTech Career Shaper Programme | Academic Year 2026-27
## PROJECT SYNOPSIS

# Customer Review Sentiment Analysis
### Deep Learning Sentiment Classification, GloVe Word Embeddings, Real-Time FastAPI Inference Engine, and Claude LLM Comparative Evaluation over Multi-Scale Product Reviews

---

### Project Metadata

| Field | Details |
| :--- | :--- |
| **Project Title** | Customer Review Sentiment Analysis (Deep Learning Classification & LLM Benchmarking) |
| **Project ID** | P_124 |
| **Domain** | Natural Language Processing / Deep Learning / Full-Stack AI |
| **Difficulty** | Advanced |
| **Format** | Individual Project |
| **Learner Name** | **Ansh Malik** |
| **Roll Number** | **2300320220052** |
| **College / Campus** | **ABES Engineering College, Ghaziabad** |
| **Mentor / Guide Name** | **Abhilasha Varshney** |
| **Academic Year** | 2026-27 |
| **Submission Date** | 3 November 2026 |
| **GitHub Repository** | [https://github.com/anshmalik0602/Ansh-HCL-project---Customer-Review-Sentiment-Analysis](https://github.com/anshmalik0602/Ansh-HCL-project---Customer-Review-Sentiment-Analysis) |

---

### Contents
1. Problem Statement | 2. Proposed Solution and Approach | 3. System Architecture | 4. Technology Stack | 5. Dataset Used | 6. Key Results and Metrics | Application Output Screenshots | 7. Challenges and Resolutions | 8. References | 9. Ethical Compliance Declaration | Appendix

---

## 1. Problem Statement

| 5W1H | Project-specific response |
| :--- | :--- |
| **Who?** | E-commerce operators, customer experience leaders, product quality managers, and consumer retail brands monitoring high-volume consumer feedback. |
| **What?** | Manual auditing of thousands of daily customer food reviews is prohibitively slow, subjective, and labor-intensive. Traditional keyword filters fail to comprehend context, negations, and complex review phrasing. |
| **When / where?** | During post-purchase customer feedback cycles, product quality escalations, marketing campaign sentiment audits, and supplier benchmarking. |
| **Why?** | Automated deep learning natural language processing transforms uncurated user feedback into actionable binary sentiment (+1 / -1) with calibrated confidence, cutting manual review triage time by over 90%. |
| **How?** | Tokenize review text sequences, align with GloVe pre-trained vectors, train Bidirectional LSTMs and Embedding Bag classifiers on PyTorch 2.x, deploy through a sub-10ms FastAPI inference engine on Apple Silicon GPU (MPS), and cross-evaluate with Anthropic Claude LLM. |

---

## 2. Proposed Solution and Approach

The project delivers an end-to-end, production-grade Sentiment Analysis application combining PyTorch deep neural networks, a high-throughput FastAPI REST backend, and an interactive glassmorphic web dashboard. The platform trains and evaluates three distinct neural architectures: an **Embedding Bag Model** (fast continuous bag-of-words lookup), an **Average Embedding Model** (global sequence mean pooling), and a **Bidirectional Long Short-Term Memory (BiLSTM) Network** capable of capturing deep forward and backward linguistic context. To resolve the limitations of traditional models, the system incorporates an integrated **Claude LLM benchmark**, enabling real-time zero-shot validation and comparative reasoning across ambiguous and subtle consumer reviews.

### 2.1 Main Features
- **Multi-Architecture Deep Learning:** Implements and compares Embedding Bag, Average Embedding, and BiLSTM architectures optimized with PyTorch 2.x and Adam optimization.
- **Apple Silicon MPS Hardware Acceleration:** Native Metal Performance Shaders (`mps`) support delivering lightning-fast inference (< 10 ms per review) and ~1.6s per training epoch.
- **Self-Contained NLP Preprocessing Pipeline:** Custom regex tokenizer, frequency-ranked vocabulary generator with special tokens (`<unk>`, `<pad>`), and sequence padding (100 tokens), eliminating legacy fragile dependencies.
- **FastAPI High-Performance REST Engine:** Production server exposing endpoints (`/api/predict`, `/api/batch`, `/api/stats`, `/api/health`) with automated OpenAPI/Swagger documentation.
- **Interactive Web Dashboard:** Single-Page Application (Tailwind CSS) featuring single review predictors, real-time confidence meters, continuous raw score readouts, and recognized token chip visualizations.
- **Batch Review Processing:** Multi-line batch evaluation computing aggregate positive/negative ratios, throughput metrics, and dynamic tabular sentiment summaries.
- **Hybrid LLM Comparative Benchmark:** Side-by-side inference engine pitting local neural BiLSTM models against Anthropic Claude LLM with reasoning explanations.

---

## 3. System Architecture

### 3.1 Data Flow Diagram - Level 0
```
[ Customer Review ] ──> [ FastAPI Server ] ──> [ Tokenize & Pad (100) ]
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
     [ PyTorch BiLSTM (MPS GPU) ]                                          [ Claude LLM Engine ]
     • Forward & Backward Recurrent                                        • Zero-Shot Reasoning
     • Sub-10ms Latency                                                    • Contextual Fallback
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    ▼
                                  [ Web Dashboard / REST API Response ]
                                  • Sentiment Badge (Positive / Negative)
                                  • Confidence Meter & Continuous Score
                                  • Latency & Token Visualization
```

### 3.2 Ingestion & Processing Pipeline
1. **TSV Ingestion:** Parses 4,000 train, 500 validation, and 500 test food reviews.
2. **Regex Tokenizer:** In-house word splitting extracting 13,676 unique tokens.
3. **Sequence Padder:** Normalizes review representations to fixed 100-token vectors.
4. **GloVe / Learned Embeddings:** 128 / 300-dimensional semantic vectors.
5. **BiLSTM Recurrent Processing:** Bidirectional temporal feature extraction.
6. **Output Layer:** Linear projection producing signed continuous sentiment scores.

---

## 4. Technology Stack

| Layer | Technology | Role in this project |
| :--- | :--- | :--- |
| **Frontend / Web UI** | HTML5, Tailwind CSS, Font Awesome | Responsive dark-mode dashboard, confidence progress gauges, batch analysis tables. |
| **Backend API** | FastAPI, Uvicorn, Pydantic | High-throughput REST API with asynchronous request lifecycle and automated Swagger docs. |
| **Deep Learning Engine** | PyTorch 2.x, Torch NN | Implements BiLSTM, EmbeddingBag, and Average Embedding neural architectures. |
| **Hardware Acceleration** | Apple Silicon MPS (Metal) | Offloads tensor operations and gradient steps directly to Apple Silicon GPU cores. |
| **Feature Representation** | GloVe 300d, Learned Embeddings | Unsupervised word representation mapping semantic similarity across 300 dimensions. |
| **LLM Inference** | Anthropic Claude API | Performs zero-shot reasoning benchmarks to contrast deep learning against modern LLMs. |
| **Evaluation Suite** | Scikit-Learn, NumPy | Calculates precision, recall, F1-scores, and multiclass confusion matrices. |
| **Interactive Notebook** | JupyterLab, IPyKernel | Exploratory data analysis, statistical review length profiling, and step-by-step training. |

### 4.1 Repository Structure
```
Ansh-HCL-project---Customer-Review-Sentiment-Analysis/
├── server.py                     # FastAPI REST API backend & model inference engine
├── static/
│   └── index.html                # Responsive web frontend (Tailwind CSS, dark theme)
├── models.py                     # PyTorch architectures (BiLSTM, AvgEmbedding, EmbeddingBag)
├── dataset.py                    # Preprocessing, regex tokenization, custom vocab, and dataloaders
├── train.py                      # Multi-model training loop with SGD/Adam and checkpoint saving
├── predict.py                    # CLI and interactive sentiment classifier
├── claude_sentiment.py           # Claude LLM integration & comparative benchmarking
├── download_glove.py             # GloVe 300d embeddings downloader & extractor
├── utils.py                      # MPS device detection, GloVe matrix builder, model serialization
├── start_server.sh               # One-click startup script for frontend and backend
├── checkpoints/
│   ├── best_model.pt             # Pre-trained BiLSTM weights (85.8% val accuracy)
│   └── embedding_bag.pt          # Pre-trained Embedding Bag weights
├── data/
│   ├── reviews_train.tsv         # 4,000 training reviews
│   ├── reviews_val.tsv           # 500 validation reviews
│   └── reviews_test.tsv          # 500 test reviews
├── Sentiment_Analysis.ipynb      # Modernized Jupyter notebook (clean MPS execution)
├── PROJECT_SYNOPSIS.html         # Printable A4 academic synopsis (exact layout match)
└── requirements_modern.txt       # Modern dependencies list
```

---

## 5. Dataset Used

| Field | Value |
| :--- | :--- |
| **Dataset file name** | `reviews_train.tsv`, `reviews_val.tsv`, `reviews_test.tsv` |
| **Total sample count** | 5,000 records (4,000 Train, 500 Validation, 500 Test) |
| **Number of attributes** | 7 columns (`sentiment`, `productId`, `userId`, `summary`, `text`, `helpfulY`, `helpfulN`) |
| **Target classes** | Binary sentiment: +1 (Positive, 2,477 samples) and -1 (Negative, 2,523 samples) |
| **Data formats accepted** | Tab-Separated Values (.tsv), plain text inputs, multi-line batch strings |
| **Vocabulary size** | 13,676 unique tokens inferred from training reviews |

---

## 6. Key Results and Metrics

| Metric | Observed value (from application execution) |
| :--- | :--- |
| **BiLSTM Validation Accuracy** | **85.80%** (Best model checkpoint at Epoch 9) |
| **BiLSTM Test Accuracy** | **83.00%** evaluated over 500 held-out test reviews |
| **Precision / Recall / F1** | Precision: 83.2% macro \| Recall: 83.2% macro \| F1-Score: **0.830** |
| **Per-Review Inference Latency** | **7.84 ms** average latency running on Apple Silicon MPS GPU |
| **Training Speed (MPS)** | **1.5 to 1.8 seconds per epoch** (10 epochs completed in ~16s) |
| **Test Confusion Matrix** | 207 True Positives, 208 True Negatives (415 / 500 correct classifications) |

### Test Confusion Matrix (500 Samples)
- **True Positives (Actual +1, Predicted +1):** 207
- **False Negatives (Actual +1, Predicted -1):** 54
- **False Positives (Actual -1, Predicted +1):** 31
- **True Negatives (Actual -1, Predicted -1):** 208

---

## 7. Challenges Faced and How Resolved

| Challenge | Resolution |
| :--- | :--- |
| **Legacy Library Incompatibilities** | Replaced deprecated `torchtext 0.9.1` with custom regex tokenization and vocabulary generation matching PyTorch 2.x. |
| **Missing Hardcoded Embeddings** | Handled missing Stanford GloVe paths with an automated downloader (`download_glove.py`) and graceful learned embedding fallbacks. |
| **Inference Latency Bottlenecks** | Enabled native Metal Performance Shaders (`mps`) support in PyTorch, reducing per-review inference latency to under 8ms. |
| **Full-Stack Decoupling & Serving** | Unified FastAPI REST routing with a responsive single-page web dashboard served with zero external build tools. |
| **Ambiguity and Review Sarcasm** | Integrated Claude 3.5 Sonnet zero-shot comparative benchmark to provide natural language reasoning for edge-case validation. |

---

## 8. References
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*, 1532-1543.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*, 32, 8024-8035.
- Ramirez, S. (2018). FastAPI: High performance, easy to learn, fast to code, ready for production. *https://fastapi.tiangolo.com*
- Anthropic. (2024). The Claude 3.5 Sonnet Model Technical Card. *https://docs.anthropic.com*

---

## 9. Ethical Compliance Declaration
I confirm that:
- [x] All dataset usage complies with the source licence and academic guidelines.
- [x] No personally identifiable information (PII) or customer credentials are stored.
- [x] AI tools used for assistance, documentation, and code refactoring are disclosed.
- [x] This project complies with ethical AI principles and data protection frameworks.

**Learner Signature:** _______________________  
**Name:** Ansh Malik  
**Date:** 3 November 2026  
**Location:** Ghaziabad, India  
