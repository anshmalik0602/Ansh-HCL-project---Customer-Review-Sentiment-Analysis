import os
import torch
import numpy as np

def get_default_device():
    """Detects best available device: Apple Silicon MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_glove_vectors(file_path):
    """Loads GloVe embeddings from a text file into a dictionary."""
    if not os.path.exists(file_path):
        return None
    print(f"Loading GloVe embeddings from {file_path}...")
    glove = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
    print(f"Loaded {len(glove)} GloVe word vectors.")
    return glove

def build_weights_matrix(glove, vocab, embed_dim=300):
    """Builds embedding weights matrix aligned with vocabulary indices."""
    weights_matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    hits = 0
    for idx, token in enumerate(vocab.itos):
        if glove and token in glove:
            weights_matrix[idx] = glove[token]
            hits += 1
        else:
            # Random normal initialization for OOV tokens
            weights_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
    if glove:
        print(f"GloVe embedding coverage: {hits}/{len(vocab)} ({hits/len(vocab)*100:.2f}%)")
    return weights_matrix

def save_checkpoint(filepath, model, vocab, model_type, config=None):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "vocab_stoi": dict(vocab.stoi),
        "model_type": model_type,
        "config": config or {}
    }
    torch.save(payload, filepath)
    print(f"Model saved to {filepath}")

def load_checkpoint(filepath, device=None):
    if device is None:
        device = get_default_device()
    from models import EmbeddingBagModel, AverageEmbeddingModel, BiLSTMModel
    from dataset import Vocabulary

    data = torch.load(filepath, map_location=device)
    vocab = Vocabulary()
    vocab.itos = data["vocab_itos"]
    vocab.stoi = data["vocab_stoi"]
    vocab.unk_idx = vocab.stoi.get("<unk>", 0)
    vocab.pad_idx = vocab.stoi.get("<pad>", 1)

    model_type = data["model_type"]
    config = data.get("config", {})

    if model_type == "embedding_bag":
        model = EmbeddingBagModel(vocab_size=len(vocab), embed_dim=config.get("embed_dim", 128))
    elif model_type == "avg_embedding":
        model = AverageEmbeddingModel(vocab_size=len(vocab), embed_dim=config.get("embed_dim", 128))
    elif model_type == "bilstm":
        model = BiLSTMModel(
            vocab_size=len(vocab),
            embed_dim=config.get("embed_dim", 128),
            hidden_dim=config.get("hidden_dim", 128)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, config
