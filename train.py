import os
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from dataset import load_data, build_vocab, get_dataloaders
from models import EmbeddingBagModel, AverageEmbeddingModel, BiLSTMModel
from utils import get_default_device, load_glove_vectors, build_weights_matrix, save_checkpoint

BASE_DIR = Path(__file__).resolve().parent

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_acc, total_count = 0, 0
    total_loss = 0.0

    for idx, (label, text) in enumerate(dataloader):
        label, text = label.to(device), text.to(device)
        optimizer.zero_grad()
        predicted = model(text)
        loss = criterion(predicted, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Sign-based accuracy for labels in {-1, 1}
        acc = (predicted * label > 0).sum().item()
        total_acc += acc
        total_count += label.size(0)
        total_loss += loss.item() * label.size(0)

    return total_loss / total_count, total_acc / total_count

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for label, text in dataloader:
            label, text = label.to(device), text.to(device)
            predicted = model(text)
            loss = criterion(predicted, label)
            acc = (predicted * label > 0).sum().item()
            total_acc += acc
            total_count += label.size(0)
            total_loss += loss.item() * label.size(0)

    return total_loss / total_count, total_acc / total_count

def run_training(args):
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"Using device: {device}")

    # Load dataset
    print("Loading data...")
    train_examples = load_data(args.train_path)
    val_examples = load_data(args.val_path)
    test_examples = load_data(args.test_path)

    print(f"Train samples: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    train_iter = [(float(x["sentiment"]), f"{x.get('summary', '')} {x.get('text', '')}") for x in train_examples]
    val_iter = [(float(x["sentiment"]), f"{x.get('summary', '')} {x.get('text', '')}") for x in val_examples]
    test_iter = [(float(x["sentiment"]), f"{x.get('summary', '')} {x.get('text', '')}") for x in test_examples]

    # Build vocab
    print("Building vocabulary...")
    vocab = build_vocab(train_iter, min_freq=args.min_freq)
    print(f"Vocabulary size: {len(vocab)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_iter, val_iter, test_iter, vocab,
        batch_size=args.batch_size,
        pad_len=args.pad_len,
        device=device
    )

    # Initialize model
    glove = None
    weights_matrix = None
    embed_dim = args.embed_dim

    if args.glove_path and os.path.exists(args.glove_path):
        glove = load_glove_vectors(args.glove_path)
        first_vec = next(iter(glove.values()))
        embed_dim = len(first_vec)
        weights_matrix = build_weights_matrix(glove, vocab, embed_dim=embed_dim)

    if args.model == "embedding_bag":
        model = EmbeddingBagModel(vocab_size=len(vocab), embed_dim=embed_dim)
    elif args.model == "avg_embedding":
        model = AverageEmbeddingModel(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            weights_matrix=weights_matrix,
            trainable=True
        )
    elif args.model == "bilstm":
        model = BiLSTMModel(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=args.hidden_dim,
            weights_matrix=weights_matrix,
            trainable=True
        )
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    model.to(device)
    criterion = nn.MSELoss()

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs ({args.model}, optimizer: {args.optimizer})...")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif val_acc <= best_val_acc:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                args.output_model,
                model,
                vocab,
                args.model,
                config={
                    "embed_dim": embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "pad_len": args.pad_len,
                    "model_type": args.model
                }
            )
            saved_indicator = " [SAVED BEST]"
        else:
            saved_indicator = ""

        print(f"Epoch {epoch:2d}/{args.epochs:2d} ({elapsed:4.1f}s) | "
              f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}%{saved_indicator}")

    print("=" * 65)
    print(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")

    # Evaluate on test set with best model
    from utils import load_checkpoint
    best_model, _, _ = load_checkpoint(args.output_model, device=device)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    print(f"Final Test Accuracy with best model: {test_acc*100:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--train_path", type=str, default=str(BASE_DIR / "data/reviews_train.tsv"))
    parser.add_argument("--val_path", type=str, default=str(BASE_DIR / "data/reviews_val.tsv"))
    parser.add_argument("--test_path", type=str, default=str(BASE_DIR / "data/reviews_test.tsv"))
    parser.add_argument("--model", type=str, default="bilstm", choices=["embedding_bag", "avg_embedding", "bilstm"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--pad_len", type=int, default=100)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--glove_path", type=str, default=str(BASE_DIR / "data/glove.6B.300d.txt"))
    parser.add_argument("--output_model", type=str, default=str(BASE_DIR / "checkpoints/best_model.pt"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_training(args)
