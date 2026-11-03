import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from dataset import generate_sequence
from utils import load_checkpoint, get_default_device

BASE_DIR = Path(__file__).resolve().parent

def predict_sentiment(text, model, vocab, config, device=None):
    if device is None:
        device = get_default_device()

    pad_len = config.get("pad_len", 100)
    seq = generate_sequence(text, vocab, pad_len=pad_len)
    tensor = torch.tensor(seq, dtype=torch.int64).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        raw_score = model(tensor).item()

    # The model outputs a continuous value where > 0 is positive, <= 0 is negative
    sentiment = "Positive" if raw_score > 0 else "Negative"
    # Probability confidence using sigmoid
    confidence = 1 / (1 + torch.exp(-torch.tensor(abs(raw_score)))).item()

    return {
        "text": text,
        "sentiment": sentiment,
        "raw_score": raw_score,
        "confidence": confidence
    }

def main():
    parser = argparse.ArgumentParser(description="Predict Sentiment for a Customer Review")
    parser.add_argument("--model_path", type=str, default=str(BASE_DIR / "checkpoints/best_model.pt"), help="Path to trained model")
    parser.add_argument("--text", type=str, default=None, help="Review text to analyze")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt loop")
    args = parser.parse_args()

    device = get_default_device()
    print(f"Loading model from {args.model_path} onto {device}...")
    model, vocab, config = load_checkpoint(args.model_path, device=device)
    print(f"Model ({config.get('model_type', 'unknown')}) loaded successfully!\n")

    if args.text:
        res = predict_sentiment(args.text, model, vocab, config, device=device)
        print(f"Review: \"{res['text']}\"")
        print(f"Sentiment: {res['sentiment']} (Raw score: {res['raw_score']:.3f}, Confidence: {res['confidence']*100:.1f}%)")
        return

    if args.interactive or not args.text:
        print("Interactive Sentiment Analysis Mode")
        print("Type a review and press Enter (or 'quit' to exit):\n")
        # Run a few built-in examples first
        test_examples = [
            "These candies are delicious! Best taste ever, loved the texture.",
            "Horrible taste, stale and made me feel sick. Never buying again.",
            "Average quality, packaging was good but flavor was bland."
        ]
        print("Sample predictions on test reviews:")
        for sample in test_examples:
            res = predict_sentiment(sample, model, vocab, config, device=device)
            emoji = "👍" if res["sentiment"] == "Positive" else "👎"
            print(f" {emoji} [{res['sentiment']}] \"{sample}\" (score: {res['raw_score']:.2f})")
        print("\n" + "-" * 50)

        while True:
            try:
                user_text = input("Enter review: ").strip()
                if not user_text:
                    continue
                if user_text.lower() in ("quit", "exit", "q"):
                    break
                res = predict_sentiment(user_text, model, vocab, config, device=device)
                emoji = "👍" if res["sentiment"] == "Positive" else "👎"
                print(f"Result: {emoji} {res['sentiment']} (confidence: {res['confidence']*100:.1f}%, score: {res['raw_score']:.2f})\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
