import os
import sys
import argparse
from pathlib import Path
from dataset import load_data
from utils import load_checkpoint, get_default_device
from predict import predict_sentiment

BASE_DIR = Path(__file__).resolve().parent

class ClaudeSentimentAnalyzer:
    def __init__(self, api_key=None, model="claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"[Claude] Notice: Could not initialize Anthropic client: {e}")

    def analyze(self, text):
        """
        Analyzes review text and returns sentiment ('Positive' or 'Negative'),
        numeric score (+1 or -1), and explanation.
        """
        if not self.client:
            return self._heuristic_fallback(text)

        prompt = f"""You are a precise sentiment analysis system for customer food product reviews.
Analyze the sentiment of the following customer review and classify it strictly as either POSITIVE or NEGATIVE.

Review:
\"\"\"{text}\"\"\"

Respond in the exact format:
SENTIMENT: [POSITIVE or NEGATIVE]
SCORE: [+1 or -1]
EXPLANATION: [Brief 1-sentence reason]
"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            sentiment = "Positive" if "SENTIMENT: POSITIVE" in content.upper() else "Negative"
            score = 1 if sentiment == "Positive" else -1
            explanation = ""
            for line in content.splitlines():
                if line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()
            return {
                "sentiment": sentiment,
                "score": score,
                "explanation": explanation or content,
                "engine": f"Claude ({self.model})"
            }
        except Exception as e:
            print(f"[Claude API Warning]: {e}. Falling back to rule-based analysis.")
            return self._heuristic_fallback(text)

    def _heuristic_fallback(self, text):
        pos_words = {"great", "good", "love", "delicious", "fresh", "best", "excellent", "yummy", "tasty", "awesome", "amazing"}
        neg_words = {"bad", "awful", "terrible", "stale", "waste", "horrible", "hate", "gross", "headache", "poor", "never"}
        lower = text.lower()
        pos_count = sum(1 for w in pos_words if w in lower)
        neg_count = sum(1 for w in neg_words if w in lower)
        sentiment = "Positive" if pos_count >= neg_count else "Negative"
        return {
            "sentiment": sentiment,
            "score": 1 if sentiment == "Positive" else -1,
            "explanation": "Evaluated using fallback lexical sentiment classifier (set ANTHROPIC_API_KEY to use Claude API).",
            "engine": "Fallback Lexical Analyzer"
        }

def compare_models(test_path, num_samples=5, model_path=None):
    device = get_default_device()
    bilstm_model, vocab, config = None, None, None
    if model_path and os.path.exists(model_path):
        bilstm_model, vocab, config = load_checkpoint(model_path, device=device)

    claude = ClaudeSentimentAnalyzer()
    test_data = load_data(test_path)[:num_samples]

    print("\n" + "=" * 80)
    print("COMPARATIVE SENTIMENT ANALYSIS: PyTorch Neural Model vs. Claude")
    print("=" * 80)

    for i, item in enumerate(test_data, 1):
        summary = item.get("summary", "")
        body = item.get("text", "")
        text = f"{summary} {body}".strip()
        ground_truth = "Positive" if item.get("sentiment") == "1" else "Negative"

        print(f"\n[{i}/{len(test_data)}] Review: \"{text[:120]}...\"")
        print(f"  📌 Ground Truth: {ground_truth}")

        if bilstm_model:
            bilstm_res = predict_sentiment(text, bilstm_model, vocab, config, device=device)
            match = "✓" if bilstm_res["sentiment"] == ground_truth else "✗"
            print(f"  🤖 PyTorch BiLSTM:  {bilstm_res['sentiment']} ({match}) [Confidence: {bilstm_res['confidence']*100:.1f}%, Raw: {bilstm_res['raw_score']:.2f}]")

        claude_res = claude.analyze(text)
        cmatch = "✓" if claude_res["sentiment"] == ground_truth else "✗"
        print(f"  🧠 {claude_res['engine']}: {claude_res['sentiment']} ({cmatch})")
        if claude_res.get("explanation"):
            print(f"     Reason: {claude_res['explanation']}")
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Analyze Review Sentiment using Claude and Compare Models")
    parser.add_argument("--text", type=str, default=None, help="Text of review to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare PyTorch BiLSTM vs. Claude on test dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples to compare")
    parser.add_argument("--model_path", type=str, default=str(BASE_DIR / "checkpoints/best_model.pt"))
    parser.add_argument("--test_path", type=str, default=str(BASE_DIR / "data/reviews_test.tsv"))
    args = parser.parse_args()

    if args.compare:
        compare_models(args.test_path, num_samples=args.samples, model_path=args.model_path)
    elif args.text:
        claude = ClaudeSentimentAnalyzer()
        res = claude.analyze(args.text)
        print(f"\nReview: \"{args.text}\"")
        print(f"Engine: {res['engine']}")
        print(f"Sentiment: {res['sentiment']} (Score: {res['score']})")
        print(f"Explanation: {res['explanation']}\n")
    else:
        print("Please provide --text \"Your review here\" or --compare to compare with test dataset.")
        print("Example: python claude_sentiment.py --text \"These chocolates were sublime!\"")

if __name__ == "__main__":
    main()
