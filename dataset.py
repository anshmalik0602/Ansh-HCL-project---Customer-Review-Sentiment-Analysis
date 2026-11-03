import re
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def load_data(data_path):
    """
    Loads TSV dataset formatted as:
    sentiment \t productId \t userId \t summary \t text \t helpfulY \t helpfulN
    """
    with open(data_path, "rb") as f:
        lines = f.readlines()
    lines = [l.decode("cp1252", errors="replace").strip().split("\t") for l in lines]
    headers = lines[0]
    examples = []
    for line in lines[1:]:
        if len(line) == len(headers):
            examples.append(dict(zip(headers, line)))
    return examples

def tokenize(text):
    """Basic English tokenizer splitting on whitespace and punctuation."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

class Vocabulary:
    def __init__(self, counter=None, min_freq=1, specials=("<unk>", "<pad>")):
        self.specials = list(specials)
        self.stoi = defaultdict(int)
        for idx, tok in enumerate(self.specials):
            self.stoi[tok] = idx
        self.itos = list(self.specials)
        self.unk_idx = self.stoi["<unk>"]
        self.pad_idx = self.stoi["<pad>"]

        if counter:
            sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            for token, freq in sorted_tokens:
                if freq >= min_freq and token not in self.stoi:
                    idx = len(self.itos)
                    self.stoi[token] = idx
                    self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_idx)

def build_vocab(train_iter, min_freq=1):
    counter = Counter()
    for _, text in train_iter:
        counter.update(tokenize(text))
    return Vocabulary(counter, min_freq=min_freq)

def generate_sequence(text, vocab, pad_len=100, pad_token="<pad>"):
    tokens = tokenize(text)
    if len(tokens) <= pad_len:
        remainder = pad_len - len(tokens)
        sequence_words = tokens + [pad_token] * remainder
    else:
        sequence_words = tokens[:pad_len]
    return [vocab[word] for word in sequence_words]

def get_collate_fn(vocab, pad_len=100, device="cpu"):
    def collate_batch(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(label)
            seq = generate_sequence(text, vocab, pad_len=pad_len)
            text_list.append(torch.tensor(seq, dtype=torch.int64).unsqueeze(0))
        label_tensor = torch.tensor(label_list, dtype=torch.float32)
        text_tensor = torch.cat(text_list, dim=0)
        return label_tensor.to(device), text_tensor.to(device)
    return collate_batch

def get_dataloaders(train_iter, val_iter, test_iter, vocab, batch_size=64, pad_len=100, device="cpu"):
    collate_fn = get_collate_fn(vocab, pad_len=pad_len, device=device)
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(val_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
