import numpy as np
import torch
import torch.nn as nn

class EmbeddingBagModel(nn.Module):
    """
    Embedding Bag Model: Computes sums or averages of 'bags' of embeddings
    without instantiating intermediate embeddings.
    """
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.hidden = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, text):
        # text: (batch_size, seq_len)
        hidden = self.hidden(text)
        out = self.output(hidden)
        return out.squeeze(-1)

class AverageEmbeddingModel(nn.Module):
    """
    Average Embedding Model: Uses word embeddings (e.g. GloVe or learned),
    takes the mean over sequence length, and applies a linear layer.
    """
    def __init__(self, vocab_size=None, embed_dim=300, weights_matrix=None, trainable=True):
        super().__init__()
        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(weights_matrix.astype(np.float32)),
                freeze=not trainable
            )
            embed_dim = weights_matrix.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, text):
        embedded = self.embedding(text)  # (batch_size, seq_len, embed_dim)
        avg_embedding = torch.mean(embedded, dim=1)
        out = self.output(avg_embedding)
        return out.squeeze(-1)

class BiLSTMModel(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) Model:
    Captures temporal word context in both forward and backward directions.
    """
    def __init__(self, vocab_size=None, embed_dim=300, hidden_dim=128, weights_matrix=None, trainable=True):
        super().__init__()
        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(weights_matrix.astype(np.float32)),
                freeze=not trainable
            )
            embed_dim = weights_matrix.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        embedded = self.embedding(text)  # (batch_size, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(embedded)
        avg_vector = torch.mean(output, dim=1)
        out = self.output(avg_vector)
        return out.squeeze(-1)
