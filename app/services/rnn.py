import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, encoded_tokens, sequence_length):
        self.encoded_tokens = encoded_tokens
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.encoded_tokens) - self.sequence_length)

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.encoded_tokens[index : index + self.sequence_length],
                dtype=torch.long,
            ),
            torch.tensor(
                self.encoded_tokens[index + 1 : index + self.sequence_length + 1],
                dtype=torch.long,
            ),
        )


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class RNNTextGenerator:
    def __init__(
        self,
        corpus,
        sequence_length=5,
        epochs=75,
        learning_rate=0.01,
        device="cpu",
    ):
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.tokens = self.tokenize(" ".join(corpus) if isinstance(corpus, list) else corpus)
        self.vocab, self.inv_vocab = self.build_vocab(self.tokens)
        self.model = LSTMModel(vocab_size=len(self.vocab)).to(self.device)

        encoded_tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in self.tokens]
        self.train(encoded_tokens, epochs=epochs, learning_rate=learning_rate)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, tokens):
        counter = Counter(tokens)
        vocab = {
            word: index + 2
            for index, (word, _) in enumerate(counter.most_common())
        }
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        inv_vocab = {index: word for word, index in vocab.items()}
        return vocab, inv_vocab

    def train(self, encoded_tokens, epochs, learning_rate):
        dataset = TextDataset(encoded_tokens, sequence_length=self.sequence_length)
        if len(dataset) == 0:
            return

        generator = torch.Generator().manual_seed(42)
        data_loader = DataLoader(
            dataset,
            batch_size=min(8, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        torch.manual_seed(42)

        for _ in range(epochs):
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, _ = self.model(inputs)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def generate_text(self, start_word, length=20, temperature=1.0):
        words = self.tokenize(start_word)
        if not words:
            return ""

        self.model.eval()
        input_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        hidden = None

        with torch.no_grad():
            for _ in range(max(0, length - len(words))):
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                output, hidden = self.model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probabilities = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1).item()
                words.append(self.inv_vocab.get(next_id, "<UNK>"))
                input_ids.append(next_id)

        return " ".join(words)
