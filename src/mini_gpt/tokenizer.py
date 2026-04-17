import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(list(chars))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str):
        chars = set(text)
        return cls(chars)
    
    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chars": self.chars
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])