import typing, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import sentence_transformers

class SpiderModels:
    def __init__(self) -> None:
        self.embedding_model = None
    
    def load_embedding_model(self) -> None:
        self.embedding_model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5')

    def get_embeddings(self, text:typing.List[str], as_tensor:bool = True) -> typing.List[typing.List[float]]:
        embeddings = self.embedding_model.encode(text, normalize_embeddings=True)
        return torch.tensor(embeddings) if as_tensor else embeddings

if __name__ == '__main__':
    s = SpiderModels()
    s.load_embedding_model()
    print(s.get_embeddings(['about', 'product', 'our solutions']))