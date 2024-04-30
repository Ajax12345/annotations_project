import typing, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import sentence_transformers

class SpiderModels:
    '''
    Useful links for the embeddings:
        https://huggingface.co/BAAI/bge-small-en-v1.5
        https://huggingface.co/spaces/mteb/leaderboard
        https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list
    
    Required pip install for embeddings:
        pip install -U sentence-transformers
    '''
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