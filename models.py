from sentence_transformers import SentenceTransformer
import typing

class SpiderModels:
    def __init__(self) -> None:
        self.embedding_model = None
    
    def load_embedding_model(self) -> None:
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    def get_embeddings(self, text:typing.List[str]) -> typing.List[typing.List[float]]:
        return self.embedding_model.encode(text, normalize_embeddings=True)


if __name__ == '__main__':
    s = SpiderModels()
    s.load_embedding_model()
    print(len(s.get_embeddings(['about', 'product', 'our solutions'])[0]))