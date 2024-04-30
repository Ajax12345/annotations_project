import typing, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import sentence_transformers
import json, os

class SpiderData:
    '''
    Observations:
        - each annotator labeled all 200 rows assigned to them

    '''
    @classmethod
    def load_data(cls, data_folder:str = 'other_group_annotations') -> typing.Any:
        d = {}
        for i in os.listdir(data_folder):
            if i.endswith('.json'):
                with open(os.path.join(data_folder, i)) as f:
                    d[i.split('.')[0]] = json.load(f)
        
        transformed = [(a, [[j['data']['location'], j['data']['url'], 
                j['data']['text'], 
                j['annotations'][0]['result'][0]['value']['choices'][0]] 
                for j in b]) 
            for a, b in d.items()]

        assert all(len({tuple(j[:-1]) for j in i}) == 1 for i in zip(*[b for _, b in transformed]))
        classes = dict(enumerate(sorted({c for _, b in transformed for *_, c in b})))
        return classes
        
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
    '''
    s = SpiderModels()
    s.load_embedding_model()
    print(s.get_embeddings(['about', 'product', 'our solutions']))
    '''

    print(SpiderData.load_data())