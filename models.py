import typing, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import sentence_transformers
import json, os, collections

class SpiderData:
    '''
    Observations:
        - each annotator labeled all 200 rows assigned to them
        - number of sentences in each dataset: 200
        - unique labels: ['About', 'Bio', 'Neither', 'Product/Feature', 'Title/role']
        - label span counts from aggregated datasets: {'About': 64, 'Bio': 75, 'Neither': 355, 'Product/Feature': 223, 'Title/role': 83}
        - Fleiss Kappa: 0.52
    '''
    @classmethod
    def iaa_fleiss(cls, classes:dict, data:typing.List[list]) -> float:
        #https://datatab.net/tutorial/fleiss-kappa
        tbl = [collections.Counter([j[-1] for j in i]) for i in zip(*data)]
        tbl = [{cl:j.get(cl, 0) for cl in classes} for j in tbl]
        ct = collections.defaultdict(int)
        njk = 0
        for row in tbl:
            for cl in classes:
                ct[cl] += row[cl]
                njk += row[cl]**2

        s_ct = sum(ct.values())
        P_e = sum(pow(i/s_ct, 2) for i in ct.values())
        P_o = (1/(len(tbl)*4*3)) * (njk - len(tbl) * 4)
        return (P_o - P_e)/(1 - P_e)

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
        _classes = {b:a for a, b in classes.items()}
        return cls.iaa_fleiss(_classes, data:=[b for _, b in transformed]), _classes, transformed


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
    s = SpiderModels()
    s.load_embedding_model()

    fleiss, classes, data = SpiderData.load_data()
    print('Classes', classes)
    *_, text, labels = zip(*data[0][1])
    embedded_text = s.get_embeddings(text)
    print(embedded_text)
    print('length of embedding array:', len(embedded_text[0]))
