import typing, torch
import sentence_transformers
import json, os, collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
from transformers import RobertaConfig


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
    def iaa_fleiss(cls, classes: dict, data: typing.List[list]) -> float:
        # https://datatab.net/tutorial/fleiss-kappa
        tbl = [collections.Counter([j[-1] for j in i]) for i in zip(*data)]
        tbl = [{cl: j.get(cl, 0) for cl in classes} for j in tbl]
        ct = collections.defaultdict(int)
        njk = 0
        for row in tbl:
            for cl in classes:
                ct[cl] += row[cl]
                njk += row[cl] ** 2

        s_ct = sum(ct.values())
        P_e = sum(pow(i / s_ct, 2) for i in ct.values())
        P_o = (1 / (len(tbl) * 4 * 3)) * (njk - len(tbl) * 4)
        return (P_o - P_e) / (1 - P_e)

    @classmethod
    def load_data(cls, data_folder: str = 'other_group_annotations') -> typing.Any:
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
        _classes = {b: a for a, b in classes.items()}
        return cls.iaa_fleiss(_classes, data := [b for _, b in transformed]), _classes, transformed


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

    def get_embeddings(self, text: typing.List[str], as_tensor: bool = True) -> typing.List[typing.List[float]]:
        embeddings = self.embedding_model.encode(text, normalize_embeddings=True)
        return torch.tensor(embeddings) if as_tensor else embeddings


def prepare_data(data):
    *_, texts, labels = zip(*data[2][1])
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    embedded_texts = s.get_embeddings(texts)
    return embedded_texts, numeric_labels, label_encoder


def split_data(embedded_texts, numeric_labels):
    return train_test_split(embedded_texts, numeric_labels, test_size=0.2, random_state=42)


def train_model(config, X_train, y_train):
    model_type = config['model_type']
    model = None
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=config['model_parameters']['n_estimators'], random_state=config['model_parameters']['random_state'])
    elif model_type == "GaussianNB":
        model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(config, model, X_test, y_test, label_encoder):
    model_type = config['model_type']
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(model_type + ' Confusion Matrix')
    plt.savefig(model_type + ' confusion_matrix.png')


def roberta(config,data):
    *_, texts, labels = zip(*data[0][1])

    c = RobertaConfig.from_pretrained('roberta-base')
    c.num_labels = 5

    # Load the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=c)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    label_mapping = {"Neither": 0, "Product/Feature": 1, "Title/role": 2, "Bio": 3, "About": 4}
    numeric_labels = [label_mapping[label] for label in labels]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(numeric_labels)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], inputs['labels'],
                                                                          test_size=0.2)

    train_data = torch.utils.data.TensorDataset(train_inputs, train_labels)
    val_data = torch.utils.data.TensorDataset(val_inputs, val_labels)
    train_loader = DataLoader(train_data, batch_size=config['data_parameters']['batch_size'])
    val_loader = DataLoader(val_data, batch_size=config['data_parameters']['batch_size'])

    optimizer = AdamW(model.parameters(), lr=config['model_parameters']['learning_rate'])

    for epoch in range(config['model_parameters']['epochs']):
        print('Epoch {}'.format(epoch + 1,))
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch[0], labels=batch[1])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        for batch in val_loader:
            with torch.no_grad():
                outputs = model(batch[0], labels=batch[1])
                val_loss = outputs.loss
    return model, val_loader


def roberta_evaluate(config, model, dataloader):
    model_type = config['model_type']
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch[0])
            logits = outputs.logits
            predictions.extend(np.argmax(logits.detach().numpy(), axis=1).flatten())
            true_labels.extend(batch[1].numpy().flatten())

    print('Accuracy:', accuracy_score(true_labels, predictions))
    print('\nClassification Report:\n', classification_report(true_labels, predictions))
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(model_type + ' Confusion Matrix')
    plt.savefig(model_type + 'confusion_matrix.png')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    s = SpiderModels()
    s.load_embedding_model()

    config = load_config('config.yaml')

    fleiss, classes, data = SpiderData.load_data()
    print('Classes:', classes)

    embedded_text, numeric_labels, label_encoder = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(embedded_text, numeric_labels)

    # """Naive Bayes"""
    # naive_bayes_config = config['models']['naive_bayes']
    # naive_bayes = train_model(naive_bayes_config, X_train, y_train)
    # evaluate_model(naive_bayes_config, naive_bayes, X_test, y_test, label_encoder)

    # """Random Forest"""
    # random_forest_config = config['models']['random_forest']
    # random_forest = train_model(random_forest_config, X_train, y_train)
    # evaluate_model(random_forest_config, random_forest, X_test, y_test, label_encoder)

    """RoBERTa"""
    roberta_config = config['models']['roberta']
    roberta_model, val_loader = roberta(roberta_config,data)
    roberta_evaluate(roberta_config, roberta_model, val_loader)
