# Loading a Extractive Question-Answering model to be fed question-context pairs

from transformers import pipeline

class Model:
    def __init__(self):
        self.roberta = pipeline(model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

    def ask(self, q: str, c: str):
        return self.roberta(question=q, context=c)
