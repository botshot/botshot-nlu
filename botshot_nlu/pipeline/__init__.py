import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class PipelineComponent(ABC):

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(self, data):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def load(self, saved_data: dict):
        raise NotImplementedError()

    @abstractmethod
    def save(self) -> Optional[dict]:
        raise NotImplementedError()

    @abstractmethod
    def feature_dim(self):
        raise NotImplementedError()


class Pipeline:

    def __init__(self, tokenizer=None, featurizer=None):
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.l2i = {}
        self.i2l = []

    def fit(self, x, y):
        self.tokenizer.fit(x)
        all_tokens = self.tokenizer.transform(x)
        self.featurizer.fit(all_tokens)

        self.i2l = list(set(y))
        self.l2i = {label:index for index, label in enumerate(self.i2l)}

    def transform(self, x, y=None):
        x = self.featurizer.transform(self.tokenizer.transform(x))
        if y is not None:
            y = self.encode_labels(y)
            return x, y
        else:
            return x

    def encode_labels(self, y):
        return [self.l2i[y_] for y_ in y]

    def decode_labels(self, y):
        return [self.i2l[y_] for y_ in y]

    def feature_dim(self):
        return self.featurizer.feature_dim()

    def save(self) -> dict:
        data = {}
        data['tokenizer'] = self.tokenizer.save()
        data['featurizer'] = self.featurizer.save()
        data['i2l'] = self.i2l
        data['l2i'] = self.l2i
        return data

    def load(self, data):
        self.tokenizer.load(data.get("tokenizer"))
        self.featurizer.load(data.get("featurizer"))
        self.i2l = data['i2l']
        self.l2i = data['l2i']
