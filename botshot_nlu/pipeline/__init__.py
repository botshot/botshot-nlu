import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from botshot_nlu.resources import Resources


class InputType(Enum):
    TEXT = 1,
    TOKENS = 2,
    LEMMAS = 3,
    VECTORS = 4


class PipelineComponent(ABC):

    def __init__(self, config: dict):  # TODO: can config be removed (e.g. for keyword models)?
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

    # @abstractmethod
    # def wants(self) -> InputType:
    #     pass

    # @abstractmethod  # TODO
    # def produces(self) -> InputType:
    #     pass


class Pipeline:

    def __init__(self, *components):
        self.components = components
        self.l2i = {}
        self.i2l = []

    def fit(self, x, y):
        for component in self.components:
            component.fit(x)
            x = component.transform(x)

        if y is not None:
            self.i2l = list(set(y))
            self.l2i = {label: index for index, label in enumerate(self.i2l)}

    def transform(self, x, y=None):
        for component in self.components:
            x = component.transform(x)
            # logging.debug(x)
        if y is not None:
            y = self.encode_labels(y)
            return x, y
        return x

    def encode_labels(self, y):
        return [self.l2i[y_] for y_ in y]

    def decode_labels(self, y):
        return [self.i2l[y_] for y_ in y]

    def feature_dim(self):
        return self.components[-1].feature_dim()

    def save(self) -> dict:
        data = {"components": []}
        for component in self.components:
            data['components'].append(component.save() or {})
        data['i2l'] = self.i2l
        data['l2i'] = self.l2i
        return data

    def load(self, data):
        for i, component in enumerate(self.components):
            component.load(data['components'][i])
        self.i2l = data['i2l']
        self.l2i = data['l2i']
