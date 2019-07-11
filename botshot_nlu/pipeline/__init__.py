import random
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from botshot_nlu.resources import Resources


class InputType(Enum):
    TEXT = 1,
    TOKENS = 2,
    LEMMAS = 3,
    VECTOR = 4,
    VECTOR_SEQUENCE = 5


class PipelineComponent(ABC):

    def __init__(self, config: dict):  # TODO: can config be removed (e.g. for keyword models)?
        self.config = config

    @abstractmethod
    def fit(self, data, labels, entities):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data, labels, entities):
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

    def __init__(self, *components, add=[]):
        self.components = [(component, component.source, component.target) for component in components]
        self.additional = {}
        for a in add:
            self.additional.setdefault(a.source, []).append(a)
        self.l2i = {}
        self.i2l = []

    def fit(self, x, y, z):
        for component in self.components:
            component[0].fit(x, y, z)
            x, y, z = component[0].transform(x, y, z)

    def transform(self, x, y=None, z=None, source=None, target=None):
        components = self._get_components(source, target)
        # added = []

        for component in components:

            # if self.additional.get(component[1]):
                # print(component[1], "!!!!!!!!!", x)
                # for a in self.additional[component[1]]:
                #     if a.target == target or not target:
                #         added.append(a.transform(x, y, z)[0])

            x, y, z = component[0].transform(x, y, z)

        # if added:
        #     for a in added:
        #         print("ADDING", a)
        #         x = np.concatenate([x, a], axis=-1)

        return x, y, z

    def _get_components(self, source, target):
        try:
            first = min([i for i, component in enumerate(self.components) if component[1] == source])
        except:
            first = 0  # TODO: throw
        try:
            last = max([i for i, component in enumerate(self.components) if component[2] == target])
        except:
            last = len(self.components) - 1  # TODO: throw
        return self.components[first:last + 1]

    def encode_labels(self, y):
        return [self.l2i[y_] for y_ in y]

    def decode_labels(self, y):
        return [self.i2l[y_] for y_ in y]

    def feature_dim(self):
        return self.components[-1][0].feature_dim()

    def save(self) -> dict:
        data = {"components": []}
        for component in self.components:
            data['components'].append(component[0].save() or {})
        data['i2l'] = self.i2l
        data['l2i'] = self.l2i
        return data

    def load(self, data):
        for i, component in enumerate(self.components):
            component[0].load(data['components'][i])
        self.i2l = data['i2l']
        self.l2i = data['l2i']
