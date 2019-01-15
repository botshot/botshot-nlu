import os
from abc import ABC, abstractmethod
from collections import namedtuple

from botshot_nlu.dataset.intent import IntentDataset


Metrics = namedtuple("Metrics", ("loss", "accuracy", "precision", "recall", "f1"))


class IntentModel(ABC):

    def __init__(self, pipeline, config=None):
        self.pipeline = pipeline
        if config is None:
            config = {}
        self.config = config

    @abstractmethod
    def train(self, dataset: IntentDataset) -> Metrics:
        raise NotImplementedError()

    @abstractmethod
    def test(self, dataset: IntentDataset) -> Metrics:
        raise NotImplementedError()

    @abstractmethod
    def unload(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, from_path: str):
        IntentModel._verify_path(from_path)

    def save(self, to_path: str):
        self._verify_path(to_path)

    @abstractmethod
    def predict(self, input):
        raise NotImplementedError()

    @classmethod
    def _verify_path(self, path: str):
        if path is None:
            raise Exception("Load/save path is None")
        elif not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            raise Exception("Load/save path {} is not a directory".format(path))
