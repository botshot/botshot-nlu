from abc import ABC, abstractmethod
from datetime import datetime, timedelta
#import asyncio
from botshot_nlu.pipeline import InputType


class KeywordExtractor(ABC):

    def __init__(self, config, entities, datasets, pipeline, resources):
        self.config = config
        self.entities = entities
        self.datasets = datasets
        self.pipeline = pipeline
        self.resources = resources
        self.update(self._get_data())
        self.next_update = datetime.now() + timedelta(minutes=10)

    def _get_data(self):
        data = {}
        for dataset in self.datasets:
            data.update(dataset.get_data(self.entities))
        return data

    @abstractmethod
    def update(self, data):
        raise NotImplementedError()

    def predict(self, text: str):
        if datetime.now() >= self.next_update:
            self.update(self._get_data())  # TODO: update model asynchronously

    # @classmethod
    # @abstractmethod
    # def wants(cls) -> InputType:  # TODO
    #     pass

# TODO: async def update():
