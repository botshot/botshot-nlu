from abc import ABC, abstractmethod
import yaml


class KeywordDataset(ABC):

    @abstractmethod
    def get_entities(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, entities):
        raise NotImplementedError()

    @abstractmethod
    def has_changed(self, entities):
        return False


class StaticKeywordDataset(KeywordDataset):

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def get_entities(self):
        return set(self.data.keys())

    def get_data(self, entities):
        return {k: self.data[k] for k in self.data.keys() & entities}
    
    def has_changed(self, entities):
        return False

    @staticmethod
    def load(*filenames):
        data = {}
        for filename in filenames:
            data.update(StaticKeywordDataset._load_keywords(filename))
        return StaticKeywordDataset(data)

    @staticmethod
    def _load_keywords(filename):
        with open(filename) as fp:
            data = yaml.safe_load(fp)
        if 'entities' not in data:
            raise Exception("The keywords file %s doesn't contain the 'entities' key" % filename)
        # TODO: validate file
        return data['entities']
