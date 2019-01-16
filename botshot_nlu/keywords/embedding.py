import numpy as np
from scipy.spatial.distance import cosine

from botshot_nlu import utils
from botshot_nlu.pipeline.embedding import Embedding
from . import KeywordExtractor


class EmbeddingKeywordExtractor(KeywordExtractor):
    # TODO: use e.g. kNN for n-grams

    def __init__(self, config, entities, datasets):
        if 'tokenizer' in config:
            self.tokenizer = utils.create_class_instance(config['tokenizer'], config=None)
        else:
            self.tokenizer = utils.get_default_tokenizer()
        self.embedding = Embedding(config['embedding_file'])
        self.threshold = config.get("threshold", 0.7)
        # TODO: self.cache_embeddings = config.get("cache_embeddings", True)  # disable if you have too many tokens
        super().__init__(config, entities, datasets)
        # TODO: configurable tokenizer + stemmer + other preprocessing
    
    def _reformat(self, data):
        data_tuples = []
        for entity, values in data.items():
            for value in values:
                if isinstance(value, str):
                    data_tuples.append((value, value, entity))
                elif isinstance(value, dict):
                    for label, expressions in value.items():
                        for expression in expressions:
                            data_tuples.append((expression, label, entity))
        return data_tuples

    def update(self, data):  # TODO: support multiple words
        self.keywords = []
        for expression, label, entity in self._reformat(data):
            if ' ' in expression: continue
            vector = self.embedding.get_vector(expression)
            self.keywords.append((vector, label, entity))

    def predict(self, text: str):
        text_vectors = []
        output = {}
        for token in self.tokenizer.tokenize(text):
            v = self.embedding.get_vector(token)
            text_vectors.append(v)

        for expression, label, entity in self.keywords:
            for vector in text_vectors:
                if 1 - cosine(expression, vector) >= self.threshold:
                    output.setdefault(entity, []).append({"value": label})
        return output
