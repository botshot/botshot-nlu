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
        tokens = self.tokenizer.tokenize(text)
        text_vectors = []
        output = {}
        for token in tokens:
            v = self.embedding.get_vector(token)
            text_vectors.append(v)

        for i, vector in enumerate(text_vectors):
            most_similar_label, most_similar_entity, max_similarity = None, None, -1.0
            for expression, label, entity in self.keywords:
                similarity = 1 - cosine(expression, vector)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_label = label
                    most_similar_entity = entity
            if max_similarity >= self.threshold:
                output.setdefault(most_similar_entity, []).append({"value": tokens[i], "similar_to": most_similar_label, "confidence": max_similarity})
        return output
