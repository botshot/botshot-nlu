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

    def update(self, data):
        self.keywords = {}  # dict of n-grams: [example tuple]
        self.max_grams = 0
        for expression, label, entity in self._reformat(data):
            tokens = self.tokenizer.tokenize(expression)
            vector = self._get_centroid(tokens)
            n_grams = len(tokens)
            self.max_grams = max(n_grams, self.max_grams)
            self.keywords.setdefault(n_grams, []).append((vector, label, entity))

    def _get_centroid(self, tokens: list):
        # TODO: support other methods than centroid: concat, doc2vec
        v = np.zeros([300])
        for token in tokens:
            v += self.embedding.get_vector(token)
        return v / len(tokens)

    def _predict_ngrams(self, tokens: list, text_vectors: list, keywords, n):
        output = {}
        # generate a centroid for each n-gram in text
        ngram_vectors = [np.mean(text_vectors[i:i+n], axis=0) for i in range(len(text_vectors) - n + 1)]
        # for each n-gram, find the most similar vector in training examples
        for i, vector in enumerate(ngram_vectors):
            most_similar_label, most_similar_entity, max_similarity = None, None, -1.0
            for expression, label, entity in keywords:
                similarity = 1 - cosine(expression, vector)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_label = label
                    most_similar_entity = entity
            if max_similarity >= self.threshold:
                matching_text = ' '.join(tokens[i:i+n])
                output.setdefault(most_similar_entity, []).append({"value": matching_text, "similar_to": most_similar_label, "confidence": max_similarity})
        return output

    def predict(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        text_vectors = []
        output = {}
        # embed all tokens in text
        for token in tokens:
            v = self.embedding.get_vector(token)
            text_vectors.append(v)
        # for each n, compare all n-token substrings with all examples of length n
        for n, keywords in self.keywords.items():
            output.update(self._predict_ngrams(tokens, text_vectors, keywords, n))
        return output
