import numpy as np

from botshot_nlu.pipeline import PipelineComponent
from .embedding import Embedding

class BagOfWordsFeaturizer(PipelineComponent):

    def __init__(self, config):
        super().__init__(config)
        self.vocab = []
        self.word2idx = {}
        self.unk = self.config.get("unk_token", "<UNK>")
        self.count = self.config.get("bag_of_words_count", False)
        self.limit = self.config.get("bag_of_words_limit")  # TODO implement self.counts then drop OR add stopwords to pipeline
        self.lowercase = self.config.get("lowercase", True)
        if self.unk:
            self._add_token(self.unk)

    def save(self):
        return {"vocab": self.vocab, "word2idx": self.word2idx, "unk": self.unk, "count": self.count, "limit": self.limit}

    def load(self, d: dict):
        self.vocab = d['vocab']
        self.word2idx = d['word2idx']
        self.unk = d['unk']
        self.count = d['count']
        self.limit = d['limit']

    def feature_dim(self):
        return len(self.vocab)

    def _add_token(self, token):
        if token in self.word2idx:
            raise KeyError("Token {} was already added".format(token))

        i = len(self.vocab)
        self.word2idx[token] = i
        self.vocab.append(token)
        return i

    def _add(self, sentence):
        for word in sentence:
            if self.lowercase:
                word = word.lower()
            if word not in self.word2idx:
                self._add_token(word)

    def fit(self, sentences):
        for sentence in sentences:
            self._add(sentence)

    def get(self, sentence):
        bow = [0.] * len(self.vocab)
        for token in sentence:
            if self.lowercase:
                token = token.lower()

            if token in self.word2idx:
                i = self.word2idx[token]
            elif self.unk:
                i = self.word2idx[self.unk]
            else:
                continue

            if self.count:
                bow[i] += 1.
            else:
                bow[i] = 1.

        return bow

    def transform(self, sentences):
        features = np.zeros([len(sentences), len(self.vocab)])
        for i, sentence in enumerate(sentences):
            features[i] = self.get(sentence)
        return features


EMB = "~/"  # FIXME


class EmbeddingCentroidFeaturizer(PipelineComponent):

    def __init__(self, config):
        super().__init__(config)
        self.unk = self.config.get("unk_token", "<UNK>")  # TODO use it
        emb_file = self.config.get("embedding")
        self.embed = Embedding(emb_file)

    def load(self, model_path):
        pass

    def save(self, model_path):
        pass

    def feature_dim(self):
        return self.embed.dimension

    def get(self, sentence):
        vecs = []
        for token in sentence:
            vec = self.embed.word2vec(token)
            if vec is not None:
                vecs.append(vec)
        centroid = np.mean(vecs, axis=0) if len(vecs) else np.zeros([self.embed.dimension])
        # TODO check for NaN when there are no words
        return centroid

    def transform(self, sentences):
        sentences = [sent for sent in sentences]
        features = np.zeros([len(sentences), self.embed.dimension])
        for i, sent in enumerate(sentences):
            features[i] = self.get(sent)
        return features

    def fit(self, data):
        pass
