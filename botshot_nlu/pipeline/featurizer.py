import numpy as np

from botshot_nlu.pipeline import PipelineComponent


class BagOfWordsFeaturizer(PipelineComponent):

    def __init__(self, unk="<UNK>", count=False, limit=None):
        super().__init__()
        self.vocab = []
        self.word2idx = {}
        self.unk = unk
        self.count = count
        self.limit = limit  # TODO implement self.counts then drop OR add stopwords to pipeline
        if self.unk:
            self._add_token(unk)

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
            if word not in self.word2idx:
                self._add_token(word)

    def fit(self, sentences):
        for sentence in sentences:
            self._add(sentence)

    def get(self, sentence):
        bow = [0.] * len(self.vocab)
        for token in sentence:

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
