import math
from typing import Optional

import numpy as np

from botshot_nlu.pipeline import PipelineComponent, InputType
from .embedding import Embedding


class BagOfWords(PipelineComponent):

    source = InputType.TOKENS
    target = InputType.VECTOR

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

    def fit(self, sentences, labels, entities):
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

    def transform(self, sentences, labels, entities):
        features = np.zeros([len(sentences), len(self.vocab)])
        for i, sentence in enumerate(sentences):
            features[i] = self.get(sentence)
        return features, labels, entities


class TfIdf(PipelineComponent):

    source = InputType.TOKENS
    target = InputType.VECTOR

    def __init__(self, config):
        super().__init__(config)
        self.vocab = []
        self.word2idx = {}
        self.occurences = {}
        self.document_cnt = 0
        # self.limit = -1 # TODO implement self.counts then drop OR add stopwords to pipeline

    def add(self, sentence):

        past_words = set()
        self.document_cnt += 1

        for token in sentence:
            if token not in self.word2idx:
                i = len(self.vocab)
                self.word2idx[token] = i
                self.vocab.append(token)
                self.occurences[token] = 1
            elif token not in past_words:
                self.occurences[token] += 1
            past_words.add(token)

    def fit(self, sentences, y, z):
        for sentence in sentences:
            self.add(sentence)

    def get(self, sentence):
        features = [0.] * len(self.vocab)

        token_counts = {}

        for token in sentence:
            token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            if token not in self.word2idx:
                continue

            i = self.word2idx[token]
            tf = (count + 1) / (len(sentence) + 1)
            idf = (self.document_cnt + 1) / (self.occurences[token] + 1)
            features[i] = tf / math.log(idf)

        return features

    def transform(self, sentences, y, z):
        sentences = [sent for sent in sentences]  # generator
        features = np.zeros([len(sentences), len(self.vocab)])
        for i, sentence in enumerate(sentences):
            features[i] = self.get(sentence)
        return features, y, z

    def save(self):
        return {
            "vocab": self.vocab, "word2idx": self.word2idx,
            "occurrences": self.occurences, "document_cnt": self.document_cnt}

    def load(self, d: dict):
        self.vocab = d['vocab']
        self.word2idx = d['word2idx']
        self.occurences = d['occurrences']
        self.document_cnt = d['document_cnt']

    def feature_dim(self):
        return len(self.vocab)


class EmbeddingCentroidFeaturizer(PipelineComponent):

    source = InputType.TOKENS
    target = InputType.VECTOR

    def __init__(self, config):
        super().__init__(config)
        self.unk = self.config.get("unk_token", "<UNK>")  # TODO use it
        emb_file = self.config.get("embedding")
        self.embed = Embedding(emb_file)

    def load(self, model_path):
        pass

    def save(self):
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

    def transform(self, sentences, labels, entities):
        sentences = [sent for sent in sentences]
        features = np.zeros([len(sentences), self.embed.dimension])
        for i, sent in enumerate(sentences):
            features[i] = self.get(sent)
        return features, labels, entities

    def fit(self, data, labels, entities):
        pass


class SequentialOneHotBagOfWords(BagOfWords):

    MAX_TOKENS = 20  # TODO: from config

    source = InputType.TOKENS
    target = InputType.VECTOR_SEQUENCE

    def transform(self, sentences, labels, entities):
        features = np.zeros([len(sentences), self.MAX_TOKENS, len(self.vocab)])

        for i, sent in enumerate(sentences):
            sent = sent[:self.MAX_TOKENS]
            for j, token in enumerate(sent):
                token = token.lower() if self.lowercase else token
                if token in self.word2idx:
                    k = self.word2idx[token]
                elif self.unk:
                    k = self.word2idx[self.unk]
                else: continue
                features[i, j, k] = 1.0

        return features, labels, entities

    def get(self, sentence):
        return self.transform([sentence], None, None)[0][0]



class SequentialBagOfWords(BagOfWords):

    MAX_TOKENS = 20  # TODO: from config

    source = InputType.TOKENS
    target = InputType.VECTOR_SEQUENCE

    def transform(self, sentences, labels, entities):
        features = np.zeros([len(sentences), self.MAX_TOKENS])

        for i, sent in enumerate(sentences):
            sent = sent[:self.MAX_TOKENS]
            for j, token in enumerate(sent):
                token = token.lower() if self.lowercase else token
                if token in self.word2idx:
                    k = self.word2idx[token]
                elif self.unk:
                    k = self.word2idx[self.unk]
                else: continue
                features[i, j] = k

        return features, labels, entities

    def get(self, sentence):
        return self.transform([sentence], None, None)[0][0]
