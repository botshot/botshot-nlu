import numpy as np


class LabelBinarizer():

    def __init__(self):
        """Hey, don't forget to save it."""
        self.l2i = {}
        self.i2l = []

    def fit(self, y):
        """
        y is the 1D set of all possible labels
        y should contain the null element @ index 0
        """
        self.i2l = list(set(y))
        self.l2i = {label: index for index, label in enumerate(self.i2l)}

    def encode_onehot(self, y):
        """
        :param y a d-dimensional array of labels for each position
        :return a one-hot array of d+1 dimensions
        """
        vocab_len = len(self.i2l)
        y = np.array(y)
        shape = list(y.shape) + [vocab_len]
        encoded = np.zeros(shape)

        for idx, l in np.ndenumerate(y):
            i = self.l2i[l]
            encoded[idx][i] = 1.0

        return encoded

    def encode_labels(self, y):
        """
        :param y a d-dimensional array of labels for each position
        :return a d-dimensional array of indices
        """
        y = np.array(y)
        encoded = np.zeros(y.shape)

        for idx, l in np.ndenumerate(y):
            i = self.l2i[l]
            encoded[idx] = i

        return encoded

    def decode_labels(self, y):
        """
        :param y: a 1-D numpy array of indices
        :return: a 1-D list of labels
        """

        decoded = [None] * len(y)

        for idx, i in enumerate(y):
            decoded[idx] = self.i2l[i]

        return decoded

    def save(self) -> dict:
        data = {'i2l': self.i2l, 'l2i': self.l2i}
        return data

    def load(self, data):
        self.i2l = data['i2l']
        self.l2i = data['l2i']


def ___entities_to_array(batch_entities: list, max_len, default='O'):

    entities = np.full(shape=[len(batch_entities), max_len], fill_value=default, dtype=str)

    for i, example in enumerate(batch_entities):
        for entity in example:
            start, end = entity['start'], entity['end']
            for j in range(start, end+1):
                entities[i, j] = entity['entity']
            # TODO: check for overlaps in advance

    return entities


def entities_to_array(batch_entities: list, max_len, default='O'):
    all_entities = []
    for example in batch_entities:
        entities = [default] * max_len
        for entity in example:
            start, end = entity['start'], entity['end']
            for j in range(start, end+1):
                entities[j] = entity['entity']
        all_entities.append(entities)
    return all_entities


def pad_entities(e, max_len=20):
    for i, e_i in enumerate(e):
        e[i] = e_i + ['O'] * (max_len - len(e_i))
    return e
