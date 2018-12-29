import random
from enum import Enum


class BatchMode(Enum):
    SEQUENTIAL = 1,
    RANDOM = 2,
    BALANCED = 3


class IntentDataset:

    def __init__(self, data_pairs, mode=BatchMode.SEQUENTIAL):
        self.data_pairs = data_pairs
        _, self.labels = zip(*data_pairs)
        self.labels = list(set(self.labels))
        self.utterances = {}
        for text, label in data_pairs:
            self.utterances.setdefault(label, []).append(text)
        self.set_mode(mode)

    def set_mode(self, mode: BatchMode):
        self.mode = mode
        if self.mode == BatchMode.SEQUENTIAL:
            self.offset = 0
            random.shuffle(self.data_pairs)

    def _get_balanced(self, batch_size):
        labels = [random.choice(self.labels) for _ in range(batch_size)]
        x, y = [], []
        for i, label in enumerate(labels):
            sentence = random.choice(self.utterances[label])
            x.append(sentence)
            y.append(label)
        return x, y

    def _get_random(self, batch_size):
        x, y = zip(*random.choices(self.data_pairs, k=batch_size))
        return x, y

    def _get_next(self, batch_size, drop_remainder=True):
        if self.offset >= self.count():
            raise IntentDataset.EndOfEpoch()
        elif self.offset + batch_size > self.count() and drop_remainder:
            raise IntentDataset.EndOfEpoch()

        x, y = zip(*self.data_pairs[self.offset:self.offset+batch_size])
        self.offset += batch_size
        return x, y

    def get_batch(self, batch_size, drop_remainder=True):
        if batch_size > self.count():
            raise Exception("Batch size > number of examples")
        if self.mode == BatchMode.SEQUENTIAL:
            return self._get_next(batch_size, drop_remainder)
        elif self.mode == BatchMode.RANDOM:
            return self._get_random(batch_size)
        elif self.mode == BatchMode.BALANCED:
            return self._get_balanced(batch_size)

    def count(self):
        return len(self.data_pairs)

    def get_all(self):
        return zip(*self.data_pairs)

    def label_count(self):
        return len(self.labels)

    def split(self, test_size=0.1):
        train, test = [], []
        for label, utterances in self.utterances.items():
            split_idx = int(len(utterances) * test_size)
            random.shuffle(utterances)
            train_utterances = utterances[split_idx+1:]
            test_utterances = utterances[:split_idx]
            if not test_utterances or not train_utterances:
                continue  # not enough examples to split
            train += [(utterance, label) for utterance in train_utterances]
            test += [(utterance, label) for utterance in test_utterances]

        #raise Exception(len(train), len(test))

        #random.shuffle(self.data_pairs)
        #split_idx = int(self.count() * test_size)
        #train = IntentDataset(self.data_pairs[split_idx+1:])
        #test = IntentDataset(self.data_pairs[:split_idx])
        return IntentDataset(train), IntentDataset(test)


    class EndOfEpoch(Exception):
        pass
