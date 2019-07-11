import random
from copy import deepcopy
from enum import Enum


class BatchMode(Enum):
    SEQUENTIAL = 1,
    RANDOM = 2,
    BALANCED = 3


class IntentDataset:

    def __init__(self, data_pairs, mode=BatchMode.SEQUENTIAL):
        self.data_pairs = data_pairs
        self.indices = list(range(len(data_pairs[0])))
        # self.data_pairs.append(([], None, []))  # TODO: to avoid bias
        # _, self.labels, _ = zip(*data_pairs)
        self.labels = list(set(self.data_pairs[1]))
        self.utterances = {}
        for i, label in enumerate(data_pairs[1]):
            self.utterances.setdefault(label, []).append(i)
        self.set_mode(mode)

    def set_mode(self, mode: BatchMode):
        self.mode = mode
        if self.mode == BatchMode.SEQUENTIAL:
            self.offset = 0
            random.shuffle(self.indices)

    def _get_balanced(self, batch_size):
        labels = [random.choice(self.labels) for _ in range(batch_size)]
        x, y, z = [], [], []
        for i, label in enumerate(labels):
            index = random.choice(self.utterances[label])
            x.append(self.data_pairs[0][index])
            y.append(self.data_pairs[1][index])
            z.append(self.data_pairs[2][index])
        return x, y, z

    def _get_random(self, batch_size):
        # x, y = zip(*random.choices(self.data_pairs, k=batch_size))
        batch = random.choices(self.labels, k=batch_size)
        x = [self.data_pairs[0][i] for i in batch]
        y = [self.data_pairs[1][i] for i in batch]
        z = [self.data_pairs[2][i] for i in batch]
        return x, y, z

    def _get_next(self, batch_size, drop_remainder=True):
        if self.offset >= self.count():
            raise IntentDataset.EndOfEpoch()
        elif self.offset + batch_size > self.count() and drop_remainder:
            raise IntentDataset.EndOfEpoch()

        batch = self.indices[self.offset:self.offset + batch_size]
        x = [self.data_pairs[0][i] for i in batch]
        y = [self.data_pairs[1][i] for i in batch]
        z = [self.data_pairs[2][i] for i in batch]
        self.offset += batch_size
        return x, y, z

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
        return len(self.indices)

    def get_all(self):
        return self.data_pairs

    def label_count(self):
        return len(self.labels)

    def entity_count(self):
        return len(self.entity_set())

    def entity_set(self) -> set:
        all_entities = [entity for line in self.data_pairs[2] for entity in line]
        return set(all_entities)

    def split(self, test_size=0.1):
        train, test = [], []
        for label, utterances in self.utterances.items():
            split_idx = int(len(utterances) * test_size)
            random.shuffle(utterances)
            train_utterances = utterances[split_idx+1:]
            test_utterances = utterances[:split_idx]
            if not test_utterances or not train_utterances:
                continue  # not enough examples to split
            train += train_utterances#[(utterance, label) for utterance in train_utterances]
            test += test_utterances#[(utterance, label) for utterance in test_utterances]

        train = [item for idx, item in enumerate(list(zip(self.data_pairs))) if idx in train]
        test = [item for idx, item in enumerate(list(zip(self.data_pairs))) if idx in test]

        #random.shuffle(self.data_pairs)
        #split_idx = int(self.count() * test_size)
        #train = IntentDataset(self.data_pairs[split_idx+1:])
        #test = IntentDataset(self.data_pairs[:split_idx])
        return IntentDataset(train), IntentDataset(test)

    def with_negative_sample(self, size=0.5):
        """Adds gibberish sentences to reduce bias."""
        raise NotImplemented()
        # TODO: as separate class - reusable with different seed
        data_pairs = deepcopy(self.data_pairs)
        n = int(len(data_pairs) * size)
        for i in range(n):
            tokens = []
            m = random.randint(1, 10)  # TODO: config, min/max tokens
            for j in range(m):
                # choose a random token from a random sentence
                sentence, _, _ = random.choice(self.data_pairs)
                # print(sentence)
                if not sentence:
                    continue
                # TODO: use vocab here, __init__ will take: dataset, size, vocab
                token = random.choice(sentence.split())
                # it will be reset every epoch, so we don't care if we create a real example once in a time
                tokens.append(token)
            utterance = ' '.join(tokens)
            print(utterance)  # TODO: how about just predicting if it's a valid sentence // LM?
            data_pairs.append((utterance, None, []))
        return IntentDataset(data_pairs)

    class EndOfEpoch(Exception):
        pass
