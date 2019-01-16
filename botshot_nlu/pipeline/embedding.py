import json

import numpy as np
import os


class Embedding:

    def __init__(self, filename, max_cache_size=1024):

        # TODO use R-tree for similarity queries

        if not os.path.exists(filename):
            raise FileNotFoundError("The embedding file {} doesn't exist.".format(filename))

        self.fp = open(filename, 'r')
        self.dimension = int(self.fp.readline().split(" ")[1])
        self.max_cache_size = max_cache_size

        # cache word offsets in the embedding file for quick lookup
        pos_path = filename + ".offsets"
        if os.path.exists(pos_path):
            with open(pos_path, 'r') as pos_f:
                self.positions = json.load(pos_f)
        else:
            with open(pos_path, 'w') as pos_f:
                print("Creating word index ...")
                # FIXME open exclusively, other celery thread might screw up
                self.positions = self.cache_positions()
                json.dump(self.positions, pos_f)

        # LRU cache to limit disk reads
        self.cache = {}
        self.recent = []

    def cache_positions(self):

        self.fp.seek(0)
        positions = {}

        self.fp.readline()  # header
        off = self.fp.tell()
        line = self.fp.readline()

        while line:
            word = line.split(" ", maxsplit=1)[0]

            # the lower-case word will be tried first when embedding
            # if it doesn't exist, it will be set to the first mixed-case word
            # TODO maybe we could always take the first one ( the most common )
            if word.lower() not in positions:
                positions[word.lower()] = off

            # this also ensures that existing lowercase words are not overwritten
            positions[word] = off
            off = self.fp.tell()
            line = self.fp.readline()

        return positions

    def get_vector(self, word: str, ngrams=3):
        v = np.zeros([self.dimension])  # TODO: use <unk> if present instead
        cnt = 0
        for i in range(len(word) - self.dimension + 1):
            substr = word[i:i+self.dimension]
            u = self.word2vec(substr)
            if u is None: continue
            v += u
            cnt += 1
        u = self.word2vec(word)
        if u is not None:
            v += u
            cnt += 1
        return v / cnt if cnt else v

    def word2vec(self, word):

        if not isinstance(word, str):
            raise ValueError("Word must be a string, but is {}".format(type(word)))

        if word.lower() in self.cache or word.lower() in self.positions:
            word = word.lower()

        # search in cache
        if word in self.cache:
            self.recent.remove(word)
            self.recent.insert(0, word)
            return self.cache[word]

        # search in embedding file
        if word in self.positions:
            off = self.positions[word]
            self.fp.seek(off)
            line = self.fp.readline().split(" ")
            vector = np.array([float(x) for x in line[1:]])

            if word.lower() == line[0].lower():

                # remove oldest cached item if cache is full
                if len(self.recent) >= self.max_cache_size:
                    removed_word = self.recent.pop()
                    del self.cache[removed_word]

                # add this word to cache
                self.recent.insert(0, word)
                self.cache[word] = vector

                return vector
            else:
                raise Exception("Word2vec cache is corrupt, have you modified the embedding file?")

        # TODO construct from sub-word vectors

        return None
