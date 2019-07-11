from typing import Optional

from symspellpy import symspellpy

from botshot_nlu.pipeline import PipelineComponent, InputType


class SymSpell(PipelineComponent):

    MAX_EDIT_DISTANCE = 2
    PREFIX_LENGTH = 7

    source = InputType.TEXT
    target = InputType.TEXT

    def fit(self, data, labels, entities):
        self.corpus = {}
        for line in data:
            for word in line:
                self.corpus[word] = self.corpus.get(word, 0) + 1
        self._init(self.corpus)

    def load(self, saved_data: dict):
        self.corpus = saved_data['spell_corpus']
        self._init(self.corpus)

    def save(self) -> Optional[dict]:
        return {"spell_corpus": self.corpus}

    def _init(self, corpus):
        self.sym_spell = symspellpy.SymSpell(self.MAX_EDIT_DISTANCE, self.PREFIX_LENGTH)
        for word, cnt in corpus.items():
            self.sym_spell.create_dictionary_entry(word, cnt)

    def transform(self, data, labels, entities):
        if labels is not None:
            return data, labels, entities  # don't apply on training data
        X = []
        for sentence in data:
            suggestions = self.sym_spell.lookup_compound(sentence, self.MAX_EDIT_DISTANCE)
            if suggestions:
                new = suggestions[0].term
                print("Correcting \"{}\" to \"{}\"".format(sentence, new))
                X.append(new)
            else:
                X.append(sentence)
        return X, None, None

    def feature_dim(self):
        return -1
