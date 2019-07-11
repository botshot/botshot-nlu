import re
from bisect import bisect
from copy import deepcopy


from botshot_nlu.tokenizer import Tokenizer


class WhitespaceTokenizer(Tokenizer):

    def transform(self, sentences, labels, entities):

        if labels is not None:
            return sentences, labels, entities  # TODO for now, training data is already tokenized

        x = []
        for i, sent in enumerate(sentences):
            x.append(self.tokenize(sent)[0])
            # if labels is None:
            #     x.append(self.tokenize(sent))
            # else:
            #     x_, y_ = self.tokenize(sent, labels[i])
            #     x.append(x_)
            #     y.append(y_)

        return x, labels, entities

    def tokenize(self, sentence: str, labels=None, entities=None):
        # always single character, doesn't change positions
        sentence = re.sub("[.,;!?\t\n]", " ", sentence)
        # get all split positions
        positions = [i for i, x in enumerate(sentence) if x == " "]
        tokens = sentence.split(" ")
        empty_tokens = [i for i, token in enumerate(tokens) if token == '']
        if labels and labels.get("entities"):
            labels = deepcopy(labels)
            for entity in labels['entities']:
                # find which token this is, then remove empty tokens
                # TODO: unit test
                start = bisect(positions, entity['start'])
                start -= len([i for i in empty_tokens if i < start])
                start = max(start, 0)

                end = bisect(positions, entity['end'])
                end -= len([i for i in empty_tokens if i < end])
                end = max(end, 0)

                entity['start'], entity['end'] = start, end

        tokens = [token for token in tokens if token != '']
        return tokens, labels, entities
