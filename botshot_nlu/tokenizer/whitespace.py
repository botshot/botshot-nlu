import re


from botshot_nlu.tokenizer import Tokenizer


class WhitespaceTokenizer(Tokenizer):

    def transform(self, sentences):
        out = []
        for sent in sentences:
            out.append(self.tokenize(sent))
        return out

    def tokenize(self, sentence: str):
        sentence = re.sub("[.,;!?\t\n]", " ", sentence)
        return [token for token in sentence.split(" ") if token]
