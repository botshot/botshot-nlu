import re


from botshot_nlu.tokenizer import Tokenizer


class WhitespaceTokenizer(Tokenizer):

    def transform(self, sentences):
        out = []
        for sent in sentences:
            sent = re.sub("[.,;!?\t\n]", " ", sent)
            out.append(sent.split(" "))
        return out
