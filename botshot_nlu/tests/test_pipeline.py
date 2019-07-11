import pytest

from botshot_nlu.pipeline import Pipeline
from botshot_nlu.tokenizer.whitespace import WhitespaceTokenizer


class TestPipeline:

    def test_process_empty(self):
        texts = ["hello", "how are you", "xaxaxa"]
        labels = [{"intent": "hi"}, {"intent": "hey"}, {"intent": None}]
        pipeline = Pipeline()
        assert pipeline.transform(texts) == (texts, None, None)
        assert pipeline.transform(texts, labels) == (texts, labels, None)

    def test_process_tokenize(self):
        texts = ["hello", "how are you", "xaxaxa"]
        labels = [{"intent": "hi"}, {"intent": "hey"}, {"intent": None}]
        pipeline = Pipeline(WhitespaceTokenizer(config={}))
        assert pipeline.transform(texts, None, None) == ([t.split() for t in texts], None, None)
        # assert pipeline.transform(texts, labels)[1] == labels TODO
