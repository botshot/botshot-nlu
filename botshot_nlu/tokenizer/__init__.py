from botshot_nlu.pipeline import PipelineComponent, InputType


class Tokenizer(PipelineComponent):

    source = InputType.TEXT
    target = InputType.TOKENS

    def fit(self, data, labels=None, entities=None):
        pass

    def load(self, from_data):
        pass

    def save(self):
        return None

    def feature_dim(self):
        return None

    def tokenize(self, sentence, labels=None, entities=None):
        pass
