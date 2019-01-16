from botshot_nlu.pipeline import PipelineComponent


class Tokenizer(PipelineComponent):

    def fit(self, data):
        pass

    def load(self, from_data):
        pass

    def save(self):
        return None

    def feature_dim(self):
        return None

    def tokenize(self, sentence):
        pass
