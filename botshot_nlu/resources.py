from botshot_nlu.pipeline.embedding import Embedding


class Resources:

    def __init__(self):
        self.embeddings = {}

    def require_embedding(self, filename):
        if filename in self.embeddings:
            return self.embeddings[filename]
        else:
            embedding = Embedding(filename)
            self.embeddings[filename] = embedding
            return embedding

    def get_embedding(self, filename):
        if filename in self.embeddings:
            return self.embeddings[filename]
        return RuntimeError("Required embedding file %s not loaded" % filename)
