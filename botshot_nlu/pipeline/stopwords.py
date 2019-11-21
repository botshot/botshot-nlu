import stop_words

from botshot_nlu.pipeline import PipelineComponent, InputType


class Stopwords(PipelineComponent):

  source = InputType.TOKENS
  target = InputType.TOKENS

  def __init__(self, config):
      super().__init__(config)
      self.lang = config.get("language", "en")
      self.words = stop_words.get_stop_words(self.lang)

  def fit(self, x, y, z, *args, **kwargs):
      return x, y, z

  def transform(self, texts, y, z):
      clean_texts = []
      for text in texts:
          text = [token for token in text if token.lower() not in self.words]
          clean_texts.append(text)
      return clean_texts, y, z

  def load(self, *args, **kwargs):
      pass

  def save(self, *args, **kwargs):
      pass

  def feature_dim(self, *args, **kwargs):
      return None

