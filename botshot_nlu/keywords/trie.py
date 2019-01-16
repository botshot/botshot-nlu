from botshot_nlu import utils
from . import KeywordExtractor


class TrieKeywordExtractor(KeywordExtractor):

    def __init__(self, config, entities, datasets):
        if 'tokenizer' in config:
            self.tokenizer = utils.create_class_instance(config['tokenizer'], config=None)
        else:
            self.tokenizer = utils.get_default_tokenizer()
        super().__init__(config, entities, datasets)
        # TODO: configurable tokenizer + stemmer + other preprocessing
    
    def _reformat(self, data):
        data_tuples = []
        for entity, values in data.items():
            for value in values:
                if isinstance(value, str):
                    data_tuples.append((value, value, entity))
                elif isinstance(value, dict):
                    for label, expressions in value.items():
                        for expression in expressions:
                            data_tuples.append((expression, label, entity))
        return data_tuples

    def update(self, data):
        data = self._reformat(data)
        root_state = {}

        for expression, label, entity in data:
            # expression = unidecode.unidecode(expression)
            # words = tokenize(expression, self.should_stem, language=self.language)
            words = self.tokenizer.tokenize(expression.lower())

            first_word = words[0]
            chars = list(first_word)
            state = root_state
            for idx, char in enumerate(chars):
                transitions = state.setdefault('transitions', {})
                state = transitions.setdefault(str(char), {})

            outputs = state.setdefault('outputs', [])
            outputs.append({
                "requires": words[1:],
                "label": label,
                "entity": entity
            })

        self.trie = root_state

    def predict(self, utterance: str):
        extracted = {}
        # text = unidecode.unidecode(text)
        # words = tokenize(text, self.should_stem, self.language)
        utterance = self.tokenizer.tokenize(utterance)

        for idx, word in enumerate(utterance):
            word = word.lower()  # TODO move elsewhere
            state = self.trie
            chars = list(word)
            for char in chars:
                state = state.get('transitions', {}).get(char)
                if not state:
                    break  # unknown word
            if state:
                outputs = state.get('outputs', [])
                outputs.sort(key=lambda x: len(x.get('requires', [])), reverse=True)
                following = set(utterance[idx + 1:])
                for output in outputs:
                    required = set(output.get('requires', []))
                    if required & following == required:
                        entity = output['entity']
                        label = output['label']
                        extracted.setdefault(entity, []).append({"value": label})

        return extracted
