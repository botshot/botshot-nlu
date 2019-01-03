import os

import yaml

from botshot_nlu.dataset.intent import IntentDataset
from botshot_nlu.intent import IntentModel
from botshot_nlu.loader import load_training_data, as_intent_pairs
from botshot_nlu.pipeline import Pipeline
from botshot_nlu.utils import create_class_instance


class TrainingHelper:

    def __init__(self, config, entities=None, save_path=None, config_dir=None, crossvalidate=False):
        if not isinstance(config, dict):
            raise Exception("Config must be a dict")
        if not save_path:
            print("Warning: Save path not provided, model won't be saved!")
        elif os.path.exists(save_path) and not os.path.isdir(save_path):
            raise Exception("Save path {} already exists and isn't a directory".format(save_path))

        self.config = config
        self.entities = entities or list(self.config.get("entities", {}).keys())
        self.save_path = save_path
        self.config_dir = config_dir or os.getcwd()
        self.crossvalidate = crossvalidate

        sources = self.config.get('sources')
        if not sources:
            raise Exception("No source files with training examples were specified")
        elif not isinstance(sources, list):
            sources = [sources]
        for i, filename in enumerate(sources):
            if not os.path.isabs(filename):
                abs_filename = os.path.join(self.config_dir, filename)
                sources[i] = abs_filename
        self.data = load_training_data(*sources)

    def start(self):
        if 'intent' in self.config.get("entities", {}):
            self.train_intent()
        self.train_entities()

    def _get_intent_model_dataset(self):
        intent_config = self.config["entities"]['intent']

        tokenizer = create_class_instance(intent_config.get('tokenizer'), config=intent_config)
        featurizer = create_class_instance(intent_config.get('featurizer'), config=intent_config)
        pipeline = Pipeline(tokenizer=tokenizer, featurizer=featurizer)

        dataset = IntentDataset(data_pairs=as_intent_pairs(self.data))
        model = create_class_instance(intent_config.get('model'), config=intent_config, pipeline=pipeline)  # type: IntentModel

        return model, dataset

    def train_intent(self):
        print("Training intent")
        model, dataset = self._get_intent_model_dataset()
        if self.crossvalidate:
            self.cross_validate_intent(model, dataset)
            input('Press enter to continue')
        metrics = model.train(dataset)
        if self.save_path:
            model.save(self.save_path)
        model.unload()

    def cross_validate_intent(self, model, dataset, k=10):
        print("Starting %d-fold cross validation" % k)
        model, dataset = self._get_intent_model_dataset()
        accuracies = []

        for _ in range(k):
            train, test = dataset.split()
            model.train(train)
            metrics = model.test(test)
            accuracies.append(metrics.accuracy)
        print(accuracies)
        print("Mean accuracy: %f" % (sum(accuracies) / len(accuracies)))

    def train_entities(self):
        pass


def read_training_config(filename):
    if not filename or not os.path.exists(filename):
        raise Exception("The training config file {} doesn't exist".format(filename))
    with open(filename) as fp:
        config = yaml.safe_load(fp)
    return config
