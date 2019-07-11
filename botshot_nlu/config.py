import os
import yaml
from shutil import copyfile

from botshot_nlu import utils, loader
from botshot_nlu.dataset.intent import IntentDataset
from botshot_nlu.dataset.keywords import StaticKeywordDataset
from botshot_nlu.intent import IntentModel
from botshot_nlu.loader import load_training_examples, as_intent_pairs, as_entity_keywords
from botshot_nlu.pipeline import Pipeline
from botshot_nlu.utils import create_class_instance


class TrainingHelper:

    def __init__(self, config, entities=None, save_path=None, config_dir=None, crossvalidate=False, training_examples=None, testing_examples=None):
        if not isinstance(config, dict):
            raise Exception("Config must be a dict")
        if not save_path:
            print("Warning: Save path not provided, model won't be saved!")
        elif os.path.exists(save_path):
            raise Exception("Save path {} already exists!".format(save_path))

        self.config = config
        self.entities = entities or list(self.config.get("entities", {}).keys())
        self.save_path = save_path
        self.config_dir = config_dir or os.getcwd()
        self.crossvalidate = crossvalidate
        self.training_examples = training_examples
        self.testing_examples = testing_examples

    def start(self):
        if self.save_path:
            os.makedirs(self.save_path)
            self.pipeline_data = {"pipelines": {}}  # stores feature and label encoding
        
        if 'intent' in self.config:
            self.train_intent()
        if 'entities' in self.config:
            self.train_entities()

        if self.save_path:
            self.copy_model_config()  # done last so that we have all training information

    def copy_model_config(self):
        # copy keyword files to prevent surprises during inference
        # note: for dynamic keywords, take a look at the Providers API
        sources = self.config["input"].get("keywords", [])
        fixed_sources = []
        keywords_dir = os.path.join(self.save_path, "keywords")
        os.makedirs(keywords_dir, exist_ok=True)
        for filename in sources:
            if not os.path.isabs(filename):
                filename = os.path.join(self.config_dir, filename)
            path_to = os.path.join(keywords_dir, os.path.basename(filename))
            copyfile(filename, path_to)
            fixed_sources.append(os.path.join("keywords", os.path.basename(filename)))
        
        if self.config.get("keywords_from_examples"):
            # entities from examples will be converted to keywords, useful when importing from Wit.ai
            examples = self._get_training_examples()
            keywords = as_entity_keywords(examples)
            path_to = os.path.join(keywords_dir, "keywords_from_examples.yml")
            with open(path_to, "w") as fp:
                yaml.dump({"entities": keywords}, fp)
            fixed_sources.append(os.path.join("keywords", "keywords_from_examples.yml"))

        # update new config file to point to correct (relative!) paths
        self.config["input"]["keywords"] = fixed_sources

        # copy config and pipeline files
        with open(os.path.join(self.save_path, "config.yml"), "w") as fp:
            yaml.dump(self.config, fp)
        with open(os.path.join(self.save_path, "pipeline.yml"), "w") as fp:
            yaml.dump(self.pipeline_data, fp)

    def _get_training_examples(self) -> list:
        # if training examples were explicitly specified, load those
        if self.training_examples:
            return loader.read_datasets(self.training_examples)
        # otherwise, find all sources in config file
        sources = self.config["input"]["examples"].copy()
        if not sources:
            raise Exception("No source files with training examples were specified")
        elif not isinstance(sources, list):
            sources = [sources]
        # convert paths relative to config directory to absolute paths
        for i, filename in enumerate(sources):
            if not os.path.isabs(filename):
                abs_filename = os.path.join(self.config_dir, filename)
                sources[i] = abs_filename
        return loader.read_datasets(*sources)

    def _load_intent_dataset(self) -> IntentDataset:
        examples = self._get_training_examples()
        dataset = IntentDataset(data_pairs=examples)
        return dataset

    def _get_intent_models(self):
        intent_config = self.config["entities"]['intent']
        pipeline = utils.create_pipeline(intent_config['pipeline'], intent_config.get('add', []), intent_config)
        model = ner_model = None
        if intent_config.get('model'):
            model = create_class_instance(intent_config.get('model'), config=intent_config, pipeline=pipeline)  # type: IntentModel
        if intent_config.get('ner-model'):
            ner_model = create_class_instance(intent_config.get('ner-model'), config=intent_config, pipeline=pipeline)  # type: EntityModel
        return pipeline, model, ner_model

    def train_intent(self):
        print("Training intent")
        dataset = self._load_intent_dataset()#.with_negative_sample(size=0.5)
        models = load_intent_models(self.config)
        if self.crossvalidate:
            raise NotImplemented()
            self.cross_validate_intent(model, dataset)
            input('Press enter to continue')

        for model in models:
            metrics = model.train(dataset)

        if self.testing_examples:
            testing_examples = load_training_examples(self.testing_examples)
            dataset = IntentDataset(data_pairs=as_intent_pairs(testing_examples))
            for model in models:
                metrics = model.test(dataset)
                print("Metrics for %s:" % model.__class__.__name__, metrics)

        if self.save_path:
            pipeline_data = {"pipelines": {}}
            for i, model in enumerate(models):
                model.save(os.path.join(self.save_path, "intent_%d" % i))
                pipeline_data['pipelines']['intent_%d' % i] = model.pipeline.save()
                model.unload()
            self.pipeline_data = pipeline_data

    def cross_validate_intent(self, model, dataset, k=10):
        print("Starting %d-fold cross validation" % k)
        pipeline, model, ner_model = self._get_intent_model()
        accuracies = []

        for _ in range(k):
            train, test = dataset.split()
            model.train(train)
            metrics = model.test(test)
            accuracies.append(metrics.accuracy)
        print(accuracies)
        print("Mean accuracy: %f" % (sum(accuracies) / len(accuracies)))

    def train_entities(self):
        for entity in self.config["entities"]:
            if entity == 'intent': continue
            entity_config = self.config['entities'][entity]
            self.train_entity(entity, entity_config)
    
    def train_entity(self, entity, entity_config):
        print("Training entity %s" % entity)
        # TODO: choose between keywords and context
        if 'keyword_model' in entity_config:
            pass


def read_training_config(filename):
    if not filename or not os.path.exists(filename):
        raise Exception("The training config file {} doesn't exist".format(filename))
    with open(filename) as fp:
        config = yaml.safe_load(fp)
    return config


class ParseHelper:

    def __init__(self, config, models):
        self.config = config
        self.models = models
    
    @staticmethod
    def load(model_path):
        with open(os.path.join(model_path, "config.yml")) as fp:
            config = yaml.safe_load(fp)
        with open(os.path.join(model_path, "pipeline.yml")) as fp:
            pipeline_data = yaml.safe_load(fp)

        models = []
        if 'intent' in config:
            models += load_intent_models(config, model_path, pipeline_data)
        if 'entities' in config:
            keyword_datasets = ParseHelper.load_keyword_datasets(config, model_path)
            models += ParseHelper.load_keyword_models(config, keyword_datasets)

        if len(models) <= 0:
            raise Exception("No models were loaded")

        return ParseHelper(config, models)

    def parse(self, text):
        results = {}
        for model in self.models:
            result = model.predict(text)
            results.update(result)
        return results

    @staticmethod
    def load_keyword_datasets(config: dict, config_dir: str):
        datasets = []

        # load keyword files
        sources = config['input'].get('keywords', [])
        if sources:
            if not isinstance(sources, list):
                sources = [sources]
            for i, filename in enumerate(sources):
                print(filename)
                if not os.path.isabs(filename):
                    abs_filename = os.path.join(config_dir, filename)
                    sources[i] = abs_filename
            dataset = StaticKeywordDataset.load(*sources)
            datasets.append(dataset)
        
        # load dynamic providers
        providers = config['input'].get('providers', [])
        for item in providers:
            if isinstance(item, str):
                provider = create_class_instance(item)
            elif isinstance(item, dict):
                for provider_cls, params in item.items():
                    if isinstance(params, list):
                        provider = create_class_instance(provider_cls, *params)
                    elif isinstance(params, dict):
                        provider = create_class_instance(provider_cls, **params)
                    else:
                        raise Exception("Can't instantiate provider %s: parameters should be list or dict" % provider_cls)
                    datasets.append(provider)
            else:
                raise Exception("Providers config is malformed")
        
        # load from (intent-)examples files

        return datasets

    @staticmethod
    def load_keyword_models(config: dict, datasets: list):
        models_spec = {}
        models = []
        for entity, entity_conf in config['entities'].items():
            keywords_config = entity_conf.get('keywords')
            if keywords_config:
                key = yaml.dump(keywords_config)
                models_spec.setdefault(key, []).append(entity)
        # all different (model, params) configurations
        for model_spec, entities in models_spec.items():
            model_spec = yaml.load(model_spec)
            model_cls = model_spec['model']
            required_datasets = [dataset for dataset in datasets if any(set(entities) & dataset.get_entities())]
            all_data = []
            for dataset in required_datasets:
                for kw in dataset.get_data(entities).values():
                    all_data += _get_examples(kw)
            pipeline = utils.create_pipeline(model_spec['pipeline'], intent_config.get('add', []), model_spec)
            pipeline.fit(all_data, y=None)
            model = utils.create_class_instance(model_cls, config=model_spec, entities=entities, datasets=required_datasets, pipeline=pipeline, resources=None)  # FIXME
            models.append(model)
        return models


def _get_examples(item):
    if isinstance(item, str):
        return [item]
    elif isinstance(item, dict):
        examples = []
        for value in item.values():
            examples += _get_examples(value)
        return examples
    elif isinstance(item, list):
        examples = []
        for i in item:
            print(_get_examples(i))
            examples += _get_examples(i)
        return examples
    return []


def get_model_configs(config: dict):
    if 'intent' not in config:
        raise Exception("Intent model not defined")
    if isinstance(config['intent'], dict):
        return [config['intent']]
    elif isinstance(config['intent'], list):
        return config['intent']
    raise Exception("Intent config is None")


def load_intent_models(config: dict, config_dir=None, pipeline_data=None):
    models = []
    for i, cfg in enumerate(get_model_configs(config)):
        print(i, cfg)
        pipeline = utils.create_pipeline(cfg['pipeline'], cfg.get('add', []), cfg)
        if pipeline_data is not None:
            pipeline.load(pipeline_data['pipelines']['intent_%d' % i])
        model = create_class_instance(cfg['model'], config=cfg, pipeline=pipeline)  # type: IntentModel
        if config_dir is not None:
            model.load(os.path.join(config_dir, 'intent_%d' % i))
        models.append(model)
    return models
