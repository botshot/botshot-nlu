import os
import yaml
from shutil import copyfile


from botshot_nlu.dataset.intent import IntentDataset
from botshot_nlu.dataset.keywords import StaticKeywordDataset
from botshot_nlu.intent import IntentModel
from botshot_nlu.loader import load_training_examples, as_intent_pairs, as_entity_keywords
from botshot_nlu.pipeline import Pipeline
from botshot_nlu.utils import create_class_instance


class TrainingHelper:

    def __init__(self, config, entities=None, save_path=None, config_dir=None, crossvalidate=False):
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

    def start(self):
        if self.save_path:
            os.makedirs(self.save_path)
            self.pipeline_data = {"pipelines": {}}  # stores feature and label encoding
        
        if 'intent' in self.config.get("entities", {}):
            self.train_intent()
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
        sources = self.config["input"]["examples"].copy()
        if not sources:
            raise Exception("No source files with training examples were specified")
        elif not isinstance(sources, list):
            sources = [sources]
        for i, filename in enumerate(sources):
            if not os.path.isabs(filename):
                abs_filename = os.path.join(self.config_dir, filename)
                sources[i] = abs_filename
        examples = load_training_examples(*sources)
        return examples

    def _load_intent_dataset(self) -> IntentDataset:
        examples = self._get_training_examples()
        dataset = IntentDataset(data_pairs=as_intent_pairs(examples))
        return dataset

    def _get_intent_model(self):
        intent_config = self.config["entities"]['intent']

        tokenizer = create_class_instance(intent_config.get('tokenizer'), config=intent_config)
        featurizer = create_class_instance(intent_config.get('featurizer'), config=intent_config)
        pipeline = Pipeline(tokenizer=tokenizer, featurizer=featurizer)
        
        model = create_class_instance(intent_config.get('model'), config=intent_config, pipeline=pipeline)  # type: IntentModel
        return pipeline, model

    def train_intent(self):
        print("Training intent")
        dataset = self._load_intent_dataset()
        pipeline, model = self._get_intent_model()
        if self.crossvalidate:
            self.cross_validate_intent(model, dataset)
            input('Press enter to continue')
        metrics = model.train(dataset)
        if self.save_path:
            model.save(os.path.join(self.save_path, "intent"))
            self.pipeline_data['pipelines']['intent'] = pipeline.save()
        model.unload()

    def cross_validate_intent(self, model, dataset, k=10):
        print("Starting %d-fold cross validation" % k)
        pipeline, model = self._get_intent_model()
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
        if 'intent' in config['entities']:
            intent_model = ParseHelper.load_intent_model(config, model_path, pipeline_data)
            models.append(intent_model)
        keyword_datasets = ParseHelper.load_keyword_datasets(config, model_path)
        models += ParseHelper.load_keyword_models(config, keyword_datasets)
        
        return ParseHelper(config, models)

    def parse(self, text):
        results = {}
        for model in self.models:
            result = model.predict(text)
            results.update(result)
        return results

    @staticmethod
    def load_intent_model(config: dict, config_dir: str, pipeline_data):
        entity_config = config['entities']['intent']
        tokenizer = create_class_instance(entity_config.get('tokenizer'), config=entity_config)
        featurizer = create_class_instance(entity_config.get('featurizer'), config=entity_config)
        pipeline = Pipeline(tokenizer=tokenizer, featurizer=featurizer)
        pipeline.load(pipeline_data['pipelines']['intent'])
        model = create_class_instance(entity_config.get('model'), config=entity_config, pipeline=pipeline)  # type: IntentModel
        model.load(os.path.join(config_dir, 'intent'))
        return model

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
            if entity == 'intent': continue
            keywords_config = entity_conf.get('keywords')
            if keywords_config:
                key = frozenset(keywords_config.items())
                models_spec.setdefault(key, []).append(entity)
        for model_spec, entities in models_spec.items():
            model_spec = dict(model_spec)
            model_cls = model_spec['model']
            required_datasets = [dataset for dataset in datasets if any(set(entities) & dataset.get_entities())]
            model = create_class_instance(model_cls, config=model_spec, entities=entities, datasets=required_datasets)
            models.append(model)
        return models
