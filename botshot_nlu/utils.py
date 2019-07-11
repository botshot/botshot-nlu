import importlib

from botshot_nlu.pipeline import Pipeline


def create_class_instance(cls: str, *args, **kwargs):
    # TODO: always return same object for same specification
    if not isinstance(cls, str):
        raise ValueError("Expected classname as string, not {}".format(type(cls)))

    module_name, cls_name = cls.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    clz = getattr(module, cls_name)

    try:
        return clz(*args, **kwargs)
    except TypeError as ex:
        raise Exception("Can't instantiate class {}! See exception above.".format(cls)) from ex


def get_default_tokenizer():
    from botshot_nlu.tokenizer.whitespace import WhitespaceTokenizer
    return WhitespaceTokenizer(config=None)


def create_pipeline(components: list, add:list, config: dict):
    objects = []
    for component in components:
        component = create_class_instance(component, config=config)
        objects.append(component)
    pipeline = Pipeline(*objects, add=[create_class_instance(x, config) for x in add])
    return pipeline
