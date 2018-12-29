import importlib


def create_class_instance(cls: str, **kwargs):
    if not isinstance(cls, str):
        raise ValueError("Expected classname as string, not {}".format(type(cls)))

    module_name, cls_name = cls.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    clz = getattr(module, cls_name)

    try:
        return clz(**kwargs)
    except TypeError as ex:
        raise Exception("Can't instantiate class {}! See exception above.".format(cls)) from ex
