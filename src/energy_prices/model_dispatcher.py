import importlib


def dispatch_model(model_name: str):
    """
    Returns a model with the given name.
    """

    module = importlib.import_module('titanic.models')
    model_class = getattr(module, model_name)

    return model_class()
