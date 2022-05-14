import pickle

import optuna

from ..core import Model, Pipeline


def load_optimized_pipeline(fname: str) -> Pipeline:
    """
    Load optimized pipeline with given name from file.
    """
    with open(f'../optimized_models/{fname}', 'rb') as f:
        pipeline = pickle.load(f)

    return pipeline

def load_optimized_params(fname: str) -> dict:
    """
    Load optimized parameters for a given model.
    """
    #TO DO
    pass
