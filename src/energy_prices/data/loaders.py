"""
Functions for loading data
"""
from typing import Tuple

import pandas as pd

from ..config import TEST_PATH, TRAIN_PATH


def load_train() -> pd.DataFrame:
    """
    Loads the training data.

    returns:
    train   : training data in a pandas DataFrame
    """
    train = pd.read_csv(TRAIN_PATH)
    return train


def load_test() -> pd.DataFrame:
    """
    Loads the test data.

    returns:
    test    : test data in a pandas DataFrame
    """
    test = pd.read_csv(TEST_PATH)
    return test


def load_train_and_test() -> Tuple[pd.DataFrame]:
    """
    Loads the train and test data from file
    """

    return load_train(), load_test()
