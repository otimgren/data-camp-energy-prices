from abc import ABC, abstractmethod

import pandas as pd


class TransformerABC(ABC):
    """
    Abstract base class for transformer classes
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fits the transformer based on data in X.
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the given dataframe X
        """

class Transformer(TransformerABC):
    """
    Base class for transformers
    """
    def __init__(self):
        self.default_attrs = {}

    def __post_init__(self):
        self.default_attrs = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the transformer and then transforms the data, and returns
        transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def set_attrs(self, attrs) -> None:
        """
        Set attributes for the transformer
        """
        for key, value in attrs.items():
            if key in self.default_attrs.keys():
                setattr(self, key, value)

    def get_attrs(self) -> None:
        """
        Get attributes for the transformer
        """
        attrs = {}
        for key in self.default_attrs.keys():
            attrs[key] = getattr(self, key)

        return attrs
