import copy
from abc import ABC, abstractmethod

import pandas as pd


class ModelABC(ABC):
    """
    Abstract parent class for models
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fits the model using the training data in train.
        """

    @abstractmethod
    def predict(self, X) -> pd.DataFrame:
        """
        Makes predictions based on data in X
        """

    @abstractmethod
    def set_params(self,**params) -> None:
        """
        Set parameters for the model
        """

    @abstractmethod
    def get_params(self) -> None:
        """
        Get parameters for the model
        """

class Model(ModelABC):
    """
    Parent class for models.
    """

    def fit_predict(self, X_train, y_train, X_test) -> pd.DataFrame:
        """
        Fits the model based on training data and then makes predictions on test
        data.
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def copy(self):
        """
        Returns a deep copy of the model.
        """
        return copy.deepcopy(self)

    def set_attrs(self, attrs) -> None:
        """
        Set attributes for the model
        """
        for key, value in attrs.items():
            if key in self.default_attrs.keys():
                setattr(self, key, value)

    def get_attrs(self) -> None:
        """
        Get attributes for the model
        """
        attrs = {}
        for key in self.default_attrs.keys():
            attrs[key] = getattr(self, key)

        return attrs

    
