import logging
import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd

from .model import Model
from .transformer import Transformer


@dataclass
class Pipeline:
    """
    Class for pipeline that transforms data, fits a model and makes 
    predictions.
    """
    transformers: List[Transformer]
    model: Model

    def __post_init__(self):
        self.save_fname = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the transformers, transforms the data and returns transformed
        dataframe.
        """
        for transformer in self.transformers:
            transformer.fit(X)
            X = transformer.transform(X)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data in X using already fit transformers.
        """
        for transformer in self.transformers:
            X = transformer.transform(X)

        return X

    def model_fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Fits the model based on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions for X after transforming it.
        """
        for transformer in self.transformers:
            X = transformer.transform(X)

        return self.model.predict(X)

    def model_fit_predict(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the model using X_train and y_train, and then makes predictions
        using X_test.
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Fit-transforms X_train using the transformers, trains model based on
        X_train and y_train, and transforms and makes predictions for X_test
        """
        X_train = self.fit_transform(X_train)
        X_test = self.transform(X_test)
        return self.model_fit_predict(X_train, y_train, X_test)

    def save(self, base_directory = 'models') -> None:
        """
        Saves the pipeline to a pickle.
        """
        # Save pipeline
        save_folder = self.model.__class__.__name__
        time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        name = f'{self.model.__class__.__name__}_{time}'
        with open(f'./{base_directory}/{save_folder}/{name}.pickle', 'wb+') as f:
            pickle.dump(self, f)

        # Save model params
        with open(f'./{base_directory}/{save_folder}/{name}_model_params.pickle', 'wb+') as f:
            pickle.dump(self.model.get_params(), f)

        # Save transformer params
        tr_params = {}
        for transformer in self.transformers:
            tr_params[transformer.__class__.__name__] = transformer.get_attrs()
        with open(f'./{base_directory}/{save_folder}/{name}_trans_params.pickle', 'wb+') as f:
            pickle.dump(tr_params, f)

        self.save_fname = name

        # Log that saving was successful
        loc = f'./{base_directory}/{save_folder}/{name}.pickle'
        logging.info(f'Saved pipeline to: {loc}')

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """
        Saves predictions to file.
        """
        save_folder = self.model.__class__.__name__
        name = self.save_fname
        predictions.to_csv(f'./results/{save_folder}/{name}_predictions.csv', index = False)

        # Log that saving was successful
        loc = f'./results/{save_folder}/{name}_predictions.csv'
        logging.info(f'Saved results to: {loc}')

    def save_cv(self, cv_result: dict) -> None:
        """
        Saves cross validation results to file.
        """
        save_folder = self.model.__class__.__name__
        name = self.save_fname
        # Save model params
        with open(f'./cv_results/{save_folder}/{name}_cv_results.pickle', 'wb+') as f:
            pickle.dump(cv_result, f)

        # Log that saving was successful
        loc = f'./cv_results/{save_folder}/{name}_cv_results.pickle'
        logging.info(f'Saved CV results to: {loc}')
