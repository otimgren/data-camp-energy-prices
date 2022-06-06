from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import sem

from ..core import Model, Pipeline
from .metrics import Metric


@dataclass
class CrossValidator:
    metrics: List[Metric]
    n_folds: int = 5
    target_name: str = 'POMP'

    def generate_folds(self, X:pd.DataFrame)-> pd.DataFrame:
        """
        Adds a fold column to the input dataframe.
        """
        X = X.copy()

        # Split data into folds for use in cross validation
        date_max = X.index.max()
        day = pd.Timedelta(1,'D')
        X['fold'] = 0
        for i, n in enumerate(range(self.n_folds,0,-1)):
            X.loc[date_max-(i+1)*100*day:date_max-i*100*day, 'fold'] = n

        return X


    def generate_train_test(self, X: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame]:
        """
        Generates train and test datasets for a given test fold
        """
        X = X.copy()
        
        train = X[X.fold < fold]
        # y_train = train.pop('price')
        # X_train = train.drop(columns = ['demand','fold'])
        
        test = X[X.fold == fold]
        # y_test = test.pop('price')
        # X_test = test.drop(columns = ['demand', 'fold'])

        return train, test

    def cross_validate(self, model: Union[Model, Pipeline], df: pd.DataFrame):
        """
        Cross validates the given model or pipeline
        """
        # Make copy of train dataframe
        df = df.copy()

        # Initialize dictionary to store results
        results = {}
        for metric in self.metrics:
            results[metric.__class__.__name__] = []

        # Loop over folds
        for fold in range(1, self.n_folds):
            train, test = self.generate_train_test(df, fold)
            preds, y_true = model.fit_predict(train, test)
            preds['true_price'] = y_true
            preds['predicted_buy'] = (preds.predicted_price - preds.price_1) > 0

            # Calculate data needed for calculating metrics
            preds['true_buy'] = (preds.true_price - preds.price_1) > 0
            preds['profit'] = (preds.true_price - preds.price_1)*preds.predicted_buy*70
            preds['max_profit'] = (preds.true_price - preds.price_1)*preds.true_buy*70


            # Calculate metrics and store results
            for metric in self.metrics:
                results[metric.__class__.__name__].append(metric.eval(preds))

        return results

    def print_cv_results(self, results: dict) -> None:
        """
        Prints summary statistics for the CV results (mean+/-sem)
        """
        for metric, values in results.items():
            print(f"{metric}:\t{np.mean(values):.3E}+/-{sem(values):.3E}")

