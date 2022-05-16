from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


class Metric(ABC):
    """
    Parent class for metrics
    """

    @abstractmethod
    def eval(self, df: pd.DataFrame) -> float:
        """
        Evaluates the metric for given predictions and truths
        """

class CorrectBuyPercentage(Metric):
    """
    Metric that calculates the proportion of cases where the prediction
    results in making the correct decision to purchase.
    """
    def eval(self, df:pd.DataFrame) -> float:
        return np.mean(df.y_true==df.y_pred)

class POMP(Metric):
    """
    Metric for Percentage of Maximum Profit, i.e. the proportion of the
    theoretical maximum profit that results from obeying the predictions
    of the model.
    """
    def eval(self, df: pd.dataframe) -> float:
        return df.profit.sum()/df.max_profit.sum()

class MAPE(Metric):
    """
    Mean absolute percentage error between predicted and true prices.
    """
    def eval(self, df:pd.DataFrame) -> float:
        return mean_absolute_percentage_error(df.true_price, df.predicted_price)
