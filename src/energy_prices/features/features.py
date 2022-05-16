from dataclasses import dataclass
from typing import List

import pandas as pd

from ..core import Transformer


@dataclass
class ColumnPicker(Transformer):
    """
    Picks the desired columns from the dataframe
    """
    columns: List[str] = [
        'demand',
        'price',
        'min_temperature',
        'max_temperature',
        'solar_exposure',
        'rainfall',
        'school_day',
        'holiday',
        'week',
    ]

    def fit(self, X: pd.DataFrame) -> None:
        """
        Does nothing since there's no need to fit anything.
        """
        pass

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Picks the desired columns from the dataframe.
        """
        X = X.copy()
        return X[self.columns]

@dataclass
class NPreviousMedian(Transformer):
    """
    Generates a new column that is the median of the n previous days
    """
    n_prev: int = 7

    def fit(self, X: pd.DataFrame) -> None:
        """
        Does nothing since there's no need to fit anything.
        """
        pass

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with previous' days median to dataframe
        """
        X = X.copy()

        cols = []
        for n in range(1,self.n_prev+1):
            cols.append(f'price_{n}')

        X['median_price'] = X[cols].median(axis = 1)

        return X

class HistoricalPrice(Transformer):
    """
    Generates a new column that contains the historical median price for 
    the week for each date.
    """
    def __init__(self) -> None:
        self.historical_prices = pd.DataFrame()

    def fit(self, X:pd.DataFrame) -> None:
        """
        Find the historical prices for each week using only data from previous years.
        """
        self.historical_prices = X.groupby('week').price.median().to_frame('historical_median_price')


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with historical median to dataframe.
        """
        X = X.copy()
        X = X.merge(self.historical_prices, on='week')

        return X

class HistoricalDemand(Transformer):
    """
    Generates a new column that contains the historical median price for 
    the week for each date.
    """
    def __init__(self) -> None:
        self.historical_prices = pd.DataFrame()

    def fit(self, X:pd.DataFrame) -> None:
        """
        Find the historical prices for each week using only data from previous years.
        """
        self.historical_prices = X.groupby('week').demand.median().to_frame('historical_median_demand')


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with historical median to dataframe.
        """
        X = X.copy()
        X = X.merge(self.historical_prices, on='week')

        return X


class HistoricalDemand(Transformer):
    """
    Generates a new column that contains the historical median demand for 
    the week for each date.
    """
    def __init__(self) -> None:
        self.historical_demand = pd.DataFrame()

    def fit(self, X:pd.DataFrame) -> None:
        """
        Find the historical demand for each week using only data from previous years.
        """
        df = pd.DataFrame()
        for year in X.year.unique().sort():
            df_medians = X[X.year < year].groupby('week').deman.median().to_frame('historical_median')
            df_medians['year'] = year
            df = pd.concat([df, df_medians])

        self.historical_demand = df


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with historical median demand to dataframe.
        """
        X = X.copy()
        X = X.merge(self.historical_demand, on=['year','week'])

        return X


class HistoricalPricePrev(Transformer):
    """
    Generates a new column that contains the historical median price for 
    the week for each date.
    """
    def __init__(self) -> None:
        self.historical_prices = pd.DataFrame()

    def fit(self, X:pd.DataFrame) -> None:
        """
        Find the historical prices for each week using only data from previous years.
        """
        df = pd.DataFrame()
        for year in X.year.unique().sort():
            df_medians = X[X.year < year].groupby('week').price.median().to_frame('historical_median')
            df_medians['year'] = year
            df = pd.concat([df, df_medians])

        self.historical_prices = df


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with historical median to dataframe.
        """
        X = X.copy()
        X = X.merge(self.historical_prices, on=['year','week'])

        return X


class HistoricalDemandPrev(Transformer):
    """
    Generates a new column that contains the historical median demand for 
    the week for each date.
    """
    def __init__(self) -> None:
        self.historical_demand = pd.DataFrame()

    def fit(self, X:pd.DataFrame) -> None:
        """
        Find the historical demand for each week using only data from previous years.
        """
        df = pd.DataFrame()
        for year in X.year.unique().sort():
            df_medians = X[X.year < year].groupby('week').deman.median().to_frame('historical_median')
            df_medians['year'] = year
            df = pd.concat([df, df_medians])

        self.historical_demand = df


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds column with historical median demand to dataframe.
        """
        X = X.copy()
        X = X.merge(self.historical_demand, on=['year','week'])

        return X


    


    