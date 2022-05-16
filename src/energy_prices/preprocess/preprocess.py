from dataclasses import dataclass
from typing import List

import pandas as pd

from ..core import Transformer


@dataclass
class PreviousDays(Transformer):
    """
    Generates new columns that contain the prices for previous days.
    """
    n_prev: int = 7

    def fit(self, X: pd.DataFrame) -> None:
        """
        Does nothing since there's no need to fit anything.
        """
        pass

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Adds colums with previous days' prices to dataframe.
        """
        X = X.copy()

        for n in range(1,self.n_prev+1):
            X[f'price_{n}'] = X.price.shift(n,'D')

        # Remove dates where previous prices are not available
        X.dropna(inplace = True)

        return X


