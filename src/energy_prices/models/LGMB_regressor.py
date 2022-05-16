import lightgbm
import pandas as pd

from ..core import Model


class LGBMRegressor(Model):
    """
    Uses an LGBM regressor to predict prices for tomorrow.
    """
    def __init__(self, **params) -> None:

        self.default_attrs = {}
        self.default_params = params

        # Initialize model
        self.model = lightgbm.LGBMRegressor(**self.default_params)

        # Define hyperparameters that optuna should optimize
        self.hyperparams = {
            "learning_rate": {"type": "float", "low": 1e-6, "high":1, "log":True},
            "n_estimators": {"type": "int", "low": 10, "high":1000},
            "max_depth":{"type": "int", "low": 1, "high":100},
            "reg_alpha": {"type": "float", "low": 1e-6, "high":1, "log":True},
            "reg_lambda": {"type": "float", "low": 1e-6, "high":1, "log":True},
        }

        # Define name of metric and the good direction
        self.metric_name = "POMP"
        self.metric_direction = 'maximize'

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit model to data in X and y.
        """
        self.model.fit(X, y)
        
    def predict(self, X) -> pd.DataFrame:
        """
        Makes predictions based on features in X.
        """
        y_pred = self.model.predict(X)
        preds = pd.DataFrame(data=y_pred)

        return preds

    def set_params(self, params: dict) -> None:
        """
        Set parameters for the model
        """
        for key, value in params.items():
            if key in vars(self.model).keys():
                setattr(self.model, key, value)

    def get_params(self) -> None:
        """
        Get parameters for the model
        """
        params = {}
        for key in vars(self.model).keys():
            params[key] = getattr(self.model, key)

        return params
