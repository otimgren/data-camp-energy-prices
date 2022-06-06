import copy

import numpy as np
import pandas as pd

import optuna

from ..core import Pipeline
from ..validation import CrossValidator


class OptunaHPOptimizer:
    """
    Object for using optuna for optimizing parameters for a model.
    """
    def __init__(self, pipeline: Pipeline, cv: CrossValidator) -> None:
        """
        Initialize the optimizer.

        inputs:
        pipeline: A pipeline object that contains the model whose 
                  hyperparameters are to be optimized. The model inside the
                  pipeline should contain the parameters to be optimized.
        """
        self.pipeline = pipeline
        self.cv = cv
        self.study = None
        self.objective = None

        self.create_study()
        self.create_objective()

    def create_study(self) -> None:
        """
        Creates an optuna study for the pipeline named after model.
        """
        study_name = self.pipeline.model.__class__.__name__
        storage_name = "sqlite:///tuning/{}.db".format(study_name)
        direction = self.pipeline.model.metric_direction
        self.study = optuna.create_study(study_name=study_name, storage=storage_name, 
                                        load_if_exists=True, direction=direction)
    
    def create_objective(self)-> None:
        """
        Creates an objective function for the optimizer.
        """
        def objective(trial, X:pd.DataFrame)-> float:

            # Generate parameters for the model
            model_params = {}
            params = copy.deepcopy(self.pipeline.model.hyperparams)
            for param, values in params.items():
                suggestion_generator = getattr(trial, f'suggest_{values.pop("type")}')
                model_params[param] = suggestion_generator(name = param, **values)

            # Set model parameters
            self.pipeline.model.set_params(model_params)

            # Cross validate the model
            X = self.cv.generate_folds(X)
            cv_result = self.cv.cross_validate(self.pipeline, X)

            # Find the value of the relevant metric
            metric = self.pipeline.model.metric_name
            return np.mean(cv_result[metric])

        self.objective = objective

    def optimize(self, X:pd.DataFrame, n_trials: int = 100)->None:
        """
        Run the optimization study.
        """
        self.study.optimize(lambda trial:self.objective(trial, X), n_trials=n_trials)

    def get_best_params(self)-> dict:
        """
        Gets the parameters for the best trial.
        """
        return self.study.best_params


