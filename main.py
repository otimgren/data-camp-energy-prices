import argparse
import logging

import pandas as pd

from energy_prices import Pipeline, dispatch_model
from energy_prices.data import load_raw
from energy_prices.features import (ColumnPicker, HistoricalDemand,
                                    HistoricalPrice, NPreviousMedian)
from energy_prices.preprocess import PreviousDays
from energy_prices.tuning import OptunaHPOptimizer
from energy_prices.validation import CrossValidator
from energy_prices.validation.metrics import MAPE, POMP, CorrectBuyPercentage


def main(model_name: str, predict: bool, cv: bool, tune: bool):
    """
    Function used for training and testing models.
    """
    # Start logging
    logging.basicConfig(filename='./logging/main.log', level=logging.INFO, 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    # Load data
    train = load_raw()

    # Define preprocessors
    preprocessors = [PreviousDays()]

    # Define data transforms
    transformers = [ColumnPicker(), NPreviousMedian(), HistoricalPrice(), HistoricalDemand()]

    # Get model using model_dispatcher
    model = dispatch_model(model_name)

    # Define pipeline
    pipeline = Pipeline(preprocessors, transformers, model)

    # Cross validate pipeline
    if cv:
        cross_validate(pipeline, train)

    # Tune hyperparameters
    if tune:
        tune_hyperparams(pipeline, train)

def cross_validate(pipeline: Pipeline, train: pd.DataFrame) -> None:
    """
    Cross validates the pipeline using data in train.
    """
    # Define cross validator
    metrics = [MAPE(), POMP(), CorrectBuyPercentage()]
    cross_validator = CrossValidator(metrics)
    
    # Perform cross validation
    train = cross_validator.generate_folds(train)
    result = cross_validator.cross_validate(pipeline, train)
    cross_validator.print_cv_results(result)

    # Save pipeline
    pipeline.save()

    # Save result
    pipeline.save_cv(result)

def tune_hyperparams(pipeline: Pipeline, train: pd.DataFrame) -> None:
    """
    Tunes the hyperparameters for the model using cross validation.
    """
    # Define cross validator
    metrics = [MAPE(), CorrectBuyPercentage(), POMP()]
    cross_validator = CrossValidator(metrics)

    # Initialize the hyperparameter optimizer
    optimizer = OptunaHPOptimizer(pipeline, cross_validator)

    # Run the optimizer
    optimizer.optimize(train, n_trials=100)

    # Get the best parameters
    best_params = optimizer.get_best_params()

    # Save best parameters and pipeline
    pipeline.model.set_params(best_params)

    # Save pipeline
    pipeline.save(base_directory='optimized_models')


if __name__ == '__main__':
    # Define parser and add arguments
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model", type=str, default='LGBMRegressor')
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--cv", action='store_true')
    parser.add_argument("--tune", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # Run the model training program
    main(args.model, args.predict, args.cv, args.tune)
