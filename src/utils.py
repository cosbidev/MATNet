# %%
from __future__ import annotations

import inspect
import itertools
import json
import os
from typing import Type, Union

import pytorch_lightning as pl
import torch
import torch.utils.data as data

from src import miner, model, metrics

experimental_setups = {"model_name": ["MATNet"],
                       "temporal_ablation": {"hour_on": [True],
                                             "day_on": [True],
                                             "month_on": [True]},

                       "branch_ablation": {"wx_history": [True, False],
                                           "wx_forecast": [True, False],
                                           "pv_forecast": [True, False],
                                           },

                       "sliding_window": {"win_length": [24],
                                          "step": [24],
                                          "time_horizon": [24]},

                       "MATNet_architecture_setup": {"num_layers": [3],
                                                     "fusion": ["Concat"],  # "fusion": ["Concat", "Sum"],
                                                     "interpolation": ["Adaptive"],
                                                     "interpolation_factor": [0],
                                                     }
                       }


def get_model(model_name: str, **kwargs) -> Union[Type[model.MPVNet], Type[model.MATNet]]:
    """ Get a model instance with the specified name and parameters.

    Parameters
    ----------
    model_name : str
        Name of the model to instantiate. Must be a key in `model_architectures`.
    **kwargs
        Keyword arguments for the model initialization. These must match the parameters
        of the model's `__init__` method, except for the `self` parameter.

    Returns
    -------
    model_class
        An instance of the specified model class, with the provided parameters.

    Raises
    ------
    ValueError
        If `model_name` is not a valid model name or if `kwargs` contains invalid parameters
        for the specified model.
    """
    # Dictionary mapping model names to model classes
    model_architectures = {
        'MATNet': model.MATNet,  # Multi-Level Fusion and Self-Attention Transformer-Based Model
    }

    # Validate the specified model name
    if model_name not in model_architectures:
        raise ValueError(f'Invalid model name: {model_name}')

    # Get the model class with the specified name
    model_class = model_architectures[model_name]

    # Get the signature of the model's __init__ method
    signature = inspect.signature(model_class.__init__)

    # Get the expected parameters for the model's __init__ method, excluding 'self'
    expected_params = {
        param_name: param.annotation
        for param_name, param in signature.parameters.items()
        if param_name != 'self'
    }

    # Filter the provided kwargs to only include valid parameters
    provided_params = {
        param_name: param_value
        for param_name, param_value in kwargs.items()
        if param_name in expected_params
    }

    # Check for any invalid parameters in the provided kwargs
    invalid_params = set(kwargs.keys()) - set(expected_params.keys())
    if invalid_params:
        raise ValueError(f'Invalid parameters for {model_name}: {invalid_params}')

    return model_class(**provided_params)


def create_filename(experiment_setup) -> str:
    """ Creates a filename based on the values in a dictionary.

    Parameters
    ----------
    experiment_setup : dict
        The input dictionary. The values can be of any type, but dictionaries and booleans are treated differently:
        - If the value is a boolean, and it is True, the corresponding key is included in the filename.
        - If the value is a dictionary, the function is called recursively on the dictionary to process its values.
        - Otherwise, the value is converted to a string and included in the filename.

    Returns
    -------
    filename : str
        A filename created from the values in `experiment_setup`.

    Examples
    --------
    >>> exp_setup = {"model_name": "foo", "flag": True, "param": 42, "subdict": {"a": 1, "b": 2}}
    >>> create_filename(exp_setup)
    'foo_flag_42_1_2'
    """
    parts = []
    for key, value in experiment_setup.items():
        if isinstance(value, bool):
            if value:
                # If the value is a boolean, and it's True, include the key in the filename
                parts.append(key)
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively process it
            parts.append(create_filename(value))
        else:
            # Otherwise, just include the value in the filename
            parts.append(str(value))
    return "_".join(part for part in parts if part)


def get_combinations(params: dict[str, list[any]]) -> dict[str, any]:
    """
    Generates all combinations of the values in a dictionary.

    Parameters
    ----------
    params : dict
        The input dictionary. The values must be lists of elements.

    Yields
    ------
    combination : dict
        A dictionary containing one combination of the values in `params`.
        The keys of the dictionary are the keys of `params`.

    Examples
    --------
    >>> params = {'x': [1, 2], 'y': ['a', 'b']}
    >>> for combination in get_combinations(params):
    ...     print(combination)
    {'x': 1, 'y': 'a'}
    {'x': 1, 'y': 'b'}
    {'x': 2, 'y': 'a'}
    {'x': 2, 'y': 'b'}
    """

    # Get the keys and values from the dictionary
    keys = params.keys()
    values = params.values()

    # Use itertools.product to generate all combinations of the values
    for combination in itertools.product(*values):
        # Zip the keys and values together to create a dictionary for the current combination
        yield dict(zip(keys, combination))


def count_true(d: dict[any, bool]) -> int:
    """ Counts the number of True values in a dictionary.

    Parameters
    ----------
    d : dict
        The input dictionary. The values must be of type bool.

    Returns
    -------
    count : int
        The number of True values in `d`.
    """
    return sum(value for value in d.values())


def generate_experimental_setups(name, transformer_model=True):
    experimental_setups["temporal_ablation"] = [item for item in
                                                get_combinations(experimental_setups["temporal_ablation"])]
    experimental_setups["branch_ablation"] = [item for item in
                                              get_combinations(experimental_setups["branch_ablation"])]
    experimental_setups["sliding_window"] = [item for item in
                                             get_combinations(experimental_setups["sliding_window"])]

    if transformer_model:
        experimental_setups["MATNet_architecture_setup"] = [item for item in get_combinations(
            experimental_setups["MATNet_architecture_setup"])]

    experiment_list = [exp for exp in get_combinations(experimental_setups)]

    with open(f'experiment_setups_{name}.json', 'w') as fout:
        json.dump(experiment_list, fout)


def find_first_digit(s: str) -> int:
    """
    Finds the index of the first digit in a string.

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    index : int
        The index of the first digit in `s`, or -1 if no digits are found.
    """
    try:
        return next(i for i, c in enumerate(s) if c.isdigit())
    except StopIteration:
        return -1


def filename2setup(exp_filename: str) -> dict:
    """
    Extracts the experimental setup from a model filename.

    Parameters
    ----------
    exp_filename : str
        The filename of the model.

    Returns
    -------
    exp_setup : dict
        A dictionary containing the experimental setup for the model. The dictionary has the following keys:
        - "model_name": str
            The name of the model.
        - "temporal_ablation": dict
            A dictionary indicating whether certain temporal ablation conditions are enabled. The dictionary has the
            following keys:
                - "hour_on": bool
                    Indicates whether the hour-level ablation is enabled.
                - "day_on": bool
                    Indicates whether the day-level ablation is enabled.
                - "month_on": bool
                    Indicates whether the month-level ablation is enabled.
        - "branch_ablation": dict
            A dictionary indicating whether certain branch ablation conditions are enabled. The dictionary has the
            following keys:
                - "wx_history": bool
                    Indicates whether the weather history branch is enabled.
                - "wx_forecast": bool
                    Indicates whether the weather forecast branch is enabled.
                - "pv_forecast": bool
                    Indicates whether the photovoltaic forecast branch is enabled.
        - "sliding_window": dict
            A dictionary containing the sliding window configuration. The dictionary has the following keys:
                - "win_length": int
                    The length of the sliding window.
                - "step": int
                    The step size of the sliding window.
                - "time_horizon": int
                    The time horizon of the sliding window.
    """
    sliding_window_values = exp_filename[find_first_digit(exp_filename):].split("_")
    exp_setup = {
        "model_name": "_".join(exp_filename.split("_", 2)[:2]),
        "temporal_ablation": {
            "hour_on": True if "hour_on" in exp_filename else False,
            "day_on": True if "day_on" in exp_filename else False,
            "month_on": True if "month_on" in exp_filename else False
        },
        "branch_ablation": {
            "wx_history": True if "wx_history" in exp_filename else False,
            "wx_forecast": True if "wx_forecast" in exp_filename else False,
            "pv_forecast": True if "pv_forecast" in exp_filename else False,
        },
        "sliding_window": {
            "win_length": int(sliding_window_values[0]),
            "step": int(sliding_window_values[1]),
            "time_horizon": int(sliding_window_values[2])
        }
    }
    return exp_setup


def flattenize_dict(d):
    """ Flattenize a dictionary by using the keys of any nested dictionaries as new keys.

    Parameters
    ----------
    d: dict
        The dictionary to flattenize.

    Yields
    -------
    tuple: A tuple containing a flattened key and its corresponding value.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            yield from flattenize_dict(value)
        else:
            yield key, value


def get_training_duration(start_timestamp, end_timestamp):
    """ Calculates the duration of a deep learning training in minutes, given the start and end Unix timestamps.

    Parameters
    ----------
    start_timestamp: int
        The Unix timestamp representing the start of the training.
    end_timestamp: int
        The Unix timestamp representing the end of the training.

    Returns
    -------
    minutes: float
        The duration of the training in minutes.
    """
    # Calculate the difference between the two timestamps in minutes.
    minutes = (end_timestamp - start_timestamp) / 60

    return minutes


def get_performances(weather_data, filename_path, experiment_setup, filename, best):
    """ This function calculates various performance metrics on the predictions made by a PyTorch model on a test set.

    Parameters
    ----------
    weather_data: str
        The file path of the weather dataset
    filename_path: str
        The file path of the model checkpoint.
    experiment_setup: dict
        A dictionary containing the experiment setup parameters.

    Returns
    -------
    results: dict
        A dictionary containing the following performance metrics:
        MAE: Mean Absolute Error
        MAPE: Mean Absolute Percentage Error
        MSE: Mean Squared Error
        RMSE: Root Mean Squared Error
        SMAPE: Symmetric Mean Absolute Percentage Error
        WMAPE: Weighted Mean Absolute Percentage Error
    """

    swx_on = experiment_setup['branch_ablation']["wx_history"]
    fwx_on = experiment_setup['branch_ablation']["wx_forecast"]

    hour_on = experiment_setup['temporal_ablation']["hour_on"]
    day_on = experiment_setup['temporal_ablation']["day_on"]
    month_on = experiment_setup['temporal_ablation']["month_on"]

    win_length = experiment_setup['sliding_window']["win_length"]
    step = experiment_setup['sliding_window']["step"]
    time_horizon = experiment_setup['sliding_window']["time_horizon"]

    # Create a test set using the experiment parameters
    # dataset = miner.MVAusgrid(root="Data", train=True, plants=None, max_kwp=True, win_length=win_length, step=step,
    #                          time_horizon=time_horizon, normalize='min-max', scaler=None, eps=1e-5, swx_on=swx_on,
    #                          fwx_on=fwx_on, hour_on=hour_on, day_on=day_on, month_on=month_on, plant=None)

    test_set = miner.MVAusgrid(root="Data", weather_data=weather_data, train=False, plants=None, max_kwp=True,
                               win_length=win_length, step=step, time_horizon=time_horizon, normalize="min-max",
                               scaler=None, eps=1e-5, swx_on=swx_on, fwx_on=fwx_on, hour_on=hour_on, day_on=day_on,
                               month_on=month_on, plant=None, perturbation=None)

    # Create a PyTorch DataLoader for the test set
    test_loader = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, drop_last=False, num_workers=0)

    # Load the model from the checkpoint file
    pretrained_filename = os.path.join(filename_path, "checkpoints", filename)
    net = model.MPVForecaster.load_from_checkpoint(pretrained_filename)

    # Make predictions on the test set using the loaded model
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, )
    pred = trainer.predict(net, dataloaders=test_loader)
    y_hat, y_true = pred[0]

    MAE = metrics.mean_absolute_error(y_hat, y_true, reduction='mean')
    MAPE = metrics.mean_absolute_percentage_error(y_hat, y_true, reduction='mean')
    MSE = metrics.mean_squared_error(y_hat, y_true, reduction='mean', squared=True)
    RMSE = metrics.mean_squared_error(y_hat, y_true, reduction='mean', squared=False)
    SMAPE = metrics.symmetric_mean_absolute_percentage_error(y_hat, y_true, reduction='mean')
    WMAPE = metrics.weighted_mean_absolute_percentage_error(y_hat, y_true, reduction='mean')
    MAAPE = metrics.mean_arc_absolute_percentage_error(y_hat, y_true, reduction='mean')
    MDA = metrics.mean_directional_accuracy(y_hat, y_true, reduction='mean')
    MASE = metrics.mean_absolute_scaled_error(y_hat, y_true, reduction='mean')

    if best:
        best = "best_val_"
    else:
        best = ""
    # Calculate and return the performance metrics
    results = {f"{best}mean_MAE": round(float(MAE[0]), 4), f"{best}std_MAE": round(float(MAE[1]), 4),
               f"{best}mean_MAPE": round(float(MAPE[0]), 4), f"{best}std_MAPE": round(float(MAPE[1]), 4),
               f"{best}mean_MSE": round(float(MSE[0]), 4), f"{best}std_MSE": round(float(MSE[1]), 4),
               f"{best}mean_RMSE": round(float(RMSE[0]), 4), f"{best}std_RMSE": round(float(RMSE[1]), 4),
               f"{best}mean_SMAPE": round(float(SMAPE[0]), 4), f"{best}std_SMAPE": round(float(SMAPE[1]), 4),
               f"{best}mean_WMAPE": round(float(WMAPE[0]), 4), f"{best}std_WMAPE": round(float(WMAPE[1]), 4),
               f"{best}mean_MAAPE": round(float(MAAPE[0]), 4), f"{best}std_MAAPE": round(float(MAAPE[1]), 4),
               f"{best}mean_MDA": round(float(MDA[0]), 4), f"{best}std_MDA": round(float(MDA[1]), 4),
               f"{best}mean_MASE": round(float(MASE[0]), 4), f"{best}std_MASE": round(float(MASE[1]), 4),
               }

    return results
