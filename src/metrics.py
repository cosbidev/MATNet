# %%
from typing import Union, Tuple

import torch


def max_error(predictions: torch.Tensor, targets: torch.Tensor,
              reduction: str = 'mean') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the maximum error between predicted and target values for batched samples.

    Parameters:
        predictions (torch.Tensor): predicted values with shape (batch_size, num_observations)
        targets (torch.Tensor): target values with shape (batch_size, num_observations)
        reduction (str): Specifies the reduction to apply to the output.
            'none': no reduction will be applied, the output will have the shape (batch_size,)
            'mean': the output will be the mean and standard deviation of the maximum errors across the batch
            'sum': the output will be the sum of the maximum errors across the batch

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]: maximum error
    """
    assert predictions.shape == targets.shape, "predictions and targets must have the same shape"

    # Compute max error
    max_errors, _ = torch.max(torch.where(predictions > targets, predictions - targets, targets - predictions), dim=1)

    if reduction == 'none':
        return max_errors
    elif reduction == 'mean':
        mean = torch.mean(max_errors)
        std = torch.std(max_errors)
        return mean, std
    elif reduction == 'sum':
        return torch.sum(max_errors)
    else:
        raise ValueError("Invalid reduction value. Choose one of 'none', 'mean', 'sum'.")


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean',
                       squared: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes mean squared error between predicted and target values.

    Parameters:
        predictions: torch.Tensor, predicted values
        targets: torch.Tensor, target values
        reduction: str, type of reduction to apply to the output.
                     'none' | 'mean' | 'sum'
        squared: bool,  if True, it will return the MSE, if false will return the RMSE

    Returns:
        torch.Tensor, MSE or RMSE loss.
    """

    assert predictions.shape == targets.shape, "predictions and targets must have the same shape"

    pow_errors = torch.pow(predictions - targets, 2)

    mean_loss = torch.mean(pow_errors, dim=1)
    if not squared:
        mean_loss = torch.sqrt(mean_loss)

    if reduction == 'none':
        return mean_loss
    elif reduction == 'mean':
        mean = mean_loss.mean()
        std = mean_loss.std()
        return mean, std
    elif reduction == 'sum':
        return mean_loss.sum()
    else:
        raise ValueError("Invalid reduction type. Choose 'none', 'mean' or 'sum'")


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor,
                        reduction: str = 'mean') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the Mean Absolute Error (MAE) between predicted and target values.
    Supports both single samples and batched samples.
    Allows for different types of reduction to be applied to the output.
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"

    # Compute absolute difference between predictions and targets
    absolute_difference = torch.abs(predictions - targets)
    mean_absolute_difference = torch.mean(absolute_difference, dim=1)

    if reduction == 'none':
        return mean_absolute_difference
    elif reduction == 'mean':
        return mean_absolute_difference.mean(), mean_absolute_difference.std()
    elif reduction == 'sum':
        return torch.sum(mean_absolute_difference)
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Must be 'none', 'mean', or 'sum'.")


def mean_absolute_percentage_error(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean',
                                   epsilon: float = 1.17e-06) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between the predicted and target values.
    Can handle batched samples and apply different reductions to the output.

    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The target values.
        reduction (str): The reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        epsilon (float): A small value to add to the denominator to avoid division by zero.

    Returns:
        torch.Tensor: The MAPE, or a tuple containing the MAPE and standard deviation if reduction='mean'.
    """
    assert predictions.shape == targets.shape, "Predicted and target shapes must match"

    abs_per_error = torch.abs(targets - predictions) / torch.clamp(torch.abs(targets), min=epsilon)
    mean_abs_per_error = torch.mean(abs_per_error, dim=1)

    if reduction == 'none':
        return mean_abs_per_error
    elif reduction == 'mean':
        return mean_abs_per_error.mean(), mean_abs_per_error.std()
    elif reduction == 'sum':
        return mean_abs_per_error.sum()
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Must be 'none', 'mean', or 'sum'.")


def symmetric_mean_absolute_percentage_error(predictions: torch.Tensor,
                                             targets: torch.Tensor,
                                             reduction: str = 'mean',
                                             epsilon: float = 1.17e-06,
                                             return_std: bool = False) -> Union[torch.Tensor,
                                                                                Tuple[torch.Tensor,
                                                                                      torch.Tensor]]:
    """
    Computes the symmetric mean absolute percentage error (SMAPE) between predicted and target values for a batch
    of samples, with the option to apply a reduction to the output.

    Parameters:
        predictions (torch.Tensor): The predicted values with shape (batch_size, ...)
        targets (torch.Tensor): The target values with shape (batch_size, ...)
        reduction (str, optional): The reduction to apply to the output.
                                  Can be one of 'none', 'mean', 'sum' (default: 'mean')
        epsilon (float): A small value to add to the denominator to avoid division by zero.
        return_std (bool, optional): If True, also return the standard deviation of the SMAPE (default: False)

    Returns:
        torch.Tensor: The SMAPE values with shape (batch_size, ...) or a scalar depending on the reduction
        torch.Tensor: The standard deviation of the SMAPE values (if return_std=True)
    """

    assert predictions.shape == targets.shape, "Predicted and target shapes must match"

    # Compute absolute percentage error
    abs_diff = torch.abs(predictions - targets)
    abs_per_error = abs_diff / torch.clamp(torch.abs(targets) + torch.abs(predictions), min=epsilon)

    # Compute absolute percentage error
    mean_ape = 2 * torch.mean(abs_per_error, dim=1)

    # Apply reduction
    if reduction == 'none':
        return mean_ape
    elif reduction == 'mean':
        return mean_ape.mean(), mean_ape.std()
    elif reduction == 'sum':
        return mean_ape.sum()
    else:
        raise ValueError("Invalid reduction option: '{}'".format(reduction))


def weighted_mean_absolute_percentage_error(predictions: torch.Tensor,
                                            targets: torch.Tensor,
                                            epsilon: float = 1.17e-06,
                                            reduction: str = 'mean') -> Union[torch.Tensor,
                                                                              Tuple[torch.Tensor,
                                                                                    torch.Tensor]]:
    """
    Computes the weighted mean absolute percentage error (WMAPE) between predicted and target values.
    Supports handling of batched samples and different types of output reductions.

    Args:
        predictions: torch.Tensor of shape (batch_size, n_predictions)
            The predicted values.
        targets: torch.Tensor of shape (batch_size, n_predictions)
            The target values.
        epsilon (float):
            A small value to add to the denominator to avoid division by zero.
        reduction: str
            Specifies the type of reduction to apply to the output.
            'none': no reduction will be applied.
            'mean': the output will be the mean of the WMAPE across all predictions.
            'sum': the output will be the sum of the WMAPE across all predictions.

    Returns:
        output: torch.Tensor or tuple of torch.Tensor
            The WMAPE or mean and standard deviation of WMAPE
    """

    assert predictions.shape == targets.shape, "Predicted and target shapes must match"

    sum_abs_error = torch.sum((predictions - targets).abs(), dim=1)
    sum_scale = torch.sum(targets.abs(), dim=1)

    weighted_mean_ape = sum_abs_error / torch.clamp(sum_scale, min=epsilon)

    if reduction == 'none':
        return weighted_mean_ape
    elif reduction == 'mean':
        return weighted_mean_ape.mean(), weighted_mean_ape.std()
    elif reduction == 'sum':
        return weighted_mean_ape.sum()
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def mean_absolute_scaled_error(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               reduction: str = 'mean') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the Mean Absolute Scaled Error (MASE) between predicted and target values.
    The MASE is computed as:
    MASE = mean(abs(pred - target)) / mean(abs(target[1:] - target[:-1]))

    Parameters:
        predictions: torch.Tensor of shape (batch_size, n_predictions)
            The predicted values.
        targets: torch.Tensor of shape (batch_size, n_predictions)
            The target values.
        reduction: str
            Specifies the type of reduction to apply to the output.
            'none': no reduction will be applied.
            'mean': the output will be the mean of the MASE across all predictions.
            'sum': the output will be the sum of the MASE across all predictions.

    Returns:
        output: torch.Tensor or tuple of torch.Tensor
            The MASE or mean and standard deviation of MASE
    """

    # Compute mean absolute percentage error
    mean_abs_error = torch.mean(torch.abs(predictions - targets), dim=1)

    # Compute mean absolute target shift difference
    mean_abs_target_shift = torch.mean(torch.abs(targets[:, 1:] - targets[:, :-1]), dim=1)

    # Compute mean absolute scaled error
    mean_ase = mean_abs_error / mean_abs_target_shift

    if reduction == 'none':
        return mean_ase
    elif reduction == 'mean':
        return mean_ase.mean(), mean_ase.std()
    elif reduction == 'sum':
        return mean_ase.sum()
    else:
        raise ValueError("Invalid reduction argument. Must be 'none', 'mean', or 'sum'.")


def mean_directional_accuracy(predictions: torch.Tensor,
                              targets: torch.Tensor,
                              reduction: str = 'mean') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """ Compute Mean directional accuracy between actual and predicted values.
    Mean directional accuracy is a measure of similarity between two direction vectors or two unit vectors.

    Parameters
    ----------
    predictions : torch.Tensor of shape (batch_size, n_observations)
        The predicted values.
    targets : torch.Tensor of shape (batch_size, n_observations)
       The target values.
    reduction: str
        Specifies the type of reduction to apply to the output.
        'none': no reduction will be applied.
        'mean': the output will be the mean of the MASE across all predictions.
        'sum': the output will be the sum of the MASE across all predictions.

    Returns
    -------
    output: torch.Tensor or tuple of torch.Tensor
            The MDA or mean and standard deviation of MDA
    """
    assert predictions.shape == targets.shape, "Predicted and target shapes must match"

    # calculate the signs of the element-wise differences
    target_sign_error = torch.sign(targets[:, 1:] - targets[:, :-1])
    pred_sign_error = torch.sign(predictions[:, 1:] - predictions[:, :-1])

    # compare the signs
    comparison = (target_sign_error == pred_sign_error).float()

    # calculate the mean
    mean_sign_error = torch.mean(comparison, dim=1)

    if reduction == 'none':
        return mean_sign_error
    elif reduction == 'mean':
        return mean_sign_error.mean(), mean_sign_error.std()
    elif reduction == 'sum':
        return mean_sign_error.sum()
    else:
        raise ValueError("Invalid reduction argument. Must be 'none', 'mean', or 'sum'.")


def mean_arc_absolute_percentage_error(predictions: torch.Tensor,
                                       targets: torch.Tensor,
                                       reduction: str = 'mean',
                                       epsilon: float = 1.17e-06) -> Union[torch.Tensor,
                                                                           Tuple[torch.Tensor,
                                                                                 torch.Tensor]]:
    """ Compute the Mean Arctangent Absolute Percentage Error (MAPE) between predicted and target values.
    Can also handle batched samples.

    Parameters:
        predictions (torch.Tensor): Tensor containing the predicted values.
        targets (torch.Tensor): Tensor containing the target values.
        reduction (str, optional): Type of reduction to apply to the output.
                                  Can be 'none', 'mean' or 'sum' (default: 'mean').
        epsilon (float, optional): Small value to avoid division by zero. (default: 1.17e-06)

    Returns:
        If reduction is 'none', returns a tensor with the MAPE for each sample.
        If reduction is 'mean', returns a tuple with the mean MAPE and standard deviation.
        If reduction is 'sum', returns the sum of the MAPE for all samples.
    """
    # Compute the absolute percentage error
    abs_error = torch.abs((targets - predictions) / torch.clamp(targets, min=epsilon))

    # Compute the arctangent of the absolute percentage error
    arc_abs_error = torch.atan(abs_error)
    # Compute the mean of the arctangent absolute percentage error
    mean_arc_abs_error = torch.mean(arc_abs_error, dim=1)

    if reduction == 'none':
        return mean_arc_abs_error
    elif reduction == 'mean':
        return mean_arc_abs_error.mean(), mean_arc_abs_error.std()
    elif reduction == 'sum':
        return mean_arc_abs_error.sum()
    else:
        raise ValueError("Invalid reduction option. Choose 'none', 'mean', or 'sum'.")
