# %%
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

nt_wx_variables = ['dt_iso']


def fusion(wx_data, pv_data, plant=None):
    """ The method fuses weather and PV production datasets together.

    Parameters
    ----------
    wx_data: pd.DataFrame
        Weather dataset
    pv_data: pd.DataFrame
        PV production dataset
    plant: list, optional, default: None
        List of plants to be considered

    Returns
    -------
    dataset: pd.DataFrame
        dataframe with weather data and photovoltaic production.
    """

    if plant is not None:
        wx_data['pv_aggregate'] = pv_data
    else:
        wx_data['pv_aggregate'] = pv_data#.sum(axis=1)

    return wx_data


def split_sliding_window(x, win_length=14, step=24, time_horizon=24):
    """ Split dataset in sliding windows and organize the dataset.

    Parameters
    ----------
    x: torch.Tensor
        Dataset to be elaborated
    win_length: int, optional, default: 336
        Length of the sliding window in hours
    step: int, optional, default: 24
        Step of the sliding window in hours
    time_horizon: int, optional, default: 24
        Length of the time series to be predicted

    Returns
    -------
    wx_history: torch.Tensor
        Historical weather samples
    pv_production: torch.Tensor
        PV production samples
    wx_forecast: torch.Tensor
        Weather forecasting samples
    y: torch.Tensor
        Time series to predict
    """

    x_elab = x.unfold(dimension=0, size=win_length, step=step).permute(0, 2, 1)
    pv_production = x_elab[:, :, -1].unsqueeze(2)[:-1]
    wx_history = x_elab[:-1, :, :-1]

    x_elab = x[step:].unfold(dimension=0, size=time_horizon, step=step).permute(0, 2, 1)
    y = x_elab[:, :, -1]
    wx_forecast = x_elab[:, :, :-1]

    return wx_history, pv_production, wx_forecast, y


def add_temporal_feature(data, hour_on, day_on, month_on):
    """ The method adds the time, day and month information to each row of the dataset.

    Parameters
    ----------
    data: pd.DataFrame
        A multivariate dataset with temporal indexing
    hour_on: bool, optional, default: True
        If to add hour variable
    day_on: bool, optional, default: True
        If to add day variable
    month_on: bool, optional, default: True
        If to add month variable

    Returns
    -------
    data: pd.DataFrame
        Elaborated dataset
    """
    for encoding_type in ["sin", "cos"]:
        if hour_on:
            data.insert(loc=data.shape[1] - 1, column=f"hour_{encoding_type}",
                        value=cyclical_encoding(data.index.hour, encoding_type))
        if day_on:
            data.insert(loc=data.shape[1] - 1, column=f"day_{encoding_type}",
                        value=cyclical_encoding(data.index.day, encoding_type))
        if month_on:
            data.insert(loc=data.shape[1] - 1, column=f"month_{encoding_type}",
                        value=cyclical_encoding(data.index.month, encoding_type))

    # if hour_on:
    #     data.insert(loc=data.shape[1] - 1, column="hour", value=data.index.hour)
    #
    # if day_on:
    #     data.insert(loc=data.shape[1] - 1, column="day", value=data.index.day)
    #
    # if month_on:
    #     data.insert(loc=data.shape[1] - 1, column="month", value=data.index.month)

    return data


def cyclical_encoding(data: np.ndarray, encoding_type: str) -> np.ndarray:
    """ Encodes data in a cyclical manner using sine or cosine functions.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to encode.
    encoding_type : str
        The type of encoding to use. Must be either "sin" or "cos".

    Returns
    -------
    encoded_data : numpy.ndarray
        The encoded data.

    Raises
    ------
    ValueError
        If `encoding_type` is not "sin" or "cos".

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> cyclical_encoding(x, "sin")
    array([ 0.0000000e+00,  1.0000000e+00,  1.2246468e-16, -1.0000000e+00,
            -2.4492936e-16])
    >>> cyclical_encoding(x, "cos")
    array([ 1.0000000e+00,  6.1232340e-17, -1.0000000e+00, -1.8369702e-16,
            1.0000000e+00])
    """
    # Validate the encoding type
    if encoding_type not in {"sin", "cos"}:
        raise ValueError(f"Invalid encoding type: {encoding_type}")

    # Encode the data using the selected function
    if encoding_type == "sin":
        return np.sin(2 * np.pi * data / max(data))
    else:
        return np.cos(2 * np.pi * data / max(data))


def _compute_noise(tensor, columns, percentage=0.1):
    """ Compute the noise to add to the original tensor.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to add noise to (shape: [batch_size, n_timesteps, n_features]).
    columns: list
        The indices of the columns to add noise to.
    percentage: float, optional, default: 0.1
        The percentage of the standard deviation of the original tensor to use as the standard deviation of the noise.

    Returns
    -------
    noise: torch.Tensor
        The noise tensor to add to the original tensor.
    """
    # Compute the standard deviation of the original tensor
    std = np.std(tensor[:, :, columns].numpy(), axis=(0, 1))
    noise = np.random.normal(0, std, size=tensor[:, :, columns].shape) * percentage

    return torch.Tensor(noise)


def add_noise_to_tensor(tensor, columns, noise):
    """ Add noise to the input tensor.
    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to add noise to (shape: [batch_size, n_timesteps, n_features]).
    columns: list
        The indices of the columns to add noise to.
    noise: torch.Tensor
        The noise tensor to add to the original tensor.

    Returns
    -------
    noisy_tensor: torch.Tensor
        The noisy tensor.
    """
    noisy_tensor = tensor.clone()

    noisy_tensor[:, :, columns] = tensor[:, :, columns] + noise
    return noisy_tensor


def _standardize(x, mean, var, eps):
    """ Standardize the input tensor.

    Parameters
    ----------
    x: torch.Tensor
        The input tensor to standardize.
    mean: torch.Tensor
        The mean of the distribution.
    var: torch.Tensor
        The variance of the distribution.
    eps: float
        A value added to the denominator for numerical stability.

    Returns
    -------
    standardized_tensor: torch.Tensor
        The standardized tensor.
    """
    return (x - mean) / torch.sqrt(var + eps)


def _normalize(x, max_value, min_value):
    x = x - min_value

    mask = (max_value - min_value) != 0
    x[:, mask.flatten()] = x[:, mask.flatten()] / (max_value - min_value)[mask]
    return x


def _compute_max_error(tensor, columns, error_percentage):
    max_values = torch.max(tensor[:, :, columns], dim=1).values
    max_error = max_values * error_percentage
    return max_error


class Scaler:
    def __init__(self, mode='min-max', eps=1e-5):
        self.mode = mode
        self.mean = None
        self.var = None
        self.min_value = None
        self.max_value = None

        self.eps = eps

    def fit(self, x):
        if self.mode == 'standardization':
            self.mean = torch.mean(x, dim=0)
            self.var = torch.var(x, dim=0, unbiased=False)

        elif self.mode == 'min-max':
            self.min_value = torch.amin(x, dim=0)
            self.max_value = torch.amax(x, dim=0)
        else:
            raise ValueError("Invalid mode. Supported modes: 'standardization', 'min-max'")

    def transform(self, x):
        if self.mode == 'standardization':
            return _standardize(x, self.mean, self.var, self.eps)
        elif self.mode == 'min-max':
            return _normalize(x, self.max_value, self.min_value)
        else:
            raise ValueError("Invalid mode. Supported modes: 'standardization', 'normalization'")


class Miner:
    """ Class Miner for data fetching.

    Parameters
    ----------
    root: str, optional, default: ./Data
        Main data root
    weather_data: str, optional, default: wx_data_owm.xlsx
        Weather data file
    train: bool, optional, default: True
        If considering the train folder
    plants: list, optional, default: None
        Which installations to consider
    max_kwp: bool, optional, default: True
        If to normalise for maximum production
    """

    def __init__(self, root="./Data", weather_data="wx_data_owm.xlsx",
                 train=True, plants=None, max_kwp=True):
        self.root = root
        self.pv_data = os.path.join(root, "dataset.xlsx")
        self.weather_data = os.path.join(root, weather_data)

        self.train = train
        self.max_kwp = max_kwp
        self.total_kwp = 0

        self.scaler = None

        if plants is not None:
            self.plants = plants
        else:
            self.plants = [33, 47, 73, 87, 88, 110, 124, 144, 151, 153,
                           157, 163, 175, 176, 188, 200, 201, 207, 222,
                           240, 256, 259, 263, 272, 281, 293]  # 230 has a strange behaviour

    def fetch_data(self, plant=None):
        """ Import phase of PV production and weather data.
        The datasets include photovoltaic production aggregate and weather data.

        Parameters
        ----------
        plant: int, optional, default: None
            Plant to be considered

        Returns
        -------
        dataset: pd.DataFrame
            Dataset with aggregated photovoltaic production and weather data.
        """

        if plant is not None:
            plants = plant
        else:
            plants = self.plants

        xls_production = pd.ExcelFile(self.pv_data)
        xls_weather = pd.ExcelFile(self.weather_data)

        info = pd.read_excel(xls_production, '07-10--06-11').loc[:0, plants]

        if self.max_kwp:
            self.total_kwp = info.sum(axis=1).item()

        if self.train:
            dates = ['07-10--06-11', '07-11--06-12']
        else:
            dates = ['07-11--06-12', '07-12--06-13']

        prod_dfs = [pd.read_excel(xls_production, date).loc[2:, plants] for date in dates]

        weather_dfs = [pd.read_excel(xls_weather, date) for date in dates]

        prod_df = pd.concat(prod_dfs)
        weather_df = pd.concat(weather_dfs)

        dataset = fusion(self.wx_processing(weather_df), self.pv_processing(prod_df, info, plant=plant), plant=plant)

        return dataset

    def pv_processing(self, data, info, plant):
        """ Method for processing the PV production dataset indexing the dataframe with the temporal range and
        scaling each value using the maximum generative capacity associated with it.

        Parameters
        ----------
        data: pd.DataFrame
            Dataset PV production
        info: pd.DataFrame
            Information on PV plants
        plant: int, optional, default: None
            Plant to be considered

        Returns
        -------
        data: pd.DataFrame
            PV production dataset processed
        """

        data = data.copy()
        if self.train:
            data.index = pd.date_range(start='2010-07-01 00:00:00', end='2012-06-30 23:00:00', freq='H')
        else:
            data.index = pd.date_range(start='2011-07-01 00:00:00', end='2013-06-30 23:00:00', freq='H')
            data = data.loc['2012-06-30 00:00:00':]

        if self.max_kwp:
            data = data.sum(axis=1)/self.total_kwp
        else:
            data = data.sum(axis=1)

        return data

    def wx_processing(self, data):
        """ Method for processing the weather dataset indexing the dataframe with the temporal range eliminating
        unnecessary columns variables and filling NaN values of 'wind_gust' and 'rain_1h' columns with zeros. Finally,
        convert categorical variable 'weather_description' into dummy variables.

        Parameters
        ----------
        data: pd.DataFrame
            Weather dataset

        Returns
        -------
        data: pd.DataFrame
            Weather dataset processed
        """

        data = data.copy()
        if self.train:
            data.index = pd.date_range(start='2010-07-01 00:00:00', end='2012-06-30 23:00:00', freq='H')
        else:
            data.index = pd.date_range(start='2011-07-01 00:00:00', end='2013-06-30 23:00:00', freq='H')
            data = data.loc['2012-06-30 00:00:00':]

        data = data.drop(columns=nt_wx_variables)

        return data

    def processing(self, win_length=336, step=24, time_horizon=24, normalize='min-max', scaler=None, eps=1e-5,
                   pv_on=True, swx_on=True, fwx_on=True, hour_on=True, day_on=True, month_on=True, perturbation=None,
                   plant=None):
        """ The method assembles the dataset and divides it into sliding windows.

        Parameters
        ----------
        win_length: int, optional, default: 336
            Length of the sliding window in hours
        step: int, optional, default: 24
            Step of the sliding window in hours
        time_horizon: int, optional, default: 24
            Length of the time series to be predicted
        normalize: [None, str], optional, default: 'min-max'
            If to normalize the dataset
        scaler: [None, Scaler], optional, default: None
            Mean of the distribution, this will be used to normalize the data
        eps: float, optional, default: 1e-5
            a value added to the denominator for numerical stability.
        pv_on: bool optional, default: True
            If to consider historical PV production data
        swx_on: bool, optional, default: True
            If to consider historical weather data
        fwx_on: bool, optional, default: True
            If to consider weather forecasting data
        hour_on: bool, optional, default: True
            If to add hour variable
        day_on: bool, optional, default: True
            If to add day variable
        month_on: bool, optional, default: True
            If to add month variable
        perturbation: [None, float], optional, default: None
            Perturbation level to be added to the weather forecast
        plant: int, optional, default: None
            Plant to be considered

        Returns
        -------
        pv_production: torch.Tensor
            PV production samples
        wx_history: torch.Tensor
            Historical weather samples
        wx_forecast: torch.Tensor
            Weather forecasting samples
        y: torch.Tensor
            Time series to predict

        """
        self.scaler = scaler

        data = self.fetch_data(plant)

        data = add_temporal_feature(data, hour_on, day_on, month_on)

        x = torch.tensor(data.values).float()
        if normalize:
            if self.scaler:
                x = self.scaler.transform(x)
            else:
                self.scaler = Scaler(mode=normalize, eps=eps)
                self.scaler.fit(x)
                x = self.scaler.transform(x)

        # Import and pre-processing of data
        wx_history, pv_production, wx_forecast, y = split_sliding_window(x, win_length=win_length, step=step,
                                                                         time_horizon=time_horizon)

        if perturbation is not None:
            if os.path.basename(self.weather_data) == "wx_data_owm_v2.xlsx":
                columns_to_add_noise = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 34]  # Column indices to apply noise

                noise = _compute_noise(wx_forecast, columns_to_add_noise, perturbation)
                # max_error = _compute_max_error(wx_forecast, columns_to_add_noise, perturbation)
                # wx_forecast = add_noise_to_tensor(wx_forecast, columns_to_add_noise, max_error)
                wx_forecast = add_noise_to_tensor(wx_forecast, columns_to_add_noise, noise)

            if os.path.basename(self.weather_data) == "wx_data_owm_v2_ITSG.xlsx":
                columns_to_add_noise = [0, 1, 2, 3, 4, 5, 6, 7, 30, 31, 32]  # Column indices to apply noise

                noise = _compute_noise(wx_forecast, columns_to_add_noise, perturbation)
                # max_error = _compute_max_error(wx_forecast, columns_to_add_noise, perturbation)
                # wx_forecast = add_noise_to_tensor(wx_forecast, columns_to_add_noise, max_error)
                wx_forecast = add_noise_to_tensor(wx_forecast, columns_to_add_noise, noise)

        if not pv_on:
            pv_production = torch.zeros(pv_production.shape)

        if not swx_on:
            wx_history = torch.zeros(wx_history.shape)

        if not fwx_on:
            wx_forecast = torch.zeros(wx_forecast.shape)

        return pv_production, wx_history, wx_forecast, y


class MVAusgrid(Dataset):
    """
    Parameters
    ----------
    root: str, optional, default: "./Data"
    train: bool, optional, default: True
    plants: list, optional, default: None
    max_kwp: bool, optional, default: True
    win_length: int, optional, default: 336
    step: int, optional, default: 24
    time_horizon: int, optional, default: 24
    normalize: [None, str], optional, default: 'min-max'
    scaler: [None, Scaler], optional, default: None
    pv_on: bool, optional, default: True
    swx_on: bool, optional, default: True
    fwx_on: bool, optional, default: True
    hour_on: bool, optional, default: True
    day_on: bool, optional, default: True
    month_on: bool, optional, default: True
    perturbation: [None, float], optional, default: None
    plant: int, optional, default: None
    """

    def __init__(self, root="./Data", weather_data="wx_data_owm.xlsx", train=True, plants=None, max_kwp=True,
                 win_length=336, step=24, time_horizon=24, normalize='min-max', scaler=None, eps=1e-5, pv_on=True,
                 swx_on=True, fwx_on=True, hour_on=True, day_on=True, month_on=True, perturbation=None, plant=None):
        self.miner = Miner(root=root, weather_data=weather_data, train=train, plants=plants, max_kwp=max_kwp)
        outs = self.miner.processing(win_length=win_length, step=step, time_horizon=time_horizon,
                                     normalize=normalize, scaler=scaler, eps=eps, pv_on=pv_on, swx_on=swx_on,
                                     fwx_on=fwx_on, hour_on=hour_on, day_on=day_on, month_on=month_on,
                                     perturbation=perturbation, plant=plant)

        self.scaler = self.miner.scaler

        self.pv_production, self.wx_history, self.wx_forecast, self.y = outs

    def __len__(self):
        return self.pv_production.shape[0]

    def __getitem__(self, idx):
        return (self.pv_production[idx], self.wx_history[idx],
                self.wx_forecast[idx]), self.y[idx]
