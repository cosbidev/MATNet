# %%
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib import utils, layers


class MPVForecaster(pl.LightningModule):
    # Wrapper of LightningModule class
    def __init__(self, model_name, train_loader, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = utils.get_model(model_name, **model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]
        self.lr = lr

    def forward(self, x1, x2, x3):
        pv_production, wx_history, wx_forecast = x1, x2, x3
        return self.model(pv_production, wx_history, wx_forecast)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def predict_step(self, batch, batch_idx):
        (pv_production, wx_history, wx_forecast), y_true = batch
        y_hat = self.model(pv_production, wx_history, wx_forecast)
        return y_hat, y_true

    def _calculate_loss(self, batch, mode="train"):
        (pv_production, wx_history, wx_forecast), y_true = batch
        y_pred = self.model(pv_production, wx_history, wx_forecast)
        loss = F.mse_loss(y_pred, y_true)

        self.log("%s_loss" % mode, loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class MPVNet(nn.Module):
    """ Multivariate Model for Photovoltaic Power Production Forecasting.
    Used as baseline in the MATNet-related journal paper. In contrast to MATNet, it is a recurrent-based architecture.

    Parameters
    ----------
    pv_features: int, optional, default: 1
        The number of features in the PV production data.
    hw_features: int, optional, default: 101
        The number of features in the historic weather data.
    fw_features: int, optional, default: 101
        The number of features in the forecasting weather data (Day Ahead).
    n_steps_in: int, optional, default: 336
        The number of time steps (hours) in the input sequence.
    n_steps_out: int, optional, default: 24
        The number of time steps (hours) in the output sequence.
    bidirectional: bool, optional, default:
        If True, the input data will be processed using a bidirectional LSTM or GRU.
    recurrent: str, optional, default: 'LSTM'
        The type of recurrent layer to use. Must be either 'LSTM' or 'GRU'.
    """

    def __init__(self, hw_features: int, fw_features: int, pv_features: int,
                 n_steps_in: int, n_steps_out: int, bidirectional: bool, recurrent: str):
        super().__init__()
        self.ap_features = pv_features
        self.hw_features = hw_features
        self.tw_features = fw_features
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

        self.d_model = 512
        self.dropout = 0.1

        if recurrent == "LSTM":
            self.agg_production = nn.LSTM(input_size=self.ap_features, hidden_size=self.d_model, num_layers=2,
                                          batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
            self.historic_weather = nn.LSTM(input_size=self.hw_features, hidden_size=self.d_model,
                                            num_layers=2,
                                            batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
            self.target_weather = nn.LSTM(input_size=self.tw_features, hidden_size=self.d_model, num_layers=2,
                                          batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        if recurrent == "GRU":
            self.agg_production = nn.GRU(input_size=self.ap_features, hidden_size=self.d_model, num_layers=2,
                                         batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
            self.historic_weather = nn.GRU(input_size=self.hw_features, hidden_size=self.d_model, num_layers=2,
                                           batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
            self.target_weather = nn.GRU(input_size=self.tw_features, hidden_size=self.d_model, num_layers=2,
                                         batch_first=True, bidirectional=bidirectional, dropout=self.dropout)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model * 4 if bidirectional else self.d_model * 2,
                      out_features=self.d_model * 2 if bidirectional else self.d_model),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model * 4 if bidirectional else self.d_model * 2,
                      out_features=self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model, out_features=self.n_steps_out),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        agg_prod, hist_wx, trg_wx = x1, x2, x3

        agg_prod, _ = self.agg_production(agg_prod)
        hist_wx, _ = self.historic_weather(hist_wx)
        trg_wx, _ = self.target_weather(trg_wx)

        fusion = torch.cat((agg_prod[:, -1, :], hist_wx[:, -1, :]), 1)
        fusion = self.fc1(fusion)
        fusion = torch.cat((fusion, trg_wx[:, -1, :]), 1)
        out = self.fc2(fusion)

        return out


class MATNet(nn.Module):
    """ Multi-Level Fusion and Self-Attention Transformer-Based Model for
    Multivariate Multi-Step Day-Ahead PV Generation Forecasting.

    Parameters
    ----------
    pv_features: int, optional, default: 1
        The number of features in the PV production data.
    hw_features: int, optional, default: 101
        The number of features in the historic weather data.
    fw_features: int, optional, default: 101
        The number of features in the forecasting weather data (Day Ahead).
    n_steps_in: int, optional, default: 336
        The number of time steps (hours) in the input sequence.
    n_steps_out: int, optional, default: 24
        The number of time steps (hours) in the output sequence.
    d_model : int
        The number of features in the input sequence.
    nhead : int
        The number of heads in the attention layer.
    dim_feedforward : int
        The number of features in the feedforward layers.

    """

    def __init__(self, hw_features: int, fw_features: int, pv_features: int, n_steps_in: int,
                 n_steps_out: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int):
        super().__init__()
        self.ap_features = pv_features
        self.hw_features = hw_features
        self.tw_features = fw_features
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = 0.1

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=0.1,
                                                   activation='relu', layer_norm_eps=1e-05,
                                                   batch_first=True, norm_first=True)

        self.production_input = nn.Sequential(
            layers.Conv1d(channels_last=True, in_channels=self.ap_features, out_channels=self.d_model,
                          kernel_size=3, stride=1, padding='same'),
            layers.PositionalEncoding(d_model=self.d_model, max_len=self.n_steps_in,
                                      dropout=self.dropout, batch_first=True),
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        )

        self.historic_weather_input = nn.Sequential(
            layers.Conv1d(channels_last=True, in_channels=self.hw_features,
                          out_channels=self.d_model, kernel_size=1, padding='same'),
            layers.PositionalEncoding(d_model=self.d_model, max_len=self.n_steps_in,
                                      dropout=self.dropout, batch_first=True),
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        )

        self.target_weather_input = nn.Sequential(
            layers.Conv1d(channels_last=True, in_channels=self.tw_features,
                          out_channels=self.d_model, kernel_size=1, padding='same'),
            layers.PositionalEncoding(d_model=self.d_model, max_len=self.n_steps_out,
                                      dropout=self.dropout, batch_first=True),
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model * 2,
                      out_features=self.d_model),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model * 2,
                      out_features=self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model, out_features=n_steps_out),
            nn.Sigmoid(),
        )

        self.linear1 = nn.Linear(in_features=self.n_steps_in,
                                 out_features=self.n_steps_in)
        self.linear2 = nn.Linear(in_features=self.n_steps_in,
                                 out_features=self.n_steps_in)
        self.linear3 = nn.Linear(in_features=self.n_steps_out,
                                 out_features=self.n_steps_out)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        agg_prod, hist_wx, trg_wx = x1, x2, x3  # (batch_size, seq_len, num_features)

        # Multi-Head Attention
        agg_prod = self.production_input(agg_prod)  # (batch_size, n_steps_in, d_model)
        hist_wx = self.historic_weather_input(hist_wx)  # (batch_size, n_steps_in, d_model)
        trg_wx = self.target_weather_input(trg_wx)  # (batch_size, n_steps_out, d_model)

        # Dense Interpolation for Encoding Order
        agg_prod = agg_prod.permute(0, 2, 1)  # (batch_size, d_model, n_steps_in)
        agg_prod = self.linear1(agg_prod)  # (batch_size, d_model, n_steps_in)
        agg_prod = agg_prod.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        hist_wx = hist_wx.permute(0, 2, 1)  # (batch_size, d_model, n_steps_in)
        hist_wx = self.linear2(hist_wx)  # (batch_size, d_model, n_steps_in)
        hist_wx = hist_wx.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        trg_wx = trg_wx.permute(0, 2, 1)  # (batch_size, d_model, n_steps_out)
        trg_wx = self.linear3(trg_wx)  # (batch_size, d_model, n_steps_in)
        trg_wx = trg_wx.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        # Fusion + FC
        fusion = torch.cat((agg_prod, hist_wx), 1)  # (batch_size, 2 x d_model)
        output = self.fc1(fusion)  # (batch_size, d_model)
        fusion2 = torch.cat((output, trg_wx), 1)  # (batch_size, 2 x d_model)
        output = self.fc2(fusion2)  # (batch_size, n_steps_out)

        return output
