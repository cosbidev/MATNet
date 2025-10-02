# %%
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import utils, layers


def dropout_without_scaling(x, p=0.5):
    mask = (torch.rand_like(x) > p).float()  # Generate binary mask
    return x * mask  # Apply mask without scaling


class MPVForecaster(pl.LightningModule):
    def __init__(self, model_name, train_loader, model_kwargs, lr, p=0):
        super().__init__()
        self.save_hyperparameters()
        self.model = utils.get_model(model_name, **model_kwargs)
        # self.example_input_array = next(iter(train_loader))[0]
        self.dropout = nn.Dropout()
        self.lr = lr

        self.p = p

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

        pv_production = dropout_without_scaling(pv_production, p=self.p)
        wx_history = dropout_without_scaling(wx_history, p=self.p)
        wx_forecast = dropout_without_scaling(wx_forecast, p=self.p)

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


class MATNet(nn.Module):
    """ Multi-Head Attention Multivariate Model for Photovoltaic Power Production Forecasting.

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
    num_heads : int
        The number of heads in the attention layer.
    dim_feedforward : int
        The number of features in the feedforward layers.
    dropout : float, optional, default: 0.1
        The dropout rate.
    """

    def __init__(self, hw_features: int, fw_features: int, pv_features: int, n_steps_in: int,
                 n_steps_out: int, d_model: int, num_heads: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1, interpolation: str = "Adaptive",
                 interp_factor: Union[int, None] = None, fusion: str = "Concat"):
        super().__init__()
        self.ap_features = pv_features
        self.hw_features = hw_features
        self.tw_features = fw_features
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.interpolation = interpolation
        self.interp_factor = interp_factor
        self.fusion = fusion

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads,
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
            nn.Linear(in_features=2 * self.d_model if self.fusion in ["Concat", "FBP"] else self.d_model,
                      out_features=self.d_model),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model * 2 if self.fusion in ["Concat", "FBP"] else self.d_model,
                      out_features=self.d_model),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.d_model, out_features=self.n_steps_out),
            nn.Sigmoid(),
        )

        if self.interpolation == "Adaptive":
            self.linear1 = nn.Linear(in_features=self.n_steps_in,
                                     out_features=self.interp_factor if self.interp_factor else self.n_steps_in)
            self.linear2 = nn.Linear(in_features=self.n_steps_in,
                                     out_features=self.interp_factor if self.interp_factor else self.n_steps_in)
            self.linear3 = nn.Linear(in_features=self.n_steps_out,
                                     out_features=self.interp_factor if self.interp_factor else self.n_steps_out)
        elif self.interpolation == "Classic":
            W = torch.zeros((self.n_steps_in, self.n_steps_in))
            for t in range(1, self.n_steps_in + 1):
                s = self.interp_factor * t / self.n_steps_in
                for m in range(1, self.n_steps_in + 1):
                    W[t - 1, m - 1] = pow(1 - abs(s - m) / self.n_steps_in, 2)
            self.register_buffer('W', W)

        if self.fusion == "SoftAttention":
            self.attention1 = layers.SoftAttention(input_dim=self.d_model, hidden_dim=self.d_model, lmb=1.0)
            self.attention2 = layers.SoftAttention(input_dim=self.d_model, hidden_dim=self.d_model, lmb=1.0)

        self.wx_forecast = None  # Check if the weather forecast is available
        self.wx_history = None  # Check if the weather history is available
        self.pv_forecast = None  # Check if the PV production is available

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        agg_prod, hist_wx, trg_wx = x1, x2, x3  # (batch_size, seq_len, num_features)

        self.wx_forecast = not torch.all(~trg_wx.bool()).item()  # Check if the weather forecast is available
        self.wx_history = not torch.all(~hist_wx.bool()).item()  # Check if the weather history is available
        self.pv_forecast = not torch.all(~agg_prod.bool()).item()  # Check if the PV production is available

        # Multi-Head Attention
        agg_prod = self.production_input(agg_prod)  # (batch_size, n_steps_in, d_model)
        hist_wx = self.historic_weather_input(hist_wx)  # (batch_size, n_steps_in, d_model)
        trg_wx = self.target_weather_input(trg_wx)  # (batch_size, n_steps_out, d_model)

        # Dense Interpolation for Encoding Order
        agg_prod = agg_prod.permute(0, 2, 1)  # (batch_size, d_model, n_steps_in)
        if self.interpolation == "Adaptive":
            agg_prod = self.linear1(agg_prod)  # (batch_size, d_model, n_steps_in)
        elif self.interpolation == "Classic":
            agg_prod = torch.matmul(agg_prod, self.W)
        agg_prod = agg_prod.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        hist_wx = hist_wx.permute(0, 2, 1)  # (batch_size, d_model, n_steps_in)
        if self.interpolation == "Adaptive":
            hist_wx = self.linear2(hist_wx)  # (batch_size, d_model, n_steps_in)
        elif self.interpolation == "Classic":
            hist_wx = torch.matmul(hist_wx, self.W)
        hist_wx = hist_wx.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        trg_wx = trg_wx.permute(0, 2, 1)  # (batch_size, d_model, n_steps_out)
        if self.interpolation == "Adaptive":
            trg_wx = self.linear3(trg_wx)  # (batch_size, d_model, n_steps_out)
        elif self.interpolation == "Classic":
            trg_wx = torch.matmul(trg_wx, self.W)
        trg_wx = trg_wx.permute(0, 2, 1)[:, -1, :]  # (batch_size, d_model)

        # Fusion + FC
        if self.fusion == "Concat":
            fusion = torch.cat((agg_prod, hist_wx), 1)  # (batch_size, 2 * d_model)
            output = self.fc1(fusion)  # (batch_size, d_model)
            fusion2 = torch.cat((output, trg_wx), 1)  # (batch_size, 2 * d_model)
            output = self.fc2(fusion2)  # (batch_size, n_steps_out)
        elif self.fusion == "SoftAttention":
            if self.pv_forecast and self.wx_history and self.wx_forecast:
                fusion = self.attention1(torch.stack([agg_prod, hist_wx], dim=1))  # (batch_size, d_model)
                output = self.fc1(fusion)
                fusion2 = self.attention2(torch.stack([output, trg_wx], dim=1))  # (batch_size, d_model)
                output = self.fc2(fusion2)
            elif self.pv_forecast and self.wx_history and not self.wx_forecast:
                fusion = self.attention1(torch.stack([agg_prod, hist_wx], dim=1))  # (batch_size, d_model)
                output = self.fc1(fusion)
                output = self.fc2(output)
            elif self.pv_forecast and not self.wx_history and self.wx_forecast:
                output = self.fc1(agg_prod)
                fusion2 = self.attention2(torch.stack([output, trg_wx], dim=1))  # (batch_size, d_model)
                output = self.fc2(fusion2)
            elif not self.pv_forecast and self.wx_history and self.wx_forecast:
                output = self.fc1(hist_wx)
                fusion2 = self.attention2(torch.stack([output, trg_wx], dim=1))  # (batch_size, d_model)
                output = self.fc2(fusion2)
            elif self.pv_forecast and not self.wx_history and not self.wx_forecast:
                output = self.fc2(agg_prod)
            elif not self.pv_forecast and self.wx_history and not self.wx_forecast:
                output = self.fc2(hist_wx)
            elif not self.pv_forecast and not self.wx_history and self.wx_forecast:
                output = self.fc2(trg_wx)
        return output
