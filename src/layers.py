# %%
import math

import torch
import torch.nn as nn
from typing import Union
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, channels_last, *args, **kwargs):
        """ 1D convolution layer.
        Args:
            in_channels (int): Number of channels in the input
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        """
        super().__init__(*args, **kwargs)
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the 1D convolution layer.
        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_len),
            (batch_size, seq_len, channels) or (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, seq_len),
            (batch_size, seq_len, channels) or (batch_size, seq_len).
        """
        if x.dim() == 3:
            if x.size(1) == self.in_channels and not self.channels_last:
                return super().forward(x)
            elif x.size(2) == self.in_channels and self.channels_last:
                output = super().forward(x.transpose(1, 2))
                return output.transpose(1, 2)
            else:
                raise ValueError("Input has incorrect number of channels")
        elif x.dim() == 2:
            if x.size(0) == self.in_channels and not self.channels_last:
                return super().forward(x)
            elif x.size(1) == self.in_channels and self.channels_last:
                output = super().forward(x.transpose(0, 1))
                return output.transpose(0, 1)
        else:
            raise ValueError("Input must have 3 or 2 dimensions (batch_size, channels, seq_len) "
                             "or (batch_size, seq_len)")


class PositionalEncoding(torch.nn.Module):
    """ A PyTorch implementation of positional encoding as described in the paper "Attention is All You Need"
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): the number of expected features in the input (required).
        batch_first (bool, optional): whether the input tensor should be of shape (batch_size, seq_len, d_model)
        or (seq_len, batch_size, d_model). Default is True (input has shape (batch_size, seq_len, d_model)).
        dropout (float, optional): the dropout probability (default=0.1).
        max_len (int, optional): the maximum length of the input sequence (default=5000).
        base (float, optional): the base used in the sinusoidal functions (default=10000.0).
    """

    def __init__(self, d_model: int, batch_first: bool = True, dropout: float = 0.1,
                 max_len: int = 5000, base: float = 10000.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.base = base

        # create the sinusoidal functions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(self.base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the PositionalEncoding layer.

        Inputs:
            x (torch.Tensor): a float tensor of shape (batch_size, seq_len, d_model) if batch_first is equal to True,
            otherwise with shape (seq_len, batch_size, d_model). It represents the input sequence to be encoded.

        Outputs:
            output (torch.Tensor): a float tensor of shape (batch_size, seq_len, d_model) if batch_first is equal
            to True, otherwise with shape (seq_len, batch_size, d_model). It represents the encoded input sequence.
        """
        if self.batch_first:
            x = x.permute(1, 0, 2)
            x = (x + self.pe[:x.size(0), :]).permute(1, 0, 2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SoftAttention(nn.Module):
    """ Soft Attention mechanism.

    Attributes
    ----------
    input_dim : int
        Number of features in the input tensor.
    hidden_dim : int
        Number of features in the hidden layer.
    lmb : float
        Scaling factor for the attention weights. If lmb is equal to 0, the attention mechanism is equivalent to a
        standard weighted sum.
    """

    def __init__(self, input_dim, hidden_dim, lmb=1.0):
        super(SoftAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lmb = lmb  #

        self.fc = nn.Linear(input_dim, hidden_dim, bias=True)
        self.u = nn.Parameter(torch.Tensor(hidden_dim))
        nn.init.zeros_(self.u)

    def forward(self, x):
        """ Forward pass of the SoftAttention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, N, input_dim). where N is the number of elements to fuse.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_dim).
        """
        h = torch.tanh(self.fc(x))  # shape: (batch_size, N, hidden_dim)
        e = torch.matmul(h, self.u)  # shape: (batch_size, N)

        alpha = torch.softmax(self.lmb * e, dim=1)  # shape: (batch_size, N)

        out = torch.sum(alpha.unsqueeze(-1) * x, dim=1)  # shape: (batch_size, input_dim)

        return out
