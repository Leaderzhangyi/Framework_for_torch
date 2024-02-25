# -*- coding: utf-8 -*-
# @File    : lstm_model.py
# @Software: PyCharm
from torch import nn
import torch
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.activation = nn.ReLU()  # 添加ReLU激活函数

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(dev)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(dev)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output[:, -1, :])
        # pred = self.activation(pred)

        # pred = pred[:, -1, :]
        return pred

