# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Software: PyCharm
from .lstm_model import LSTM
from .transformer import TransformerModel
from .tcn_model import LSTM_NET
import torch
def create_model(opt):
     model = LSTM(input_size=opt.input_size,hidden_size=opt.hidden_size,num_layers=opt.num_layer,
          output_size=opt.output_size,batch_size=opt.batch_size)
     
     # model = TransformerModel(input_dim=70,output_dim=1,embed_dim=32,num_layers=3,num_heads=4,feedforward_dim=128,dropout=0.3)

     # model_params = {
     # 'data_input_col': 6,
     # 'TCN_kernel_num': [64, 128],
     # 'TCN_kernel_size': 5,
     # 'TCN_dropout': 0.3,
     # 'LSTM_hidden_size': 90,
     # 'LSTM_num_layers': 3,
     # 'LSTM_dropout': 0.3,
     # 'output_size': 1,
     # }
     # model = LSTM_NET(**model_params)

     if opt.test == True:
          model.load_state_dict(torch.load(opt.results_dir))
     return model

