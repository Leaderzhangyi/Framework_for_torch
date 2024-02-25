from torch.nn import Module, LSTM, Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm


class Crop(nn.Module):

    # 定义初始化方法，接受一个参数crop_size，表示要裁剪的长度
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    # 定义前向传播方法，接受一个参数x，表示输入的张量
    def forward(self, x):
        """
        这是一个裁剪块，用来裁剪多出来的padding
        :param x:
        :return:
        """
        # 返回x的切片，保留所有的维度，但是在最后一个维度上去掉最后的crop_size个元素，然后调用contiguous方法，保证内存中的数据是连续的
        return x[:, :, :-self.crop_size].contiguous()


class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        """
        相当于一个residual block
        :param n_inputs:    输入通道数，就是一个时间点上有几维特征，除去第一层外也就是上层TCN的卷积核数目
        :param n_outputs:   输出通道数，就是有几个卷积核，要做几次卷积，保存在那个列表中，
        :param kernel_size: 卷积核尺寸
        :param stride:      步长，一般为1
        :param dilation:    膨胀系数
        :param dropout:     dropout概率
        """
        super(TemporalCasualLayer, self).__init__()
        # 计算填充长度，等于卷积核大小减一乘以卷积扩张率
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }

        # 创建一个一维卷积层（Conv1d）对象，使用权重归一化（weight_norm）进行包装，输入通道数为n_inputs，输出通道数为n_outputs，其他参数从conv_params字典中获取，赋值给self.conv1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        # 经过conv1，输出的size其实是（Batch, input_channel, seq_len + padding）
        self.crop1 = Crop(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 创建另一个一维卷积层（Conv1d）对象，使用权重归一化（weight_norm）进行包装，输入通道数和输出通道数都为n_outputs，其他参数从conv_params字典中获取，赋值给self.conv2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)

        # 判断输入通道数和输出通道数是否相等，如果不相等，就创建一个一维卷积层（Conv1d）对象，输入通道数为n_inputs，输出通道数为n_outputs，
        # 卷积核大小为1，用来实现通道数的变换，赋值给self.bias；如果相等，就把self.bias赋值为None
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 这里是并联一✖一卷积的残差链接
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        # 判断self.bias是否为None，如果是，就把x赋值给b；如果不是，就调用self.bias对x进行处理，得到b
        b = x if self.bias is None else self.bias(x)
        # 调用self.relu对y和b相加的结果进行处理，返回最终的输出,残差连接
        return self.relu(y + b)


class TemporalConvolutionNetwork(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，现在给出的TCN结构很好支持每个时刻为一个
        :param num_inputs:      输入通道数
        :param num_channels:    每层的hidden_channel数目，是一个列表，就是每一层残差块的输出维度
        :param kernel_size:     卷积核尺寸
        :param dropout:         dropout概率
        """
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }
        for i in range(num_levels):
            dilation = 2 ** i  # 膨胀系数 1，2，4，8
            in_ch = num_inputs if i == 0 else num_channels[i - 1]  # 每一层的输入通道，第一层TCN的输入通道就是网络的输入通道，之后每一层的输入通道是上一层的输出通道数目
            out_ch = num_channels[i]  # 确定每一层的输出通道数目
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)  # 定义残差块的层数

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):

    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size,
                                              dropout=dropout)  # TCN部分
        # self.linear = nn.Linear(num_channels[-1], output_size)  # 全连接层
        self.pool = nn.MaxPool1d(kernel_size)  # 默认在最后一个维度池化

    def forward(self, x):
        y = self.tcn(x)
        return self.pool(y)  # 对TCN输出进行池化


# class LSTM_NET(Module):

#     def __init__(self, data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout,
#                  LSTM_hidden_size, LSTM_num_layers, LSTM_dropout, output_size):
#         super().__init__()
#         self.hidden_size = LSTM_hidden_size
#         self.num_layers = LSTM_num_layers
#         self.tcn = TCN(data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout)
#         self.lstm = LSTM(TCN_kernel_num[-1], LSTM_hidden_size,
#                          LSTM_num_layers, LSTM_dropout, batch_first=True)
#         self.linear = Linear(in_features=TCN_kernel_num[-1], out_features=output_size)

#     def forward(self, x):
#         h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)
#         c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)
#         tcn_feature_map = self.tcn(x)
#         tcn_feature_map = tcn_feature_map.transpose(1, 2)
#         lstm_out, (_, _) = self.lstm(tcn_feature_map, (h0.detach(), c0.detach()))
#         linear_out = self.linear(lstm_out)

#         return linear_out

class LSTM_NET(Module):
    def __init__(self, data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout,
                 LSTM_hidden_size, LSTM_num_layers, LSTM_dropout, output_size):
        super().__init__()
        self.hidden_size = LSTM_hidden_size
        self.num_layers = LSTM_num_layers
        self.TCN_kernel_num = TCN_kernel_num
        self.tcn = TCN(data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout)
        self.lstm = LSTM(input_size=TCN_kernel_num[-1], hidden_size=LSTM_hidden_size, num_layers=LSTM_num_layers,
                         batch_first=True)
        # print("TCN_kernel_num[-1].shape")
        self.linear = Linear(in_features=LSTM_hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        tcn_feature_map = self.tcn(x)
        # (512,57,90)
        tcn_feature_map = tcn_feature_map.permute(0, 2, 1)
        # (512,500,18)
        # tcn_feature_map = tcn_feature_map.transpose(1, 2)  # 将第二维和第三维进行转置，形状变为[batch_size, num_channels, seq_len]

        lstm_out, (_, _) = self.lstm(tcn_feature_map, (h0, c0))
        linear_out = self.linear(lstm_out[:, -1, :])  # 取最后一个时间步的输出

        return linear_out


class TCN_NET(Module):
    def __init__(self, data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout, output_size):
        super().__init__()

        self.TCN_kernel_num = TCN_kernel_num
        self.tcn = TCN(data_input_col, TCN_kernel_num, TCN_kernel_size, TCN_dropout)

        self.linear = Linear(in_features=TCN_kernel_num[-1], out_features=output_size)

    def forward(self, x):
        tcn_feature_map = self.tcn(x)

        tcn_feature_map = tcn_feature_map.permute(0, 2, 1)

        linear_out = self.linear(tcn_feature_map[:, -1, :])  # 取最后一个时间步的输出

        return linear_out
