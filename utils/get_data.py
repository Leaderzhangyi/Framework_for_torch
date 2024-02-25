# -*- coding: utf-8 -*-
# @File    : get_data.py
# @Software: PyCharm

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.base_dataset import TimeDataset
from torch.utils.data import DataLoader
import pandas as pd 

def sliding_window(df_data:pd.DataFrame, seq_len:int,target_index:int):
    X = []
    Y = []
    df_x = df_data.drop(df_data.columns[target_index],axis=1)
    for i in range(len(df_data) - seq_len):
        X.append(df_x.iloc[i:i + seq_len].values)
        Y.append(df_data.iloc[i+seq_len,target_index])
    # X = np.array(X).transpose(0,2,1)
    return X, Y


def generate_dataset(normalized_data,batch_size):
    myloader = {}
    for name,item in zip(["Train","Val","Test"],normalized_data):
        myloader[name] = DataLoader(dataset = TimeDataset(item[0],item[1]),shuffle=False,batch_size=batch_size)
    return myloader



def stand_and_load(*args):
    x_train, x_val, x_test, y_train, y_val, y_test,batch_size = args

    # seq_len,features_len  20,71
    # print(x_train.size())
    # print(y_train.size())
    # 归一化
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    data_to_normalize = [(x_train, y_train), (x_val, y_val), (x_test, y_test)]

    normalized_data = []

    # 循环归一化数据
    for x, y in data_to_normalize:
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, 1)

        x_norm = scalerX.fit_transform(x_flat)
        y_norm = scalerY.fit_transform(y_flat)

        x_norm = torch.from_numpy(x_norm).reshape(x.shape).type(torch.FloatTensor)
        y_norm = torch.from_numpy(y_norm).reshape(y_flat.shape).type(torch.FloatTensor)
        # print(y_norm.size())
        normalized_data.append((x_norm, y_norm))

    return generate_dataset(normalized_data,batch_size)




def split_training_datasets(df,seq_len,train_rate,test_len,target_index,batch_size):
    X,Y = sliding_window(df,seq_len,target_index)
    x_train,y_train,x_test,y_test =  X[0:-test_len], Y[0:-test_len], X[-test_len:],Y[-test_len:]
    train_len = round(len(x_train) * train_rate)
    X_train, X_val, Y_train, Y_val = x_train[0:train_len], x_train[train_len:], y_train[0:train_len], y_train[train_len:]
    print(np.array(X_train).shape)
    x_train, y_train, x_val, y_val, x_test, y_test = map(torch.tensor, map(np.array,(X_train, Y_train, X_val, Y_val, x_test, y_test)))
    x_train, y_train, x_val, y_val, x_test, y_test = map(lambda x: x.to(torch.float32),
                                                         (x_train, y_train, x_val, y_val, x_test, y_test))

    # x_train = x_train.reshape(-1,seq_len,x_train.size()[])
    # print(x_train.size())
    return stand_and_load(x_train, x_val, x_test, y_train, y_val, y_test,batch_size)
