# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Software: PyCharm
from utils.get_data import split_training_datasets
import pandas as pd

# def read_data(path):
#     data = pd.read_csv(path)
#     data["CDATETIME"] = pd.to_datetime(data["CDATETIME"])
#     data.set_index('CDATETIME', inplace=True, drop=True)
#     data.sort_index(inplace=True)
#     data.drop(columns=["燃料实际配比01", "燃料实际配比02"], inplace=True)
#     return data

def read_data(path):
    return pd.read_excel(path)


def create_dataset(opt):
    df = read_data(opt.dataroot)
    print(df.head())
    data_loaders = split_training_datasets(df,seq_len = opt.seq_len,train_rate=0.7,test_len=8000,target_index=df.columns.get_loc(opt.target),batch_size=opt.batch_size)
    return  data_loaders










