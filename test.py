# -*- coding: utf-8 -*-
# @File    : train.py
# @Software: PyCharm

import time
import torch
import matplotlib.pyplot as plt 
from option.test_option import TestOptions
from data import create_dataset
from models import create_model
from tqdm import  tqdm
import sys
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    opt = TestOptions().parse()
    # 通过参数建立DataSet
    dataset = create_dataset(opt)
    print(dataset)

    # 通过参数建立模型
    model = create_model(opt).to(dev)
    
    model.eval()
    mse_loss = torch.nn.MSELoss(reduction="sum")

    realL,predL = [],[]
    with torch.no_grad():
        progressbar = tqdm(dataset["Test"],desc="Testing")
        for data in progressbar:
            input,target = data
            input = input.to(dev)
            # print(target.size())
            realL.append(target.numpy())
            # print(input)
            pred = model(input)
            predL.append(pred.detach().cpu().numpy())

    # [[batch,1],[batch,1]...]
    real = [i.item() for x in realL for i in x]
    pred = [i.item() for x in predL for i in x]


    #plot_size = len(real)
    plot_size = 1000
    plt.figure(figsize = (20,8),dpi = 80)
    plt.plot(range(plot_size),real[:plot_size],label = "real",c = 'k')
    plt.plot(range(plot_size),pred[:plot_size],label = "pred",c = 'r')
    # plt.axvline(x = 554,color='b',ls="--",label = "fault")
    # plt.axvline(x = 509,color='b',ls="--")
    plt.ylabel("%")
    plt.title("LSTM")
    plt.legend()
    plt.savefig("pred_res.png")
    plt.show()






    # plt.title('Training Progress')
    # # 设置y轴的刻度为对数刻度，使得损失值的变化更加明显
    # plt.yscale("log")
    # # 绘制训练集的损失曲线，使用蓝色实线，标签为“train”
    # plt.plot(trainLoss, label='train')
    # # 绘制验证集的损失曲线，使用橙色实线，标签为“validation”
    # plt.plot(valLoss, label='validation')
    # # 设置y轴的标签为“Loss”
    # plt.xlabel("Epoch")
    # plt.legend()
    # plt.savefig("./loss.png",dpi = 120)
    # plt.show()





