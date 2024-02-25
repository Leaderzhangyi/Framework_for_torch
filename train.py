# -*- coding: utf-8 -*-
# @File    : train.py
# @Software: PyCharm


import time
import torch
import matplotlib.pyplot as plt 
from option.train_option import TrainOptions
from data import create_dataset
from models import create_model
from tqdm import  tqdm
import numpy as np 
import sys
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_test(realL,predL,index):
    

    real = [i.item() for x in realL for i in x]
    pred = [i.item() for x in predL for i in x]
     #plot_size = len(real)
    plot_size = len(real)
    plt.figure(figsize = (20,8),dpi = 80)
    plt.plot(range(plot_size),real[:plot_size],label = "real",c = 'k')
    plt.plot(range(plot_size),pred[:plot_size],label = "pred",c = 'r')
    plt.ylabel("%")
    plt.title("LSTM")
    plt.legend()
    plt.savefig(f"pred_lstm_{index}.png")
    plt.close()

def get_loss_plot(x,y):
    
    plt.figure(figsize=(12,8))
    plt.title('Training Progress')
    # 设置y轴的刻度为对数刻度，使得损失值的变化更加明显
    plt.yscale("log")
    # 绘制训练集的损失曲线，使用蓝色实线，标签为“train”
    plt.plot(x, label='train')
    # 绘制验证集的损失曲线，使用橙色实线，标签为“validation”
    plt.plot(y, label='validation')
    # 设置y轴的标签为“Loss”
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("./loss.png",dpi = 120)
    plt.show()


if __name__ == '__main__':
    opt = TrainOptions().parse()

    # 通过参数建立DataSet
    dataset = create_dataset(opt)
    print(dataset)

    # 通过参数建立模型

    model = create_model(opt).to(dev)
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    mse_loss = torch.nn.MSELoss()

    trainLoss,valLoss = [],[]

    min_val_loss =  sys.maxsize
    for epoch in range(opt.epochs):
        progressbar = tqdm(dataset["Train"],desc="Training")
        for step,data in enumerate(progressbar):
            input,target = data
            input = input.to(dev)
            target = target.to(dev)
            # print(input)
            pred = model(input)
            train_loss = mse_loss(pred,target)
            progressbar.set_postfix(loss = train_loss.item())
            # print(f"{target[0]} --- pred:{pred[0]}")
            if (step + 1) % 50 == 0:
                plot_test(target.detach().cpu().numpy(),pred.detach().cpu().numpy(),step + 1)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        with torch.no_grad():
            for batch in dataset["Val"]:
                val_input, val_target = batch
                val_input = val_input.to(dev)
                val_target = val_target.to(dev)
                val_pred = model(val_input)
                val_loss = mse_loss(val_pred,val_target)

        trainLoss.append(train_loss.item())
        valLoss.append(val_loss.item())

        if val_loss.item() < min_val_loss:
            torch.save(model.state_dict(),"beat_params.pth")
            min_val_loss = val_loss.item()

        print(f'Epoch {epoch + 1}/{opt.epochs}, Train Loss: {round(train_loss.item(), 4)}, Val Loss: {round(val_loss.item(), 4)}')
    get_loss_plot(trainLoss,valLoss)



    





