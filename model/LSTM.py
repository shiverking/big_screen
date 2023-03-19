# -*- coding: utf-8 -*-
"""
@File    :   LSTM_IR.py
@Time    :   2022/11/3 20:14:11
@Author  :   Jiaxuan Jiang
@Contact :   jiaxuan_Jiang@stu.xjtu.edu.cn
@Desc    :   None
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from model.DBN import DataProcessing
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

hidden_size = 32  # 隐藏层的特征维度
layer_size = 1  # 模型的层数
output_size = 8  # 意图的个数
batch_size_train = 32
batch_size_test = 1
epochs = 300  # 训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def data_rolling(x, seque_size):
    """
    generate the sequential data
    :param x: original data
    :param seque_size: time-step
    :return: time series data
    """
    x_Rolling = []
    for i in range(x.shape[0] - seque_size):
        x_new = x[i:i + seque_size, ]
        x_Rolling.append(x_new)
    return np.array(x_Rolling)


def data_split(dataset, train_name, test_name, seque_size):
    train_data = dataset[dataset['名称'] == train_name]
    test_data = dataset[dataset['名称'] == test_name]

    # normalization
    x_trainNormal = MinMaxScaler().fit_transform(train_data.values[:, 3:])
    x_testNormal = MinMaxScaler().fit_transform(test_data.values[:, 3:])

    x_trainRolling = data_rolling(x_trainNormal, seque_size)
    y_trainRolling = train_data.iloc[seque_size:, 0].values

    x_testRolling = data_rolling(x_testNormal, seque_size)
    y_testRolling = test_data.iloc[seque_size:, 0].values

    # 借助TensorDataset直接将数据包装成dataset类，再使用dataloader(每次从dataset中基于某种采样原则取出一个batch的数据)
    data_train = TensorDataset(torch.from_numpy(x_trainRolling), torch.from_numpy(y_trainRolling))
    data_test = TensorDataset(torch.from_numpy(x_testRolling), torch.from_numpy(y_testRolling))
    data_trainLoader = DataLoader(data_train, batch_size=batch_size_train, shuffle=False)
    data_testLoader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

    return data_trainLoader, data_testLoader


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)  # input.shape (batch, L, input_size)
        # 全连接层
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_data):
        #  h0,c0.shape (1, batch, hidden_size)   out.shape (L, batch, num_directions x hidden_size)
        h0 = torch.randn(self.layer_dim, input_data.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.rand(self.layer_dim, input_data.size(0), self.hidden_dim).requires_grad_().to(device)
        lstm_out, (h_n, cn) = self.lstm(input_data, (h0.detach(), c0.detach()))
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=-1)
        out = self.linear(feature_map)
        return out


def accuracy_Train(train_loader, sequence_size, input_size, model, epochMax=30, learning_rate=0.1):
    timeTotal_start = time.time()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 模型训练
    loss_list, accuracy_list, iteration_list, t_trainList = [], [], [], []  # 迭代次数
    iter, correct, total = 0, 0.0, 0.0
    for epoch in range(epochMax):
        for i, (seq, labels) in enumerate(train_loader):
            time_start = time.time()
            model.train()
            seq = seq.view(-1, sequence_size, input_size).float().requires_grad_().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            # 前向传播
            outputs = model(seq)
            # 计算损失
            loss = loss_function(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 计数器加1
            iter += 1
            # 获取预测概率最大值的下标
            predict = torch.max(outputs.data, 1)[1]
            # 统计测试集的大小
            total += labels.size(0)
            # 统计判断预测正确的数量
            correct += (predict == labels).sum()
            time_end = time.time()  # 记录结束时间
            t_trainList.append(time_end - time_start)

        # 计算accuracy
        accuracy = correct / total * 100
        accuracy_list.append(accuracy)
        loss_list.append(loss.data)
        iteration_list.append(iter)

    timeTotal_end = time.time()  # 记录结束时间
    # print("accuracy-train", accuracy.numpy())
    # print("Average TrainTime:{}".format(np.average(t_trainList)))
    # print("Running TotalTime:{}".format(timeTotal_end - timeTotal_start))

    return model, iteration_list, loss_list, accuracy_list


def figure_loss(iteration_list, loss_list, accuracy_list, train_name):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(iteration_list, loss_list)
    axes[0].set_ylabel("train_Loss LSTM")

    axes[1].plot(iteration_list, accuracy_list, color='r')
    axes[1].set_xlabel("Number of iteration")
    axes[1].set_ylabel("train_Accuracy LSTM")
    fig.suptitle("model_train LSTM ({}})".format(train_name))
    plt.show()


def accuracy_Test(test_loader, net, test_name, sequence_size, input_size):
    Intention = [[] for _ in range(output_size)]
    y_predict, y_test, t_list = [], [], []

    net.eval()
    target_num = torch.zeros((1, output_size))
    predict_num = torch.zeros((1, output_size))
    acc_num = torch.zeros((1, output_size))

    # 迭代测试集，获取数据、预测
    for j, (seqT, labelsT) in enumerate(test_loader):
        time_start = time.time()
        seqT = seqT.view(-1, sequence_size, input_size).float().to(device)
        labelsT = labelsT.to(device)
        # 模型预测
        outputs = net(seqT)
        predicted = torch.max(outputs.data, 1)[1]

        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
        tar_mask = torch.zeros(outputs.size()).scatter_(1, labelsT.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)  # 得到数据中每类的数量
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量

        predict = torch.max(outputs.data, 1)[1]  # 获取预测概率最大值的下标
        y_predict.append(predict.cpu().numpy())  # 获取预测概率最大值的下标
        y_test.append(labelsT.cpu().numpy())  # 统计测试集的大小
        prob = F.softmax(outputs.data, dim=1)  # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax

        for k in range(output_size):
            Intention[k].append((prob[0][k]).cpu().numpy())

        time_end = time.time()  # 记录结束时间
        t_list.append(time_end - time_start)

    Aver_testTime = np.average(t_list)
    perfor_report = classification_report(y_test, y_predict)

    plt.figure()
    '''陈嘉文加'''
    plot_print = list() 
    IntentionName = ['超高空盘旋', '高空盘旋', '中空高速巡航', '中空巡航', '中空盘旋', '低空高速盘旋', '低空盘旋', '低空高速巡航']
    for k in range(output_size):
        plot_print.append({'name': IntentionName[k], 'data': [float(x) for x in Intention[k]]})
        plt.plot(range(len(y_predict)), Intention[k])
    plt.legend(['超高空盘旋', '高空盘旋', '中空高速巡航', '中空巡航', '中空盘旋', '低空高速盘旋', '低空盘旋', '低空高速巡航'])
    plt.xlabel("Number of Step")
    plt.ylabel("Intention")
    plt.title("model_test LSTM ({})".format(test_name))

    return Aver_testTime, perfor_report, target_num, predict_num, acc_num, plt, plot_print, IntentionName


def LSTM_RES():
    time_start = time.time()  # 记录开始时间
    data = pd.read_excel('./data/正南打击利佩茨克机场数据/data_new.xlsx')  # 导入原始数据
    typeName = '战斗机'
    sequence_size = 1
    data_proc = DataProcessing(data, typeName)  # 数据预处理
    data_coding = data_proc.data_Code  # 输入模型的数据
    trainName = 'F-22 科加尔尼西亚 #5'  # 输入训练目标
    testName = 'F-22 科加尔尼西亚 #6'  # 输入测试目标——【目标选择】
    trainData, testData = data_split(data_coding, trainName, testName, sequence_size)  # 训练/测试集划分

    n_feature = data_coding.iloc[:, 3:]
    input_size = n_feature.shape[1]  # 特征维度

    model = LSTM(input_size, hidden_size, layer_size, output_size)

    modelTr, iter_list, loss_list, acc_list = accuracy_Train(train_loader=trainData, sequence_size=1, input_size=input_size,\
            model=model, epochMax=30, learning_rate=0.1)  # 模型训练结果
    Aver_time, perform_report, target_num, predict_num, acc_num, plt, plot_print, legend = accuracy_Test(testData, modelTr, testName,
                                                                                     sequence_size, input_size)  # 模型测试结果
    # plt.show()    # 【可视化展示1】：各模型的意图识别结果图
    # print("Average TestTime:{}".format(Aver_time))
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start 
    # 【可视化展示2-2】：统计信息图（同DBN，展示一次就好）+性能指标值输出+各个模型预测精度对比图
    print("perform_report:'\n' {}".format(perform_report))

    # 【可视化展示2-3】：统计信息图+性能指标值输出+各个模型预测精度对比图
    precision = (acc_num / predict_num * 100).squeeze(0).numpy().tolist()
    precision = [0 if math.isnan(x) else x for x in precision]
    IntentionName = ['超高空盘旋', '高空盘旋', '中空高速巡航', '中空巡航', '中空盘旋', '低空高速盘旋', '低空盘旋', '低空高速巡航']

    fig, ax = plt.subplots()
    ax.barh(IntentionName, precision, align='center', label='LSTM', color='orange')
    for a, b in zip(IntentionName, precision):
        plt.text(b + 1, a, "%.2f%%" % b, ha='center', va='center', fontsize=8, color='k')
    ax.set_yticks(IntentionName)
    ax.set_xlabel('precision(%)')
    ax.set_title('各个模型预测精度对比图')
    # plt.show()

    # 【可视化展示3】：不同模型在不同冷启动时间下的准确率表
    sequence_size = [1, 5, 10, 15, 20]
    precision_list, accuracy_list, Aver_timeList = [], [], []
    for i in sequence_size:
        trainData, testData = data_split(data_coding, trainName, testName, i)  # 训练/测试集划分

        modelTr, iter_list, loss_list, acc_list = accuracy_Train(train_loader=trainData, sequence_size=i, input_size=input_size,\
            model=model, epochMax=30, learning_rate=0.1)  # 模型训练结果
        Aver_time, perform_report, target_num, predict_num, acc_num, plt, _, _ = accuracy_Test(testData, modelTr,
                                                                                         testName, i, input_size)  # 模型测试结果
        precision = (acc_num / predict_num * 100).squeeze(0).numpy().tolist()
        precision_list.append(precision)       # 各类精确率
        accuracy = (100. * acc_num.sum(1) / target_num.sum(1)).numpy()
        accuracy_list.append(accuracy)         # 整体准确率

        Aver_timeList.append(Aver_time)        # 预测时间
    precision_data = np.vstack((np.array(precision_list).T, np.array(accuracy_list).reshape(-1)))

    fig, ax = plt.subplots()
    IntentionName.append('整体')
    rowColours = ["#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F2F2F2"]
    column_labels = ["LSTM(t=1)", "LSTM(t=5)", "LSTM(t=10)", "LSTM(t=15)", "LSTM(t=20)"]
    colColors = ["#00ccff"] * len(column_labels)

    ax.axis('off')       # 取消坐标轴
    ax.table(cellText=np.round(precision_data, 2), colLabels=column_labels, colColours=colColors, rowColours=rowColours,
             rowLabels=IntentionName, cellLoc='center', rowLoc='center', loc="center")
    ax.set_title('不同模型在不同冷启动时间下的准确率表')
    # plt.show()

    # 【可视化展示4】：运行时间
    plt.figure()
    x = ['time_step=1', 'time_step=5', 'time_step=10', 'time_step=15', 'time_step=20']
    y = Aver_timeList

    plt.plot(x, y, color='k', linewidth=1, marker='o', markersize=5, label="预测时间")
    for a, b in zip(x, y):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=11, color='k')
    plt.title("各模型的预测时间", color='k')
    plt.show()
    print(plot_print, legend, time_sum)
    return plot_print, legend, time_sum, x, y, IntentionName, precision

if __name__ == '__main__':
    LSTM_RES()