# -*- coding: utf-8 -*-
"""
@File    :   KF-DBN.py
@Time    :   2023/1/3 14:29:00
@Author  :   Jiaxuan Jiang
@Contact :   jiaxuan_Jiang@stu.xjtu.edu.cn
@Desc    :   None
"""

import torch
import numpy as np
import pandas as pd
import time
import math
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
from haversine import haversine
from sklearn.metrics import classification_report

plt.rcParams['font.sans-serif'] = "Microsoft YaHei"  # 正常显示中文标签


class DataProcessing:
    def __init__(self, data, Label):
        global data_Lab
        data_Dist = self.data_distance(data)
        if Label == '战斗机':
            data_Lab = self.data_label_zdj(data_Dist)
        elif Label == '轰炸机':
            data_Lab = self.data_label_hzj(data_Dist)
        elif Label == '无人机':
            data_Lab = self.data_label_wrj(data_Dist)
        elif Label == '侦察机':
            data_Lab = self.data_label_zcj(data_Dist)
        self.data_Code = self.data_coding(data_Lab)

    def data_distance(self, data):
        location_Petersburg = (60.109679822132, 30.3128465163732)  # Levashovo air base （圣彼得堡）(纬度，经度)
        location_Moscow = (55.70488556267239, 36.90488369415833)  # Kubinka air base （莫斯科
        location_Lipetsk = (52.63728683057108, 39.44316608579129)  # 利佩茨克空军机场 （利佩茨克州）
        location_Crimea = (45.09695744641434, 33.66372515706009)  # 萨基空军基地 （克里米亚）
        col_1, col_2, col_3, col_4 = [], [], [], []
        for i in range(data.shape[0]):
            location_a = (data.loc[i]['当前纬度'], data.loc[i]['当前经度'])
            col1, col2 = haversine(location_a, location_Petersburg), haversine(location_a, location_Moscow)
            col3, col4 = haversine(location_a, location_Lipetsk), haversine(location_a, location_Crimea)
            col_1.append(col1), col_2.append(col2), col_3.append(col3), col_4.append(col4)
        data.insert(0, 'distance_Petersburg', col_1), data.insert(1, 'distance_Moscow', col_2)
        data.insert(2, 'distance_Lipetsk', col_3), data.insert(3, 'distance_Crimea', col_4)
        return data

    def data_label_zdj(self, data):
        label = [] * data.shape[0]
        for i in range(data.shape[0]):
            if data.loc[i]['当前海拔高度'] > 11000:
                label.insert(i, '超高空盘旋')
            elif 7620 < data.loc[i]['当前海拔高度'] <= 11000:
                label.insert(i, '高空盘旋')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '中空高速巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '中空巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '中空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '低空高速盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '低空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '低空高速巡航')
            else:
                print(i + 1)
        data.insert(0, 'Intent', label)
        return data

    def data_label_zcj(self, data):
        label = [] * data.shape[0]
        for i in range(data.shape[0]):
            if data.loc[i]['当前海拔高度'] > 11000:
                label.insert(i, '超高空盘旋')
            elif 7620 < data.loc[i]['当前海拔高度'] <= 11000:
                label.insert(i, '高空盘旋')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 350:
                label.insert(i, '中空高速巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] < 350:
                label.insert(i, '中空巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 350:
                label.insert(i, '中空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] >= 350:
                label.insert(i, '低空高速盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 350:
                label.insert(i, '低空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 350:
                label.insert(i, '低空高速巡航')
            else:
                data = data.drop(index=[i], axis=0)
        data.insert(0, 'Intent', label)
        return data

    def data_label_wrj(self, data):
        label = [] * data.shape[0]
        for i in range(data.shape[0]):
            if data.loc[i]['当前海拔高度'] > 11000:
                label.insert(i, '超高空盘旋')
            elif 7620 < data.loc[i]['当前海拔高度'] <= 11000:
                label.insert(i, '高空盘旋')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '中空高速巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '中空巡航')
            elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '中空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '低空高速盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                    or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and data.loc[i][
                '当前速度'] < 480:
                label.insert(i, '低空盘旋')
            elif data.loc[i]['当前海拔高度'] <= 610 and (
                    data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                    or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and data.loc[i][
                '当前速度'] >= 480:
                label.insert(i, '低空高速巡航')
            else:
                print(i + 1)
        data.insert(0, 'Intent', label)
        return data

    def data_label_hzj(self, data):
        label = [] * data.shape[0]
        for i in range(data.shape[0]):
            if data.loc[i]['名称'] == 'B-52 RAF Fairford #3' or data.loc[i]['名称'] == 'B-52 RAF Fairford #2' or \
                    data.loc[i]['名称'] == 'B-52 RAF Fairford #1':
                if data.loc[i]['当前海拔高度'] > 11000:
                    label.insert(i, '超高空盘旋')
                elif 7620 < data.loc[i]['当前海拔高度'] <= 11000:
                    label.insert(i, '高空盘旋')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] >= 450:
                    label.insert(i, '中空高速巡航')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] < 450:
                    label.insert(i, '中空巡航')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] < 450:
                    label.insert(i, '中空盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] >= 450:
                    label.insert(i, '低空高速盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] < 450:
                    label.insert(i, '低空盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] >= 450:
                    label.insert(i, '低空高速巡航')
                else:
                    print(i + 1)
            else:
                if data.loc[i]['当前海拔高度'] > 11000:
                    label.insert(i, '超高空盘旋')
                elif 7620 < data.loc[i]['当前海拔高度'] <= 11000:
                    label.insert(i, '高空盘旋')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] >= 480:
                    label.insert(i, '中空高速巡航')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] < 480:
                    label.insert(i, '中空巡航')
                elif 610 < data.loc[i]['当前海拔高度'] <= 7620 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] < 480:
                    label.insert(i, '中空盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] >= 480:
                    label.insert(i, '低空高速盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] <= 120 or data.loc[i]['distance_Moscow'] <= 120
                        or data.loc[i]['distance_Lipetsk'] <= 120 or data.loc[i]['distance_Crimea'] <= 120) and \
                        data.loc[i]['当前速度'] < 480:
                    label.insert(i, '低空盘旋')
                elif data.loc[i]['当前海拔高度'] <= 610 and (
                        data.loc[i]['distance_Petersburg'] > 120 or data.loc[i]['distance_Moscow'] > 120
                        or data.loc[i]['distance_Lipetsk'] > 120 or data.loc[i]['distance_Crimea'] > 120) and \
                        data.loc[i]['当前速度'] >= 480:
                    label.insert(i, '低空高速巡航')
                else:
                    print(i + 1)
        data.insert(0, 'Intent', label)
        return data

    def data_coding(self, data_origin):
        # coding
        class_mapping = {'超高空盘旋': 0, '高空盘旋': 1, '中空高速巡航': 2, '中空巡航': 3, '中空盘旋': 4, '低空高速盘旋': 5,
                         '低空盘旋': 6, '低空高速巡航': 7}
        Z1 = data_origin['Intent'].map(class_mapping)
        data_origin = data_origin.drop(['Intent'], axis=1)
        data_origin.insert(0, '意图', Z1)
        # feature selection
        data_origin = data_origin.loc[:, ('意图', '步数', '名称', '所在推演方ID', 'distance_Petersburg', 'distance_Moscow',
                                          'distance_Lipetsk', 'distance_Crimea', '当前经度', '当前纬度', '当前朝向', '当前速度',
                                          '雷达状态', '发射导弹', '当前海拔高度')]
        # one-hot coding
        columns = ['雷达状态', '发射导弹', '所在推演方ID']
        data_origin = pd.get_dummies(data_origin, columns=columns)
        return data_origin

    def data_split(self, train_name, test_name):
        train_data = self.data_Code[self.data_Code['名称'] == train_name]
        test_data = self.data_Code[self.data_Code['名称'] == test_name]
        # normalization
        x_trainNormal = MinMaxScaler().fit_transform(train_data.values[:, 3:])
        x_testNormal = MinMaxScaler().fit_transform(test_data.values[:, 3:])
        x_train, y_train = torch.from_numpy(x_trainNormal), torch.from_numpy(train_data['意图'].values)
        x_test, y_test = torch.from_numpy(x_testNormal), torch.from_numpy(test_data['意图'].values)
        return x_train, y_train, x_test, y_test


class KF_DBN:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.K = 8  # Dimension of latent variable w
        self.n_step = 2000
        self.o = torch.tensor(xTrain).float()
        self.z = torch.tensor(yTrain).long()
        self.n_feature = xTrain.shape[1]  # Dimension of observation data
        self.T = xTrain.shape[0]  # Timeseries length
        self.X_test = xTest
        self.Y_test = yTest
        self.T_test = xTest.shape[0]
        self.IntentionName = ['超高空盘旋', '高空盘旋', '中空高速巡航', '中空巡航', '中空盘旋', '低空高速盘旋', '低空盘旋', '低空高速巡航']

    def parameter_optimal(self, typename, lr=1e-4):
        # parameter_initial
        s = torch.nn.parameter.Parameter(torch.ones(self.T, self.n_feature))
        w = torch.nn.parameter.Parameter(torch.ones(self.K, self.n_feature))
        A = torch.nn.parameter.Parameter(torch.eye(self.n_feature))
        B = torch.nn.parameter.Parameter(torch.ones(self.n_feature))
        C = torch.nn.parameter.Parameter(torch.eye(self.n_feature))
        D = torch.nn.parameter.Parameter(torch.ones(self.n_feature))
        Q = torch.nn.parameter.Parameter(torch.eye(self.n_feature))
        R = torch.nn.parameter.Parameter(torch.eye(self.n_feature))

        loss_list = []
        iteration_list = []
        optimizer = torch.optim.Adam([s, w, A, B, C, D, Q, R], lr)
        pres_loss = 100.0
        for step in range(self.n_step):
            optimizer.zero_grad()

            loss = 0.0
            for t in range(self.T):
                loss = loss + 0.5 * torch.log(torch.det(R)) + 0.5 * torch.mm(
                    torch.mm((self.o[t].unsqueeze(1) - torch.mm(C, s[t].unsqueeze(1)) - D.unsqueeze(1)).t(), R.inverse()),
                    (self.o[t].unsqueeze(1) - torch.mm(C, s[t].unsqueeze(1)) - D.unsqueeze(1))).squeeze(0)

            for t in range(self.T - 1):
                loss = loss + 0.5 * torch.log(torch.det(Q)) + 0.5 * torch.mm(
                    torch.mm((s[t + 1].unsqueeze(1) - torch.mm(A, s[t].unsqueeze(1)) - B.unsqueeze(1)).t(),
                             Q.inverse()),
                    (s[t + 1].unsqueeze(1) - torch.mm(A, s[t].unsqueeze(1)) - B.unsqueeze(1))).squeeze(0)

            for t in range(self.T):
                for k in range(self.K):
                    if self.z[t] == k:
                        base_list = [torch.dot(w[j], s[t]) for j in range(self.K)]
                        max_base = max(base_list)
                        base_list = [base_list[j] - max_base for j in range(self.K)]
                        exp_item = [torch.exp(base_list[j]) for j in range(self.K)]
                        denominator = torch.sum(torch.tensor(exp_item))
                        numerator = exp_item[k]
                        loss = loss - torch.log(torch.div(numerator, denominator))

            loss.backward()  # compute autograd
            optimizer.step()  # take a gradient step

            print('---------------------------------------------------------------------')
            print('step: {}, loss: {}'.format(step, loss.item()))
            print('loss difference: {}'.format(abs(pres_loss - loss.item())))
            pres_loss = loss.item()
            loss_list.append(pres_loss)
            iteration_list.append(step)

        # 参数存储
        torch.save(s, './parameters/s({}).pt'.format(typename)), torch.save(w, './parameters/w({}).pt'.format(typename))
        torch.save(A, './parameters/A({}).pt'.format(typename)), torch.save(B, './parameters/B({}).pt'.format(typename))
        torch.save(C, './parameters/C({}).pt'.format(typename)), torch.save(D, './parameters/D({}).pt'.format(typename))
        torch.save(Q, './parameters/Q({}).pt'.format(typename)), torch.save(R, './parameters/R({}).pt'.format(typename))

        return s, w, A, B, C, D, Q, R, loss_list, iteration_list

    def figure_loss(self, iteration_list, loss_list):
        plt.plot(iteration_list, loss_list)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Loss")
        plt.title("Loss KF-DBN")
        plt.show()

    def kf_predict(self, a, b, q):
        Sigma_pre = torch.eye(self.n_feature).unsqueeze(0) * torch.ones((self.T, 1, 1))
        Mu_pre = torch.zeros(self.n_feature).unsqueeze(0) * torch.ones((self.T, 1))
        for t in range(self.T - 1):
            Mu_pre[t + 1] = (torch.mm(a, Mu_pre[t].unsqueeze(1)) + b.unsqueeze(1)).squeeze()
            Sigma_pre[t + 1] = torch.mm(torch.mm(a, Sigma_pre[t]), a.t()) + q
        return Sigma_pre, Mu_pre

    def kf_update(self, data_o, T, a, b, c, d, q, r):
        Sigma_pre, Mu_pre = self.kf_predict(a, b, q)
        Sigma_update = torch.eye(self.n_feature).unsqueeze(0) * torch.ones((T, 1, 1))
        Mu_update = torch.zeros(self.n_feature).unsqueeze(0) * torch.ones((T, 1))
        K = torch.zeros((self.n_feature, self.n_feature)).unsqueeze(0) * torch.ones((T, 1, 1))
        for t in range(T - 1):
            K[t + 1] = torch.mm(torch.mm(Sigma_pre[t + 1], c.t()),
                                (r + torch.mm(torch.mm(c, Sigma_pre[t + 1]), c.t())).inverse())  # 卡尔曼增益
            Sigma_pre[t + 1] = torch.mm((torch.eye(self.n_feature) - torch.mm(K[t + 1], c)), Sigma_pre[t + 1])
            Mu_pre[t + 1] = (torch.mm((torch.eye(self.n_feature) - torch.mm(K[t + 1], c)),
                                         Mu_pre[t + 1].unsqueeze(1)) - torch.mm(K[t + 1],
                                                                                (d - data_o[t + 1]).unsqueeze(
                                                                                    1))).squeeze()
        return Mu_pre, Sigma_pre

    def train_accuracy(self, a, b, c, d, q, r, W, train_name):
        Mu_update, Sigma_update = self.kf_update(self.o, self.T, a, b, c, d, q, r)
        # computer the intention
        z_kf = torch.randint(0, self.K, (self.T,))
        z_prob = torch.ones((self.K,)) * torch.ones((self.T, 1))
        Intention = [[] for _ in range(self.K)]
        t_list = []
        for t in range(self.T):
            for k in range(self.K):
                time_start = time.time()
                z_prob[t] = torch.nn.functional.softmax(torch.mm(W, Mu_update[t].unsqueeze(1)).squeeze(), dim=0)
                z_kf[t] = torch.argmax(z_prob[t])
                Intention[k].append((z_prob[t][k]).tolist())
                time_end = time.time()  # 记录结束时间
                t_list.append(time_end - time_start)
        # computer the accuracy
        count_list = [0] * self.K
        count = 0
        for i in range(len(self.z)):
            for k in range(self.K):
                if self.z[i] == z_kf[i] == k:
                    count_list[k] += 1
                    count += 1

        # print("TrainRunning Time:{}, '\n' Average TrainTime:{}".format(t_list, np.average(t_list)))
        # print('train_z_kf:{},train_accuracy_kf:{}'.format(z_kf, (count / len(z)) * 100))

        # # figure
        # for k in range(self.K):
        #     plt.plot(range(self.T), Intention[k])
        # plt.legend(self.IntentionName)
        # plt.xlabel("Number of Step")
        # plt.ylabel("Intention")
        # plt.title("model_train MF-DBN ({})".format(train_name))
        # plt.show()

    def accuracy_Test(self, y_test_num, b, c, d, q, r, W, test_name, y_test):
        Mu_update, Sigma_update = self.kf_update(self.X_test.float(), self.T_test, y_test_num, b, c, d, q, r)
        # computer the intention
        z_kf = torch.randint(0, self.K, (self.T_test,))
        z_probs = torch.ones((self.K,)) * torch.ones((self.T_test, 1))
        Intention = [[] for _ in range(self.K)]
        t_list = []
        for t in range(self.T_test):
            for k in range(self.K):
                time_start = time.time()
                z_probs[t] = torch.nn.functional.softmax(torch.mm(W, Mu_update[t].unsqueeze(1)).squeeze(), dim=0)
                z_kf[t] = torch.argmax(z_probs[t])
                Intention[k].append((z_probs[t][k]).tolist())
                time_end = time.time()  # 记录结束时间
                t_list.append(time_end - time_start)
        # computer the accuracy
        acc_num = torch.zeros(self.K,)  # 各类别分类正确的样本数量
        count = 0
        for i in range(len(self.Y_test)):
            for k in range(self.K):
                if self.Y_test[i] == z_kf[i] == k:
                    acc_num[k] += 1
                    count += 1

        target_num, predict_num = torch.zeros(self.K,), torch.zeros(self.K,)   # 各类别真实的样本数量, 各类别预测的样本数量
        y_test_num, z_kf_num = torch.unique(self.Y_test, return_counts=True), torch.unique(z_kf, return_counts=True)
        target_dic, predict_dic = {}, {}
        for i in range(y_test_num[0].shape[0]):
            target_dic[y_test_num[0][i].numpy().tolist()] = y_test_num[1][i].numpy().tolist()
        for k in range(self.K):
            if k in target_dic.keys():
                target_num[k] = target_dic[k]
            else:
                target_num[k] = 0

        for i in range(z_kf_num[0].shape[0]):
            predict_dic[z_kf_num[0][i].numpy().tolist()] = z_kf_num[1][i].numpy().tolist()
        for k in range(self.K):
            if k in predict_dic.keys():
                predict_num[k] = predict_dic[k]
            else:
                predict_num[k] = 0

        Aver_testTime = np.average(t_list)
        perfor_report = classification_report(y_test, z_kf)

        # figure1 （意图识别结果）
        plt.figure()  # 声明一个新画布
        '''陈嘉文加'''
        plot_print = list() 
        for k in range(self.K):
            plot_print.append({'name': self.IntentionName[k], 'data': Intention[k]})
            plt.plot(range(self.T_test), Intention[k])
        plt.legend(self.IntentionName)
        plt.xlabel("Number of Step")
        plt.ylabel("Intention")
        plt.title("model_test MF-DBN ({})".format(test_name))

        return Aver_testTime, perfor_report, target_num, predict_num, acc_num, plt, plot_print, self.IntentionName


def DBN_RES():
    time_start = time.time()  # 记录开始时间
    data_orign = pd.read_excel('./data/正南打击利佩茨克机场数据/data_new.xlsx')  # 导入原始数据
    typeName = '战斗机'
    data_proc = DataProcessing(data_orign, typeName)  # 数据预处理
    data_coding = data_proc.data_Code  # 输入模型的数据
    trainName = 'F-22 科加尔尼西亚 #5'  # 输入训练目标
    testName = 'F-22 科加尔尼西亚 #6'  # 输入测试目标——【目标选择】
    o, z, x_test, y_test = data_proc.data_split(trainName, testName)  # 训练/测试集划分
    model = KF_DBN(o, z, x_test, y_test)  # 模型训练
    # s, w, A, B, C, D, Q, R, loss_list, iteration_list = model.parameter_optimal(typeName)
    # 如果需要展示训练过程，调用上面这个式子；若直接用训练好的模型进行预测，则直接加载学习好的参数，用下面的式子
    s, w, A, B, C, D, Q, R = torch.load('./parameters/s({}).pt'.format(typeName)), \
                             torch.load('./parameters/w({}).pt'.format(typeName)), \
                             torch.load('./parameters/A({}).pt'.format(typeName)), \
                             torch.load('./parameters/B({}).pt'.format(typeName)), \
                             torch.load('./parameters/C({}).pt'.format(typeName)), \
                             torch.load('./parameters/D({}).pt'.format(typeName)), \
                             torch.load('./parameters/Q({}).pt'.format(typeName)), \
                             torch.load('./parameters/R({}).pt'.format(typeName))
    model.train_accuracy(A, B, C, D, Q, R, w, trainName)  # 模型训练结果

    Aver_time, perform_report, target_num, predict_num, acc_num, plt, plot_print, legend = model.accuracy_Test(A, B, C, D, Q, R, w, testName, y_test)   # 模型测试结果
    # plt.show()    # 【可视化展示1】：各模型的意图识别结果图
    # print("Average TestTime:{}".format(Aver_time))
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start 
    # 【可视化展示2-1】：统计信息图+性能指标值输出+各个模型预测精度对比图
    plt.figure()  # 声明一个新画布
    dis = 0
    separate1 = (dis, dis + 0.05, dis + 0.2, dis + 0.35, dis + 0, dis + 0.45, dis + 0.5, dis + 0)
    plt.pie(target_num, autopct='%.01f%%', explode=separate1, radius=1.0, textprops={'fontsize': 8})
    print('饼图数据')
    plt.legend(model.IntentionName, frameon=False, bbox_to_anchor=(-0.35, 0.5), loc=6)
    plt.title("统计信息({})".format(testName), loc="center")
    # plt.show()

    # 【可视化展示2-2】：统计信息图+性能指标值输出+各个模型预测精度对比图
    print("perform_report:'\n' {}".format(perform_report))

    # 【可视化展示2-3】：统计信息图+性能指标值输出+各个模型预测精度对比图
    precision = (acc_num / predict_num * 100).numpy().tolist()
    precision = [0 if math.isnan(x) else x for x in precision]
    precision_dbn = precision
    fig, ax = plt.subplots()
    ax.barh(model.IntentionName, precision, align='center', label='KF-DBN')
    for a, b in zip(model.IntentionName, precision):
        plt.text(b+1, a, "%.2f%%" % b, ha='center', va='center', fontsize=8, color='k')
    ax.set_yticks(model.IntentionName)
    ax.set_xlabel('precision(%)')
    ax.set_title('各个模型预测精度对比图')
    # plt.show()

    # 【可视化展示3】：不同模型在不同冷启动时间下的准确率表
    fig, ax = plt.subplots()
    model.IntentionName.append('整体')
    rowColours = ["#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F0C9C0", "#F2F2F2"]

    column_labels = ["KF-DBN(t=1)", "KF-DBN(t=5)", "KF-DBN(t=10)", "KF-DBN(t=15)", "KF-DBN(t=20)"]
    colColors = ["#377eb8"] * len(column_labels)

    accuracy = (100. * acc_num.sum() / target_num.sum()).numpy()
    precision_list = [precision] * len(column_labels)
    accuracy_list = [accuracy] * len(column_labels)
    precision_data = np.vstack((np.array(precision_list).T, np.array(accuracy_list)))

    ax.axis('off')
    dbn_table = {'colLabels': column_labels, 'rowLabels' :model.IntentionName, 'data':np.round(precision_data, 2)}
    ax.table(cellText=np.round(precision_data, 2), colLabels=column_labels, colColours=colColors, rowColours=rowColours,
             rowLabels=model.IntentionName, cellLoc='center', rowLoc='center', loc="center")
    ax.set_title('不同模型在不同冷启动时间下的准确率表')
    plt.show()

    # 【可视化展示4】：运行时间
    plt.figure()  # 声明一个新画布
    x = ['time_step=1', 'time_step=5', 'time_step=10', 'time_step=15', 'time_step=20']
    y = [Aver_time] * len(x)

    plt.plot(x, y, color='k', linewidth=1, marker='o', markersize=5, label="预测时间")
    for a, b in zip(x, y):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=11, color='k')
    plt.title("各模型的预测时间", color='k')
    print(model.IntentionName, precision)
    return plot_print, legend, model.IntentionName, target_num.tolist(), f"统计信息({testName})", time_sum, x, y, model.IntentionName, precision_dbn, dbn_table