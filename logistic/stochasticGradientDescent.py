import numpy as np
import os
import random

'''
随机梯度上升：
和批量梯度下降法原理类似，区别在与求梯度时没有用所
有的m个样本的数据，而是仅仅选取一个样本j来求梯度
'''


class StochasticGradientAscent(object):
    def __init__(self, dataArrayIn, labelsArrayIn):
        self.dataArrayIn = dataArrayIn
        self.labels = labelsArrayIn


    def sigmoid(self, xIn):
        return 1 / (1 + np.exp(-xIn))


    def run(self):
        data = np.mat(self.dataArrayIn)
        labels = np.mat(self.labels).transpose()
        m, n = np.shape(data)
        alpha = 0.001
        weights = np.ones((n,1))
        for k in range(m):
            h = self.sigmoid(np.sum(data[k] * weights))  # 梯度上升矢量化公式
            error = labels[k] - h
            weights = weights + alpha  * data[k].transpose() * error
        return weights.tolist()


    def improvedRun(self, numIter=150):
        data = np.mat(self.dataArrayIn)
        labels = np.mat(self.labels).transpose()
        m, n = np.shape(data)
        weights = np.ones((n,1))
        weights_array = np.array([])
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
                randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
                h = self.sigmoid(sum(data[randIndex] * weights))  # 选择随机选取的一个样本，计算h
                error = labels[randIndex] - h  # 计算误差
                weights = weights + alpha * data[randIndex].transpose() * error # 更新回归系数
                del (dataIndex[randIndex])  # 删除已经使用的样本
                weights_array = np.append(weights_array, weights)
        weights_array = weights_array.reshape(numIter*m , n)
        return weights.tolist(), weights_array
