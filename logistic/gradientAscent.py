import numpy as np
import os

'''
批量梯度上升：
梯度下降法最常用的形式，具体做法也就
是在更新参数时使用所有的样本来进行更新
'''

class GradientAscent(object):
    def __init__(self, dataArrayIn, labelsArrayIn):
        self.dataArrayIn = dataArrayIn
        self.labels = labelsArrayIn


    def sigmoid(self, xIn):
        return 1/(1 + np.exp(-xIn))


    def run(self):
        data = np.mat(self.dataArrayIn)
        labels = np.mat(self.labels).transpose()
        m, n = np.shape(data)
        alpha = 0.01
        maxCycles = 500
        weights_array = np.array([])
        weights = np.ones((n,1), dtype=np.float64)
        for k in range(maxCycles):
            h = self.sigmoid(data*weights)  # 梯度上升矢量化公式
            error = labels - h
            weights = weights + alpha * data.transpose() * error
            weights_array = np.append(weights_array, weights)
        weights_array = weights_array.reshape(maxCycles, n)
        return weights.tolist(), weights_array
