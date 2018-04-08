import numpy as np
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from gradientAscent import GradientAscent
from stochasticGradientDescent import  StochasticGradientAscent as SGA


os.chdir(os.path.abspath(os.path.dirname(__file__)))


def loadData():
    dataMat = []
    labelMat = []
    with open(r'data\TestSet.txt','r') as file:
        lines = file.readlines()
        for line in lines:
            words = line.strip().split()
            dataMat.append([1.0, float(words[0]), float(words[1])])
            labelMat.append(int(words[2]))
    return dataMat, labelMat


def plotData(dataset, labels):
    index = 0
    class0Xcoord = []
    class0Ycoord = []
    class1Xcoord = []
    class1Ycoord = []
    for label in labels:
        if label == 1:
            class1Xcoord.append(dataset[index][1])
            class1Ycoord.append(dataset[index][2])
        else:
            class0Xcoord.append(dataset[index][1])
            class0Ycoord.append(dataset[index][2])
        index += 1
    fig = plt.figure('data')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(class0Xcoord, class0Ycoord, c='red', marker='s')
    ax.scatter(class1Xcoord, class1Ycoord, c='green')
    plt.show()


def plotBestFit(name, dataset, labels, plotweights):
    index = 0
    class0Xcoord = []
    class0Ycoord = []
    class1Xcoord = []
    class1Ycoord = []
    for label in labels:
        if label == 1:
            class1Xcoord.append(dataset[index][1])
            class1Ycoord.append(dataset[index][2])
        else:
            class0Xcoord.append(dataset[index][1])
            class0Ycoord.append(dataset[index][2])
        index += 1
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(class0Xcoord, class0Ycoord, c='red', marker='s')
    ax.scatter(class1Xcoord, class1Ycoord, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = -(plotweights[0] + plotweights[1]*x)/plotweights[2]
    plt.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()


def plotWeight(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))

    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

def plotWeight1(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    fig = plt.figure('Weight1')
    x1 = np.arange(0, len(weights_array1), 1)
    ax1 = fig.add_subplot(211)
    # 绘制w0与迭代次数的关系
    l1 = ax1.plot(x1, weights_array1[:, 0],
             color='red',
             linestyle='-.',  # 线条样式
             label='W0',  #标签
             )
    # 绘制w1与迭代次数的关系
    l2 = ax1.plot(x1, weights_array1[:, 1],
             color='b',
             linestyle='-.',  # 线条样式
             label='W1',  #标签
             )
    # 绘制w2与迭代次数的关系
    l3 = ax1.plot(x1, weights_array1[:, 2],
             color='c',
             linestyle='-',  # 线条样式
             label='W2',  #标签
             )
    ax1.legend(loc='upper left', frameon=False)

    ax2 = fig.add_subplot(212)
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    ax2.plot(x2, weights_array2[:, 0],
             color='red',
             linestyle='-.',  # 线条样式
             label='W0',  #标签
             )
    # 绘制w1与迭代次数的关系
    ax2.plot(x2, weights_array2[:, 1],
             color='b',
             linestyle='-.',  # 线条样式
             label='W1',  #标签
             )
    # 绘制w2与迭代次数的关系
    ax2.plot(x2, weights_array2[:, 2],
             color='c',
             linestyle='-',  # 线条样式
             label='W2',  #标签
             )
    ax2.legend(loc='upper left', frameon=False)


    plt.xlabel('Time')
    plt.ylabel('W')


    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadData()
    # plotData(dataMat, labelMat)
    ga = GradientAscent(dataMat, labelMat)
    weights, weight_arrayGA = ga.run()
    plotBestFit('ga', dataMat, labelMat, weights)

    sga = SGA(dataMat, labelMat)
    weights2 = sga.run()
    weights3, weight_arraySGA = sga.improvedRun(200)
    plotBestFit('sga', dataMat, labelMat, weights2)
    plotBestFit('sgaIm', dataMat, labelMat, weights3)

    plotWeight1(weight_arrayGA, weight_arraySGA)
