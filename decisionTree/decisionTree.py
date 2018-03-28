import numpy as np
import math
from treePlotter import Plotter
import json


def calShannonEnt(dataSet):
    '''
    熵：表示随机变量不确定性，即混乱程度的量化指标。
    熵越大，不确定性越大，越无序；越小，确定性越大，越有序。
    :param dataSet: 待计算熵的数据集
    :return:
    '''
    numEntries = len(dataSet)
    labelDict = {}
    for data in dataSet:
        currentLabel = data[-1]
        # if currentLabel not in labelDict.keys():
        #     labelDict[currentLabel] = 0
        # labelDict[currentLabel] += 1
        labelDict[currentLabel] = labelDict.get(currentLabel,0) + 1
    shannonEnt = 0.0
    for key in labelDict.keys():
        pro = float(labelDict[key]) / numEntries
        shannonEnt -= pro * math.log(pro, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    对原始数据集按照指定特征进行划分
    :param dataSet: 原始数据
    :param axis: 数据划分的特征
    :param value: 特征的值
    :return:
    """
    result = []
    for data in dataSet:
        if data[axis] == value:
            reducedFeatureVec = data[:axis]
            reducedFeatureVec.extend(data[axis + 1:])
            result.append(reducedFeatureVec)
    return result


def chooseBestDataSplit(dataSet):
    """
    寻找划分数据集的最好特征（划分之后信息熵最小，也就是信息增益最大的特征）
    :param dataSet:
    :return:
    """
    featureNum = len(dataSet[0]) - 1
    baseEntroy = calShannonEnt(dataSet)  # 原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(featureNum):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntroy = 0.0
        for value in uniqueVals:
            subData = splitDataSet(dataSet, i, value)
            prob = len(subData) / float(len(dataSet))
            newEntroy += prob * calShannonEnt(subData)
        infoGain = baseEntroy - newEntroy  # 寻求信息熵最小
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def majorityCnt(classList):
    """

    :param classList: 分类标签列表
    :return: 数量最多的分类
    """
    classCnt = {}
    for value in classList:
        if value not in classCnt.keys():
            classCnt[value] = 0
        classCnt[value] += 1
    sortedClassCnt = sorted(classCnt.items(), key=lambda item:item[1], reverse=True)
    return sortedClassCnt[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [data[-1] for data in dataSet]
    # 类别相同，无需继续细分
    if classList.count(classList[0]) == len(classList):
        return  classList[0]
    # 只有一列数据时候，直接返回数量最多分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestDataSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}} # 用于记录树结果
    del(labels[bestFeature])
    featureValues = [value[bestFeature] for value in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet\
                                                         (dataSet, bestFeature, value) ,subLables)
    return myTree


def classify(inputTree, featLables, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLables.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) is dict:
                classify(secondDict, featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def saveTree(inputTree, fileName):
    jsObj = json.dumps(inputTree)
    with open(fileName, mode='w') as file:
        file.write(jsObj)


def readInTree(fileName):
    with open(fileName, mode='r') as file:
        data = file.readline()
        tree = json.loads(data)
    return tree

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


if __name__ == '__main__':
    plot = Plotter()

    # dataset, labels = createDataSet()
    # splitData = splitDataSet(dataset, 0, 0)
    # print(splitData)
    #
    # bestFeature = chooseBestDataSplit(dataset)
    # print(bestFeature)
    #
    # tree = createTree(dataset, labels)
    # print(tree)
    #
    # plot.createPlot(tree)

    with open('lenses.txt',mode='r') as file:
        lines = file.readlines()
        data = [line.strip().split('\t') for line in lines]

    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    #tree = createTree(data, labels)
    #plot.createPlot(tree)
    #saveTree(tree, 'tree.json')
    plot.createPlot(readInTree('tree.json'))




