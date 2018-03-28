from Algorithm import KNNAlgorithm
import numpy as np
import matplotlib as plt


def readTxtData(fileName:str):
    with open(fileName, mode='r') as file:
        lines = file.readlines()
        lineNum = len(lines)
        resultMat = np.zeros([lineNum, 3])
        classLabelVector = []
        for i in range(lineNum):
            line = lines[i].strip()
            data = line.split()
            resultMat[i, :] = data[:3]
            classLabelVector.append(int(data[-1]))
        return resultMat, classLabelVector


def normalization(data):
    minColunm = data.min(0)
    maxColunm = data.max(0)
    range = maxColunm - minColunm
    normData = np.zeros(data.shape)
    m = data.shape[0]
    normData = data - np.tile(minColunm, (m, 1))
    normData = normData / np.tile(range, (m, 1))
    return  normData, minColunm, maxColunm


def visulization(data):
    pass

if __name__ == '__main__':
    ratio = 0.10
    knn = KNNAlgorithm()
    dataMat, labels = readTxtData('datingTestSet2.txt')
    knn.setData(dataMat, labels, 3)
    normData, min, max = normalization(dataMat)
    clounms = normData.shape[0]
    testMatNum = int(clounms *ratio)
    errCnt = 0.0
    for i in range(testMatNum):
        result = knn.classify(normData[i,:])
        if result is not labels[i]:
            errCnt += 1.0
    print('the total error rate is {0}'.format(errCnt/clounms))

