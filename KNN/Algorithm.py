import numpy as np
import operator
import matplotlib as plot


def create_data():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.2, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


class KNNAlgorithm(object):
    def __init__(self):
        print('new knn')

    def setData(self, dataSet, labels, k):
        self.dataSet = dataSet
        self.labels = labels
        self.k = k

    def classify(self, data):
        """
        计算data与数据集中点的距离，距离递增排序，选取距离最小的K个点；
        确定K个点所在类别出现频率
        :param data:
        :return: k个点中出现频率最高的类别
        """
        dataSetSize = self.dataSet.shape[0]
        diffMatrix = np.tile(data, [dataSetSize, 1]) - self.dataSet
        sqDiffMatrix = diffMatrix ** 2
        distance = sqDiffMatrix.sum(axis=1)**0.5
        sortedDistanceIndex = np.argsort(distance)
        classCount = {}
        for i in range(self.k):
            label = self.labels[sortedDistanceIndex[i]]
            classCount[label] = classCount.get(label,0) + 1
        # print(classCount.items())
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = create_data()
    knn = KNNAlgorithm()
    knn.setData(group, labels, 3)
    res = knn.classify([0, 0])
    print(res)
