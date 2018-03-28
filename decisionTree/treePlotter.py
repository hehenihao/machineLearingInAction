import matplotlib.pyplot as plt

class Plotter(object):
    def __init__(self):
        # 定义文本框 和 箭头格式 【 sawtooth 波浪方框, round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅，没错是变浅】
        self.decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")


    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                                textcoords='axes fraction', va="center", ha="center", bbox=nodeType,
                                arrowprops=self.arrow_args)


    def getLeavesNum(self, myTree):
        numLeaves = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        # 根节点开始遍历
        for key in secondDict.keys():
            if type(secondDict[key]) is dict:
                numLeaves += self.getLeavesNum(secondDict[key])
            else:
                numLeaves += 1
        return numLeaves


    def getTreeDepth(self, myTree):
        treeDepth = 0
        thisDepth = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        # 根节点开始遍历
        for key in secondDict.keys():
            if type(secondDict[key]) is dict:
                thisDepth += self.getTreeDepth(secondDict[key])
            else:
                thisDepth += 1
            if thisDepth > treeDepth:
                treeDepth = thisDepth
        return treeDepth

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


    def plotTree(self, myTree, parentPt, nodeTxt):
        # 获取叶子节点的数量
        numLeafs = self.getLeavesNum(myTree)
        # 获取树的深度
        # depth = getTreeDepth(myTree)

        # 找出第1个中心点的位置，然后与 parentPt定点进行划线
        # x坐标为 (numLeafs-1.)/totalW/2+1./totalW，化简如下
        cntrPt = (self.xOff + (1.0 + float(numLeafs)) / 2.0 / self.totalW, self.yOff)
        # print cntrPt
        # 并打印输入对应的文字
        self.plotMidText(cntrPt, parentPt, nodeTxt)

        firstStr = list(myTree.keys())[0]
        # 可视化Node分支点；第一次调用plotTree时，cntrPt与parentPt相同
        self.plotNode(firstStr, cntrPt, parentPt, self.decisionNode)
        # 根节点的值
        secondDict = myTree[firstStr]
        # y值 = 最高点-层数的高度[第二个节点位置]；1.0相当于树的高度
        self.yOff = self.yOff - 1.0 / (0.8*self.totalD)
        for key in secondDict.keys():
            # 判断该节点是否是Node节点
            if type(secondDict[key]) is dict:
                # 如果是就递归调用[recursion]
                self.plotTree(secondDict[key], cntrPt, str(key))
            else:
                # 如果不是，就在原来节点一半的地方找到节点的坐标
                self.xOff = self.xOff + 1.0 / self.totalW
                # 可视化该节点位置
                self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, self.leafNode)
                # 并打印输入对应的文字
                self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0 / self.totalD


    def createPlot(self, inTree):
        # 创建一个figure的模版
        fig = plt.figure(1, facecolor='green')
        fig.clf()

        axprops = dict(xticks=[], yticks=[])
        # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
        self.ax1 = plt.subplot(111, frameon=False, **axprops)

        self.totalW = float(self.getLeavesNum(inTree))
        self.totalD = float(self.getTreeDepth(inTree))
        # 半个节点的长度；xOff表示当前plotTree未遍历到的最左的叶节点的左边一个叶节点的x坐标
        # 所有叶节点中，最左的叶节点的x坐标是0.5/totalW（因为totalW个叶节点在x轴方向是平均分布在[0, 1]区间上的）
        # 因此，xOff的初始值应该是 0.5/totalW-相邻两个叶节点的x轴方向距离
        self.xOff = -0.5 / self.totalW
        # 根节点的y坐标为1.0，树的最低点y坐标为0
        self.yOff = 1.0
        # 第二个参数是根节点的坐标
        self.plotTree(inTree, (0.5, 1.0), '')
        plt.show()


if __name__ == '__main__':
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]

    tp = Plotter()

    print(tp.getLeavesNum(listOfTrees[1]))
    print(tp.getTreeDepth(listOfTrees[1]))

    tp.createPlot(listOfTrees[1])