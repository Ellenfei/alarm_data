import random
import numpy as np
from enum import Enum
from data import Data
# DataType = Enum('BANDWIDTH', 'DELAY', 'LOSS_RATE')
class DataType(Enum):
    BANDWIDTH = 0
    DELAY = 1
    LOSS_RATE = 2

class Utils:
    def __init__(self):
        self.BANDWIDTH_EXPECT = 15
        self.BANDWIDTH_VARIANCE = 3
        self.MAX_BANDWIDTH = 100

        self.LOSS_RATE_EXPECT = 0.1
        self.LOSS_RATE_VARIANCE = 0.05
        self.MAX_LOSS_RATE = 0.3

        self.DELAY_VARIANCE = 1.5
        self.DELAY_EXCEPT = 15
        self.MAX_DELAY = 50

    def generateRandomNumber(self, dataType, expect, variance, normal):
        if normal:
            randomNumber = random.gauss(expect, variance)
        else:
            magnification = 1
            if dataType==DataType(0):
                magnification = self.MAX_BANDWIDTH
            elif dataType==DataType(1):
                magnification = self.MAX_DELAY
            elif dataType==DataType(2):
                magnification = self.MAX_LOSS_RATE

            randomNumber = random.random()*magnification

        return randomNumber
    def generateNormalData(self, dataType, expect, variance):
        return self.generateRandomNumber(dataType, expect, variance, True)

    def generateAnormalData(self, dataType):
        return self.generateRandomNumber(dataType, 0, 0, False)

    def generateVertex(self, normal):
        vertexPair = []
        if normal:
            srcVertex = random.randint(1, 10)
            desVertex = random.randint(20, 24)
        else:
            srcVertex = random.randint(1, 30)
            desVertex = random.randint(1, 30)
        vertexPair.append(srcVertex)
        vertexPair.append(desVertex)
        return vertexPair

    def generateData(self, normal):
        vertexPair = self.generateVertex(normal)
        srcVertex = vertexPair[0]
        desVertex = vertexPair[1]
        if normal:
            bandwidth = self.generateNormalData(DataType(0), self.BANDWIDTH_EXPECT,
                                                  self.BANDWIDTH_VARIANCE)
            delay = self.generateNormalData(DataType(1), self.DELAY_EXCEPT,
                                              self.DELAY_VARIANCE)
            loss_rate = self.generateNormalData(DataType(2), self.LOSS_RATE_EXPECT,
                                                  self.LOSS_RATE_VARIANCE)
            flag = 1
        else:
            bandwidth = self.generateAnormalData(DataType(0))
            delay = self.generateAnormalData(DataType(1))
            loss_rate = self.generateAnormalData(DataType(2))
            flag = 0

        data = Data(nodeS=srcVertex, nodeD=desVertex, bandwidth=bandwidth,
                    delay=delay, loss=loss_rate, flag=flag)
        # listData = []
        # listData.append([data.nodeS, data.nodeD, data.bandwidth,
        #                 data.delay, loss_rate, data.flag])
        # toMatrix = np.mat(listData)
        # print(toMatrix)
        return data
# test = Utils()
# for i in range(10):
#     test.generateData(True)
# for i in range(10):
#     test.generateData(False)
