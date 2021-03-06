import sys
import numpy as np
import tensorflow as tf
import thrift
sys.path.append('gen-py')

from generateData import Utils
from data_format import DataService
from data_format.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class DataServiceHandler:
    def __init__(self):
        n_input = 5
        n_hidden = 6
        n_output = 1
        self.parameters = []
        self.parametersData = []
        self.parametersData.append([tf.Variable(tf.random_normal([n_input, n_hidden])),
                                    tf.Variable(tf.zeros([1, n_hidden]) + 0.1),
                                    tf.Variable(tf.random_normal([n_hidden, n_output])),
                                    tf.Variable(tf.zeros([1, n_output]) + 0.1)])

    # def ping(self):
    #     print('ping()')

    def buildModel(self, type, datas):
        if type == BussinessType.VIDEO:
            print('视频业务参数')
        elif type == BussinessType.ECOMMERCE:
            print('电子商务业务参数')
        elif type == BussinessType.EMAIL:
            print('电子邮件业务参数')
        elif type == BussinessType.FILE_TRANSFER:
            print('文件传输业务参数')

        # 1.将数据集转换为numpy矩阵
        listData = []
        for i in range(len(datas)):
            listData.append([datas[i].nodeS, datas[i].nodeD, datas[i].bandwidth, datas[i].delay,
                             datas[i].loss, datas[i].flag])
        toMatrix = np.mat(listData)
        # print(toMatrix)
        x_data = toMatrix[:, 0:5]
        y_data = toMatrix[:, 5]

        # 2.定义网络参数(输入层/隐藏层/输出层神经元个数)
        n_input = 5
        n_hidden = 6
        n_output = 1
        learning_rate = 0.01

        # 3.定义节点准备接收数据
        xs = tf.placeholder(tf.float32, [None, n_input])
        ys = tf.placeholder(tf.float32, [None, n_output])

        # 4.定义神经层（输入层/隐藏层/输出层）
        layer1 = self.add_layer1(xs, n_input, n_hidden, activation_function=tf.nn.relu)
        layer2 = self.add_layer2(layer1, n_hidden, n_output, activation_function=tf.nn.sigmoid)

        # 5.损失函数
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -layer2), reduction_indices=[1]))

        # 6.损失函数最小化，学习率为0.01
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 7.初始化变量
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


        # self.parametersData = []
        # self.parametersData.append([tf.Variable(tf.random_normal([n_input, n_hidden])),
        #                             tf.Variable(tf.zeros([1, n_hidden]) + 0.1),
        #                             tf.Variable(tf.random_normal([n_input, n_hidden])),
        #                             tf.Variable(tf.zeros([1, n_hidden]) + 0.1)])
        # print(self.parametersData[len(self.parametersData-1)][0])

        # 8.设置迭代次数，训练模型
        for i in range(10000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            # print(sess.run(self.weights_one))
            self.parameters.append(Parameters(w1=sess.run(self.weights_one), b1=sess.run(self.biases_one),
                                              w2=sess.run(self.weights_two), b2=sess.run(self.biases_two)))
            self.parametersData.append([np.mat(self.parameters[i].w1), np.mat(self.parameters[i].b1),
                                   np.mat(self.parameters[i].w2), np.mat(self.parameters[i].b2)])
            # print(sess.run(layer2, feed_dict={xs: x_data, ys: y_data}))
            if i % 50 == 0:
                # print(sess.run(layer2, feed_dict={xs: x_data, ys: y_data}))
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                # print(self.parametersData)

        return self.parametersData

    # BP神经层函数
    def add_layer1(self, input, in_size, out_size, activation_function=None):
        # 权重和偏置量
        self.weights_one = tf.Variable(tf.random_normal([in_size, out_size]))
        self.biases_one = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # self.weights_one = self.parametersData[len(self.parametersData)-1][0]
        # self.biases_one = self.parametersData[len(self.parametersData)-1][1]
        # 输出
        out = tf.matmul(input, self.weights_one) + self.biases_one
        # 激活函数
        if activation_function is None:
            outputs = out
        else:
            outputs = activation_function(out)
        return outputs
    def add_layer2(self, input, in_size, out_size, activation_function=None):
        # 权重和偏置量
        # self.weights_two = self.parametersData[0][2]
        self.weights_two = tf.Variable(tf.random_normal([in_size, out_size]))
        # self.biases_two = self.parametersData[0][3]
        self.biases_two = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # 输出
        out = tf.matmul(input, self.weights_two) + self.biases_two
        # 激活函数
        if activation_function is None:
            outputs = out
        else:
            outputs = activation_function(out)
        return outputs

type = 1

datas = []
test1 = Utils()

for i in range(100):
    datas.append(test1.generateData(True))
    datas.append(test1.generateData(False))
# 定义list接收数据,将数据转换为list


test = DataServiceHandler()
parameters = test.buildModel(type, datas)

# handler = DataServiceHandler()
# processor = DataService.Processor(handler)
# transport = TSocket.TServerSocket(port=9090)
# tfactory = TTransport.TBufferedTransportFactory()
# pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#
# server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
#
# print('Starting the server.......')
# server.serve()
# print('done')


