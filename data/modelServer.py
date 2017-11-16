import sys
import numpy as np
import tensorflow as tf
import thrift
sys.path.append('gen-py')

from data_format import DataService
from data_format.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from bpnn import add_layer1, add_layer2

class DataServiceHandler:
    def __init__(self):
        self.parameters = []

    def ping(self):
        print('ping()')

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
        layer1 = add_layer1(xs, n_input, n_hidden, activation_function=tf.nn.relu)
        layer2 = add_layer2(layer1, n_hidden, n_output, activation_function=None)

        # 5.损失函数
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys -layer2), reduction_indices=[1]))

        # 6.损失函数最小化，学习率为0.01
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # 7.初始化变量
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        parametersData = []

        # 8.设置迭代次数，训练模型
        for i in range(100):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            self.parameters.append(Parameters(w1=sess.run(self.weights_one), b1=sess.run(self.biases_one),
                                              w2=sess.run(self.weights_two), b2=sess.run(self.biases_two)))
            parametersData.append([np.mat(self.parameters[0].w1), np.mat(self.parameters[0].b1),
                                   np.mat(self.parameters[0].w2), np.mat(self.parameters[0].b2)])
            if i % 2 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                #print(parametersData)
                print(sess.run(self.weights_one))
                #print('hello')
                #print(sess.run(self.biases_two))

        #print(parametersData)
        return parametersData
type = 1
datas = [Data(nodeS=2,nodeD=13,bandwidth=12.275,delay=12.6841,loss=0.0508,flag=1),
         Data(nodeS=5,nodeD=10,bandwidth=14.275,delay=11.6841,loss=0.0908,flag=1),
         Data(nodeS=3,nodeD=10,bandwidth=14.875,delay=14.1382,loss=0.0821,flag=1),
         Data(nodeS=5,nodeD=10,bandwidth=15.075,delay=14.241,loss=0.0808,flag=1),
         Data(nodeS=1,nodeD=11,bandwidth=13.275,delay=11.6841,loss=0.0208,flag=1)]
test = DataServiceHandler()
parameters = test.buildModel(type, datas)
