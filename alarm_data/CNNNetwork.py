import tensorflow as tf
import numpy as np
import data_process
from generateData import Utils
# 定义函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

# def max_pooling_22(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 1. 数据导入
# data = data_process.datas
# input_data1 = np.resize(input_datas, (1125, 5, 1))
# datas = tf.cast(data, tf.float32)
# xs = datas[:, 0:5]
# ys = datas[:, 5]
# sess = tf.InteractiveSession()
# xs_ = sess.run(xs)
# ys_ = sess.run(ys)

datas = []
test1 = Utils()
for i in range(500):
    datas.append(test1.generateData(False))
for i in range(10000):
    datas.append(test1.generateData(True))
# input_data = np.array(datas)
listData = []
for i in range(len(datas)):
    listData.append([datas[i].nodeS, datas[i].nodeD, datas[i].bandwidth, datas[i].delay,
                    datas[i].loss, datas[i].flag])
toMatrix = np.mat(listData)
xs_ = toMatrix[:, 0:5]
ys_ = toMatrix[:, 5]
sess = tf.InteractiveSession()
# xs_ = sess.run(xs)
# ys_ = sess.run(ys)

# y_actual = tf.random_normal_initializer

# 2.第一层卷积
# 参数
W_conv1 = weight_variable([2, 1, 10])
b_conv1 = bias_variables([10])
x = tf.placeholder(tf.float32, [None, 5])
x_data = tf.reshape(x, [-1, 5, 1])
h_conv1 = tf.nn.relu(conv1d(x_data, W_conv1) + b_conv1)

# 3.第二层卷积
W_conv2 = weight_variable([2, 10, 8])
b_conv2 = bias_variables([8])
# 激活函数
h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)
h_conv3 = tf.reshape(h_conv2, [-1, 40])

# 4.全连接层
W_fc1 = weight_variable([40, 10])
b_fc1 = bias_variables([10])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3, W_fc1) + b_fc1)

# 5.dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 6.输出层(一个输出)
W_fc2 = weight_variable([10, 1])
b_fc2 = bias_variables([1])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_predict))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 0), tf.argmax(y_, 0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={x: xs_, y_: ys_, keep_prob: 0.5})
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: xs_, y_: ys_, keep_prob: 1.0})
        print('step %d, training accuracy %g' %(i, train_accuracy))
    # sess.run(train_step, feed_dict={x: xs_, y_: ys_, keep_prob : 0.5})
print('test accuracy %g' % (accuracy.eval(feed_dict={x: xs_, y_: ys_, keep_prob: 1.0})))