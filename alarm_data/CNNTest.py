import tensorflow as tf
import numpy as np
import data_process

# 1. 数据处理
input_data = data_process.datas
# 改变维度
input_data1 = np.resize(input_data, (1125, 5, 1))
values1 = tf.cast(input_data1, tf.float32)
# 卷积核
conv_filter1 = tf.Variable(np.random.rand(2, 1, 3), dtype=np.float32)
# 卷积层
conv_out1 = tf.nn.conv1d(values1, conv_filter1, stride=1, padding='SAME')
print('conv1d:', conv_out1)