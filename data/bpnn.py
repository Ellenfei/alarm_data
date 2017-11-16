import tensorflow as tf
# BP神经层函数
def add_layer1(input, in_size, out_size, activation_function=None):
    # 权重和偏置量
    self.weights_one = tf.Variable(tf.random_normal([in_size, out_size]))
    self.biases_one = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 输出
    out = tf.matmul(input, self.weights_one) + self.biases_one
    # 激活函数
    if activation_function is None:
        outputs = out
    else:
        outputs = activation_function(out)
    return outputs

def add_layer2(input, in_size, out_size, activation_function=None):
    # 权重和偏置量
    self.weights_two = tf.Variable(tf.random_normal([in_size, out_size]))
    self.biases_two = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 输出
    out = tf.matmul(input, self.weights_two) + self.biases_two
    # 激活函数
    if activation_function is None:
        outputs = out
    else:
        outputs = activation_function(out)
    return outputs
