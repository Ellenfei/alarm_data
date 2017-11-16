import tensorflow as tf

with tf.name_scope('graph') as scope:
    m1 = tf.constant([[3., 3.]], name='matrix1')
    m2 = tf.constant([[2.], [2.]], name='matrix2')
    product = tf.matmul(m1, m2, name='product')

sess = tf.Session()

writer = tf.summary.FileWriter('logs/', sess.graph)

init = tf.global_variables_initializer()

sess.run(init)

