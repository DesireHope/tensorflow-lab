import tensorflow as tf

# 声明w1、w2两个变量。
# 通过设置随机数种子来保证每次运行的结果相同
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

tf.assign(w1, w2, validate_shape=False)