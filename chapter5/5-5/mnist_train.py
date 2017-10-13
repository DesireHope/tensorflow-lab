import tensorflow as tf

import os
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100    # 一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8      # 基础学习率
LEARNING_RATE_DECAY = 0.99    # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000        # 训练轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/Users/user/PycharmProjects/tensorflow-second/chapter5/5-5/model"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用函数定义前向传播过程
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.arg_max(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 训练时不再测试在验证数据集上的表现，会有一个单独的过程来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            # 保存模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/Users/user/PycharmProjects/tensorflow-second/chapter5/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()