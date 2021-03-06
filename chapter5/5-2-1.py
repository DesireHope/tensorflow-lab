import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# MNIST数据集相关的常数
INPUT_NODE = 784    # 输入层节点数
OUTPUT_NODE = 10    # 输出层节点数
# 配置神经网络的参数
LAYER1_NODE = 500   # 隐藏层节点数
BATCH_SIZE = 100    # 一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8      # 基础学习率
LEARNING_RATE_DECAY = 0.99    # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000        # 训练轮数
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率


# 给定神经网络的输入和所有参数
# 这个只能适用三层的全连接神经网络
def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里是用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 先使用avg_class.average函数来计算得出变量的滑动平均值，
        # 然后再计算相应的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) \
               + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 计算在当前参数下神经网络前向传播的结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量
    # 将该变量定义为不可训练变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 交叉熵为损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.arg_max(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
                 .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 检验使用了滑动平均模型的前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))
            # 产生这一轮使用的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs,
                                          y_: ys})

        # 训练结束后，在测试数据上检测
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average"
              "model is %g" % (TRAINING_STEPS, test_acc))


def inference_new(input_tensor, reuse=False):
     with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                     initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
     with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                     initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
     return layer2


def train_new(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    y = inference_new(x)
    # 定义存储训练轮数的变量
    # 将该变量定义为不可训练变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference_new(x, True)

    # 交叉熵为损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.arg_max(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失
    with tf.variable_scope("layer1", reuse=True):
        weights1 = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE])
    with tf.variable_scope("layer2", reuse=True):
        weights2 = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE])
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 检验使用了滑动平均模型的前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))
            # 产生这一轮使用的训练数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs,
                                          y_: ys})

        # 训练结束后，在测试数据上检测
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average"
              "model is %g" % (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("/Users/user/PycharmProjects/tensorflow-second/chapter5/MNIST_data", one_hot=True)
    train_new(mnist)


if __name__ == '__main__':
    tf.app.run()
