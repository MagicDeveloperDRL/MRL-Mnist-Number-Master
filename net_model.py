'''''''''
@file: net_model.py
@author: MRL Liu
@time: 2021/4/14 22:52
@env: Python,Numpy
@desc: 本模块提供定义模型、训练模型、评估模型的方法
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datahelpter
# 神经网络参数
INPUT_NODE=784 # 输入维度
OUTPUT_NODE=10 # 输出维度
LAYER1_NODE=500 # 第一层维度
# 学习率
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZTION_RATE=0.0001
# 训练参数
TRAINING_STEPS=3000
BATCH_SIZE=100
MOVING_AVERAGE_DECAY=0.99
#模型保存的路径和文件名
DATASET_SAVE_PATH='./mnist/' # 数据集保存路径
MODEL_SAVE_PATH='./models/'
LOGS_SAVE_PATH='./logs/'
MODEL_NAME='model.ckpt'


class net_model(object):
    def __init__(self,num_examples):
        self.n_input =INPUT_NODE
        self.n_layer_1 =LAYER1_NODE
        self.n_output = OUTPUT_NODE
        # 滑动平均模型相关参数
        self.moving_average_decay = MOVING_AVERAGE_DECAY
        # 训练相关参数
        self.training_step = TRAINING_STEPS
        self.batch_size = BATCH_SIZE
        # 学习率的相关参数（指数衰减法）
        self.learn_rate_base = LEARNING_RATE_BASE # 学习率初始值
        self.learn_rate_decay = LEARNING_RATE_DECAY # 学习率衰减率
        self.learn_rate_num = num_examples / self.batch_size, # 学习率衰减次数
        # 相关路径
        self.model_save_path = MODEL_SAVE_PATH
        self.logs_save_path = LOGS_SAVE_PATH
        self.model_name = MODEL_NAME
    def test_accuracy(self,_images,_labels):
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [None, self.n_input], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, self.n_output], name='y-input')
            validate_feed = {x: _images, y_: _labels}
            y = self._define_net(x, regularizer__function=None, is_historgram=False)
            # 计算准确率
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # 滑动平均模型变量
            variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)  # 定义一个滑动平均类
            variables_to_restore = variables_averages.variables_to_restore()  # 生成变量重命名的列表
            # 创建加载变量重命名后的保存器
            saver = tf.train.Saver(variables_to_restore)
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.model_save_path)  # 获取ckpt的模型文件的路径
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)  # ckpt.model_checkpoint_path保存了最新次数的模型文件路径
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  # 获取训练次数
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)  # 运行计算图，获取准确率
                    print('After %s training step(s), accuracy on validation  is %g.' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
    def test_random(self,_images,_labels):
        # 随机挑选9个照片
        random_indices = random.sample(range(len(_images)), min(len(_images), 9))
        images, labels = zip(*[(_images[i], _labels[i]) for i in random_indices])
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [None, self.n_input], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, self.n_output], name='y-input')
            validate_feed = {x: images,y_: labels}
            y = self._define_net(x, regularizer__function=None, is_historgram=False)
            # 滑动平均变量
            variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)  # 定义一个滑动平均类
            variables_to_restore = variables_averages.variables_to_restore()  # 生成变量重命名的列表
            # 创建加载变量重命名后的保存器
            saver = tf.train.Saver(variables_to_restore)
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.model_save_path)  # 获取ckpt的模型文件的路径
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复模型参数
                    pred = sess.run(y, feed_dict=validate_feed)  # 运行计算图，获取准确率
                    datahelpter.plot_images(images=images, cls_true=np.argmax(labels, 1), cls_pred=np.argmax(pred, 1),img_size=28,num_channels=1)
                else:
                    print('No checkpoint file found')
                    return


    def train(self,train):
        """ 训练一个计算图模型"""
        self._define_graph()
        merged_summary_op = tf.summary.merge_all() # 合并所有的summary为一个操作节点，方便运行
        saver = tf.train.Saver()# 网络模型保存器
        # 开始训练
        with tf.Session() as sess:
            tf.global_variables_initializer().run()  # 初始化所有变量
            train_writer = tf.summary.FileWriter(self.logs_save_path, sess.graph) # 文件输出对象，用于生成graph event文件
            for i in range(1, self.training_step + 1):
                xs, ys = train.next_batch(self.batch_size) # 获取一个批次
                # 定期保存网络
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, self.model_name), global_step=self.global_step)  # 保存cnpk模型
                    # 执行优化器、损失值和step
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_value, step = sess.run([self.optimizer, self.loss, self.global_step], feed_dict={self.x_input: xs, self.y_input: ys},
                                                   options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i) # 添加到meta中来保存信息
                    print('Epoich: %d , loss: %g. and save model successfully' % (step, loss_value))
                # 定期打印信息和记录变量
                elif i % 10 == 0:
                    # 直接执行优化器、损失值和step和合并操作
                    _, loss_value, step, summary = sess.run([self.optimizer, self.loss, self.global_step, merged_summary_op],
                                                            feed_dict={self.x_input: xs, self.y_input: ys})
                    print('Epoich: %d , loss: %g.' % (step, loss_value))
                    train_writer.add_summary(summary, i) # 添加到graph event文件中用于TensorBoard的显示
                else:
                    _, step = sess.run( [self.optimizer, self.global_step],feed_dict={self.x_input: xs, self.y_input: ys})# 优化参数
            train_writer.close()

    def _define_graph(self):
        """ 定义一个计算图"""
        # 定义计算图的输入结构
        with tf.name_scope('input'):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input], name='x-input')  # 网络输入格式
            self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_output], name='y-input')  # 网络标签格式
        # 定义计算图的输出变量
        with tf.name_scope('output'):
            regularizer = tf.contrib.layers.l2_regularizer(REGULARIZTION_RATE)  # L2正则化函数
            self.output = self._define_net(self.x_input,regularizer__function = regularizer,is_historgram=True)
        # 定义一个不用于训练的step变量，用来更新衰减率
        self.global_step = tf.Variable(0, trainable=False,name='global_step')
        # 定义一个滑动平均模型，该技巧用于使神经网络更加健壮
        with tf.name_scope('moving_averages'):
            variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay,self.global_step)  # 生成一个滑动平均的类：v/ExponentialMovingAverage：0
            variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 定义一个更新变量滑动平均的操作
        # 定义计算图的损失函数
        with tf.name_scope('loss'):
            #cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_input * tf.log(self.output), reduction_indices=[1]))  # 使用交叉熵计算损失函数
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=tf.argmax(self.y_input, 1))
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            self.loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 交叉熵损失添加上正则化损失
            tf.summary.scalar("loss", self.loss) # 使用TensorBoard监测该变量
        # 定义计算图的准确率
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y_input, 1))  # 计算本次预测是否正确
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将正确值转化为浮点类型，再转化为概率
            tf.summary.scalar("accuracy", self.accuracy) # 使用TensorBoard监测该变量
        # 定义计算图的学习率
        with tf.name_scope('learning_rate'):
            # 学习率的变化
            learning_rate = tf.train.exponential_decay(self.learn_rate_base,
                                                       self.global_step,
                                                       mnist.train.num_examples / BATCH_SIZE,
                                                       self.learn_rate_decay)
            tf.summary.scalar("learning_rate", learning_rate)  # 使用TensorBoard监测该变量
        # 定义计算图的优化器
        with tf.name_scope('optimizer'):
            # 定义优化器
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,global_step=self.global_step)
            self.optimizer = tf.group(train_step, variable_averages_op)
        # 查看所有的变量
        for v in tf.global_variables():
            print(v.name)
    def _define_net(self, input,regularizer__function=None,is_historgram=True):
        """ 定义一个全连接神经网络"""
        # 定义layer1层
        layer1 = self._define_layer(input, # 输入张量
                                    self.n_input, # 输入维度
                                    self.n_layer_1, # 输出维度
                                    index_layer=1,  # 本神经层命名序号
                                    activation_function=tf.nn.relu,# 激活函数
                                    regularizer__function=regularizer__function,# 正则化函数
                                    is_historgram=is_historgram)  # 是否用TensorBoard可视化该变量
        # 定义layer2层
        output = self._define_layer(layer1,
                                    self.n_layer_1,
                                    self.n_output,
                                    index_layer=2,
                                    activation_function=None,
                                    regularizer__function=regularizer__function,
                                    is_historgram=is_historgram)  # 预测值
        #output.name = "output"
        print(output.name)
        return output
    def _define_layer(self,inputs, in_size, out_size, index_layer, activation_function=None,regularizer__function=None,is_historgram=True):
        """ 定义一个全连接神经层"""
        layer_name = 'layer%s' % index_layer # 定义该神经层命名空间的名称
        with tf.variable_scope(layer_name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('weights'):
                weights = tf.get_variable('w', [in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
                if regularizer__function != None: # 是否使用正则化项
                    tf.add_to_collection('losses', regularizer__function(weights))  # 将正则项添加到一个名为'losses'的列表中
                if is_historgram: # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/weights', weights)#第一个参数是图表的名称，第二个参数是图表要记录的变量
            with tf.variable_scope('biases'):
                biases = tf.get_variable('b', [1, out_size], initializer=tf.constant_initializer(0.0))
                if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/biases', biases)
            with tf.variable_scope('wx_plus_b'):
                # 神经元未激活的值，矩阵乘法
                wx_plus_b = tf.matmul(inputs, weights) + biases
            # 使用激活函数进行激活
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                tf.summary.histogram(layer_name + '/outputs', outputs)
        # 返回神经层的输出
        return outputs
    def _define_layer1(self,inputs, in_size, out_size, index_layer, activation_function=None,regularizer__function=None,is_historgram=True):
        """ 定义一个全连接神经层"""
        layer_name = 'layer%s' % index_layer # 定义该神经层命名空间的名称
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]), name='w')
                if regularizer__function != None: # 是否使用正则化项
                    tf.add_to_collection('losses', regularizer__function(weights))  # 将正则项添加到一个名为'losses'的列表中
                if is_historgram: # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/weights', weights)#第一个参数是图表的名称，第二个参数是图表要记录的变量
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
                if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/biases', biases)
            with tf.name_scope('wx_plus_b'):
                # 神经元未激活的值，矩阵乘法
                wx_plus_b = tf.matmul(inputs, weights) + biases
            # 使用激活函数进行激活
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                tf.summary.histogram(layer_name + '/outputs', outputs)
            # 返回神经层的输出
        return outputs




if __name__=='__main__':
    #tf.app.run()
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    model = net_model(mnist.train.num_examples)  # 创建模型
    # 训练模型
    #model.train(mnist.train)  # 训练模型
    # 测试模型
    model.test_accuracy(mnist.validation.images, mnist.validation.labels)
    model.test_random(mnist.validation.images,mnist.validation.labels)
