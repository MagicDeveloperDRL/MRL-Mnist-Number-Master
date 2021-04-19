'''''''''
@file: DataHelpter.py
@author: MRL Liu
@time: 2021/4/19 16:05
@env: Python,Numpy
@desc: 本模块提供可视化读取后的图片文件数据的方法
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import random

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def plot_images(images, cls_true, img_size=28, cls_pred=None, num_channels=1):
    # 检测图像是否存在
    if len(images) <= 0 or len(images)>9:
        print("没有图像来展示或者图像个数过多")
        return
    # 创造一个3行3列的画布
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    fig.canvas.set_window_title('Random Images show')  # 设置字体大小与格式
    for i, ax in enumerate(axes.flat):
        # 显示图片
        if len(images) < i + 1:
            break
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # 展示图像的语义标签和实际预测标签
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # 设置每张图的标签为其xlabel.
        ax.set_xlabel(xlabel)

        # 设置图片刻度
        ax.set_xticks([0, img_size])
        ax.set_yticks([0, img_size])

    plt.show()

if __name__=='__main__':
    # 获取数据集
    mnist = input_data.read_data_sets('./mnist/', one_hot=False)
    # 获取各个数据集的样本数量
    n1 = mnist.train.num_examples
    n2 = mnist.validation.num_examples
    n3 = mnist.test.num_examples
    # 取出32个样本
    xs, ys = mnist.train.next_batch(32)
    # 获取训练集的图片数据和标签
    _images, _labels = mnist.train.images, mnist.train.labels
    # 随机抽取9个数据样本
    random_indices = random.sample(range(len(_images)), min(len(_images), 9))
    images, labels = zip(*[(_images[i], _labels[i]) for i in random_indices])
    # 可视化样本
    plot_images(images=images, cls_true=labels,img_size=28,num_channels=1)