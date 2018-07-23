import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import data_utils


def recognize(image_path, weights_path, is_vis=True):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)#读取图片
    image = cv2.resize(image, (100, 32))#调整图片分辨率
    image = np.expand_dims(image, axis=0).astype(np.float32)#将图片格式转为浮点型

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')#为输入数据占位
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)#声明网络的类
    with tf.variable_scope('shadow'):#通过tf.variable_scope生成一个上下文管理器
        net_out = net.build_shadownet(inputdata=inputdata)#创建网络，指定输入数据
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)#对数据解码


    # 设置session配置参数
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # 初始化保存数据
    saver = tf.train.Saver()
    sess = tf.Session(config=sess_config)#创建图运算

    with sess.as_default():#创建一个上下文管理器
        saver.restore(sess=sess, save_path=weights_path)#载入训练好的网络权重
        preds = sess.run(decodes, feed_dict={inputdata: image})#网络计算
        preds = data_utils.sparse_tensor_to_str(preds[0])#得到的结果保存为字符串型
        print('预测的图像为 %s 结果为 %s' %(ops.split(image_path)[1], preds[0]))#打印结果

        if is_vis:#如果在recognize()中，将is_vis=True，则显示图片
            plt.figure('CRNN 图片')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()

    return


if __name__ == '__main__':

    image_path='data/test_images/test_01.jpg'
    weights_path='model/shadownet/shadownet_2018-4-21-11-47-46.ckpt-199999'

    if not ops.exists(image_path):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_path))

    # 辨别图像中文字
    recognize(image_path=image_path, weights_path=weights_path,is_vis=True)
