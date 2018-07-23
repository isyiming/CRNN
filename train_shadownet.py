import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse

from crnn_model import crnn_model
from local_utils import data_utils
from global_configuration import config


def train_shadownet(dataset_dir, weights_path=None):

    print("读取训练数据")
    images, labels, imagenames = data_utils.read_features(dataset_dir,num_epochs=None)#读取.tfrecords文件
    #创建一个乱序序列用于训练
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)
    inputdata = tf.cast(x=inputdata, dtype=tf.float32)#占位

    print("初始化网络")# 在这里声明了创建网络的类
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow', reuse=False):#通过tf.variable_scope生成一个上下文管理器
        net_out = shadownet.build_shadownet(inputdata=inputdata)#创建网络，指定输入数据

    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=25*np.ones(32)))#按照设定的维度求张量平均值
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out, 25*np.ones(32), merge_repeated=False)#对数据解码
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))#按照设定的维度求张量平均值
    global_step = tf.Variable(0, name='global_step', trainable=False)#初始化图变量
    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE#设定初始学习速率
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)#按照指数衰减方式改变学习速率
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#设定为全局变量

    with tf.control_dependencies(update_ops):#Adadelta算法的优化器
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)


    # 设置tensorflow的模型管理模式
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()#自动管理模式，导入之前已经保存的模型继续训练

    # 设置模型保存路径
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # gpu参数
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)#创建图运算

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # 迭代次数
    train_epochs = config.cfg.TRAIN.EPOCHS

    print("开始训练")
    with sess.as_default():
        if weights_path is None:
            print('完全重新开始训练')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print('在之前的模型:'+'weights_path'+'上继续训练')
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for epoch in range(train_epochs):
            _, c, seq_distance, preds, gt_labels, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])

            # calculate the precision
            preds = data_utils.sparse_tensor_to_str(preds[0])
            gt_labels = data_utils.sparse_tensor_to_str(gt_labels)

            accuracy = []

            for index, gt_label in enumerate(gt_labels):
                pred = preds[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            #
            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                print('Epoch: %d cost= %f seq distance= %f train accuracy= %f' %(epoch + 1, c, seq_distance, accuracy))

            summary_writer.add_summary(summary=summary, global_step=epoch)
            saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return


if __name__ == '__main__':

    dataset_dir='data/train_feature.tfrecords'
    weights_path='model/shadownet/shadownet_2018-4-21-11-47-46.ckpt-199999'
    if not ops.exists(dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(dataset_dir))

    train_shadownet(dataset_dir, weights_path)
    print('Done')
