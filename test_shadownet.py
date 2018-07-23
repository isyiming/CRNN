import os.path as ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math


from local_utils import data_utils
from crnn_model import crnn_model
from global_configuration import config


def test_shadownet(dataset_dir, weights_path, is_vis=False, is_recursive=True):

    images_t, labels_t, imagenames_t = data_utils.read_features(dataset_dir, num_epochs=None)#读取.tfrecords文件

    if not is_recursive:
    	#如果设置is_recursive为flase，则创建一个乱序的数据序列。
    	#capacity读取数据范围；min_after_dequeue越大，数据越乱
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=32, capacity=1000+32*2,
                                                                     min_after_dequeue=2, num_threads=4)
    else:
    	#如果设置is_recursive为True，则不打乱数据顺序
        images_sh, labels_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                                                             batch_size=32, capacity=1000 + 32 * 2, num_threads=4)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)#将图像数据类型转为float32

    # 在这里声明了创建网络的类
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)

    with tf.variable_scope('shadow'):#通过tf.variable_scope生成一个上下文管理器
        net_out = net.build_shadownet(inputdata=images_sh)#创建网络，指定输入数据

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32), merge_repeated=False)#对数据解码

    # 设置session配置参数
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # 初始化保存数据
    saver = tf.train.Saver()

    #创建图运算
    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(dataset_dir):
        test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / 32))

    with sess.as_default():#创建图计算的默认会话，当上下文管理器关闭时，这个对话不会关闭

        # 加载网络权重
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()#创建一个协调器，管理线程 
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)#启动QueueRunner, 此时文件名队列已经进队

        print('开始预测文字......')
        if not is_recursive:#如果设置is_recursive为flase，则创建一个乱序的数据序列。，和最开始创建数据系列方式保持一致
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])#运行图计算
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = data_utils.sparse_tensor_to_str(predictions[0])#获取的预测文字结果
            gt_res = data_utils.sparse_tensor_to_str(labels)#真实的结果
            accuracy = []#用来保存准确率

            for index, gt_label in enumerate(gt_res):#enumerate方式同时获取来一个list的索引和对应元素
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):#这里逐项对比预测结果和真实结果，记录准确结果个数
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)#错误的/全部的几位准确率
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print(' test accuracy 为 %f' %(accuracy))

            for index, image in enumerate(images):
                print('预测图片 %s 准确的label为: %s **** 预测的 label: %s' %(imagenames[index], gt_res[index], preds_res[index]))
                if is_vis:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:#这里是非乱序获取数据序列的，和上面的if对应
            accuracy = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = data_utils.sparse_tensor_to_str(predictions[0])
                gt_res = data_utils.sparse_tensor_to_str(labels)

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
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

                for index, image in enumerate(images):
                    print('预测图片 %s 准确的label为: %s **** 预测的label: %s' %(imagenames[index], gt_res[index], preds_res[index]))
                    if is_vis:#如果在recognize()中，将is_vis=True，则显示图片
                        plt.imshow(image[:, :, (2, 1, 0)])
                        plt.show()

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Test accuracy is %f' %(accuracy))

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':

    dataset_dir='data/test_feature.tfrecords'
    weights_path='model/shadownet/shadownet_2018-4-21-11-47-46.ckpt-199999'
    # 测试网络
    test_shadownet(dataset_dir, weights_path)
