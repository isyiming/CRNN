import os
import os.path as ops
import argparse
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from data_provider import data_provider
from local_utils import data_utils

def TextDataProvider(dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None):
    '''
        这个函数用来读取我们用于训练的数据，包括：图像，对应的标签（在sample.txt中），和图片名称
        sample.txt文件保存的格式为： 图片路径+图片名称 ‘空格’ 图片中的文字内容 

        将这些单独的图片（label和图片名称同样的）保存为np矩阵方便训练网络读取
        这个函数得到的np矩阵传递给write_features，保存为.tfrecords格式文件
    '''
    assert ops.exists(dataset_dir)#判断输入路径是否存在

    # 读取test数据
    test_dataset_dir = ops.join(dataset_dir, 'Test')#test数据的图片保存路径
    assert ops.exists(test_dataset_dir)#判断输入路径是否存在
    test_anno_path = ops.join(test_dataset_dir, annotation_name)#test数据的label路径
    assert ops.exists(test_anno_path)#判断输入路径是否存在

    with open(test_anno_path, 'r') as anno_file:#创建上下文管理器打开sample.txt (里面保存图片image对应的标签label)
        info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])#逐行读取txt文件，并把内容保存在np矩阵中
        test_images = np.array([cv2.imread(ops.join(test_dataset_dir, tmp), cv2.IMREAD_COLOR)
                               for tmp in info[:, 0]])#读取所有的图片保存为np矩阵
        test_labels = np.array([tmp for tmp in info[:, 1]])#将所有的label保存为np矩阵形式
        test_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])#将所有的图片名称保存为np矩阵形式
    anno_file.close()#注销txt文件

    # 读取train数据，这里的和上面的一样，就不注释了
    train_dataset_dir = ops.join(dataset_dir, 'Train')
    assert ops.exists(train_dataset_dir)
    train_anno_path = ops.join(train_dataset_dir, annotation_name)
    assert ops.exists(train_anno_path)

    with open(train_anno_path, 'r') as anno_file:
        info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
        train_images = np.array([cv2.imread(ops.join(train_dataset_dir, tmp), cv2.IMREAD_COLOR)
                                for tmp in info[:, 0]])
        train_labels = np.array([tmp for tmp in info[:, 1]])
        train_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])
    anno_file.close()

    test={'test_images':test_images,'test_labels':test_labels,'test_imagenames':test_imagenames}
    train={'train_images':train_images,'train_labels':train_labels,'train_imagenames':train_imagenames}

    return test,train#以字典方式传递给write_features然后保存


def write_features(dataset_dir, save_dir):
    '''
        这个函数用来将数据写入.tfrecords文件保存
    '''
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    print('正在录入训练数据 ......')
    [test,train] = TextDataProvider(dataset_dir=dataset_dir, annotation_name='sample.txt',
                                              validation_set=True, validation_split=0.15, shuffle='every_epoch',
                                              normalization=None)
    print('训练数据分析完成')

    print('写入训练数据至 tf records')
    train_images=train['train_images']
    train_labels=train['train_labels']
    train_imagenames=train['train_imagenames']

    train_images = [cv2.resize(tmp, (100, 32)) for tmp in train_images]#改变图像大小，目的：使图片大小统一
    train_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in train_images]#将图片这种整形数据转为bytes形式
    train_tfrecord_path = ops.join(save_dir, 'train_feature_new.tfrecords')#要保存的.tfrecords文件路径
    data_utils.write_features(tfrecords_path=train_tfrecord_path, labels=train_labels, images=train_images,
                                     imagenames=train_imagenames)#写入.tfrecords文件

    print('写入测试数据至 tf records')#这里和上面一样，也不注释了
    test_images=test['test_images']
    test_labels=test['test_labels']
    test_imagenames=test['test_imagenames']
    test_images = [cv2.resize(tmp, (100, 32)) for tmp in test_images]
    test_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in test_images]

    test_tfrecord_path = ops.join(save_dir, 'test_feature_new.tfrecords')
    data_utils.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images,
                                     imagenames=test_imagenames)

    return 0


if __name__ == '__main__':

    dataset_dir=path+'data/sample'
    save_dir=path+'data'

    if not ops.exists(dataset_dir):#判断路径是否存在
        raise ValueError('Dataset {:s} doesn\'t exist'.format(dataset_dir))

    write_features(dataset_dir=dataset_dir, save_dir=save_dir)
