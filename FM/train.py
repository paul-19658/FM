'''
# Time   : 2020/10/21 17:51
# Author : junchaoli
# File   : train_lstm.py
'''

'''
2024/11/4 zc
原作者的tf和keras版本在今天已经不再更新维护了，也找不到了
我把原版的一些错误修复了，以及
基于 tf 2.18.0
keras 3.6.0
把代码做了一些改动
现在可以直接在这个版本跑
'''

from model import FM
from utils import create_criteo_dataset

import numpy as np
import tensorflow as tf
from keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

import argparse
parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('-k', type=int, help='v_dim', default=8)
parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
args=parser.parse_args()

if __name__ == '__main__':
    file_path = '../Data/train.txt'
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)
    # test
    # print(X_train.shape)
    # print(y_train.shape)

    # shape问题 改动02
    y_train = np.array(y_train).reshape(-1, 1)

    # print(y_train.shape)
    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg

    model = FM(k, w_reg, v_reg)
    optimizer = optimizers.SGD(0.01)
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    # model.fit(train_dataset, epochs=200)
    # print(model.evaluate(X_test, y_test))
    # model.summary()

    # 转换格式？ 改动01
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')

    summary_writer = tf.summary.create_file_writer('./tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))





