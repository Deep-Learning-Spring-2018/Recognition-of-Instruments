#!/usr/bin/env python
# -*-coding:utf-8 -*-
#########################################################################
# File Name   :  cnn_network.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-06-29 17:16
# purpose     :
#########################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def main_test_cnn_layer():

    # **********************************************************
    # TODO
    # This Place is to Prepared data
    # :X_train_mpr: ? * 32 * 32 * 2 ndarray(float)
    # :X_train_spectrogram: ? * 32 * 32 * 2 ndarray(float)
    # :X_train_mpr: ? * 2 ndarray(int64)
    # :X_train_spectrogram: ? * 2 ndarray(int64)


    # :X_val_mpr: ? * 32 * 32 * 2 ndarray(float)
    # :X_val_spectrogram: ? * 32 * 32 * 2 ndarray(float)
    # :X_val_mpr: ? * 2 ndarray(int64)
    # :X_val_spectrogram: ? * 2 ndarray(int64)

    # :X_test_mpr: ? * 32 * 32 * 2 ndarray(float)
    # :X_test_spectrogram: ? * 32 * 32 * 2 ndarray(float)
    # :X_test_mpr: ? * 2 ndarray(int64)
    # :X_test_spectrogram: ? * 2 ndarray(int64)

    # ATTENTION: X_train can be mpr data or spectrogram data, but should prepared sperately.
    # btw, I have no idea how to prepared data sperated for two net, multicolumn CNN is NOT implemented yet

    # This is official mnist data
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(
        shape=(-1, 28, 28, 1))
    # **********************************************************

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 16

    x_mpr = tf.placeholder(
        tf.float32, shape=[batch_size, 32, 32,
                           2])  # [batch_size, height, width, channels]


    x_spectrogram = tf.placeholder(
        tf.float32, shape=[batch_size, 32, 32,
                           2])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size])

    net_mpr = tl.layers.InputLayer(x_mpr, name='input_mpr')
    net_spectrogram = tl.layers.InputLayer(
        x_spectrogram, name='input_spectrogram')
    # Professional conv API for tensorflow expert
    # net = tl.layers.Conv2dLayer(net,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn1')     # output: (?, 28, 28, 32)
    # net = tl.layers.PoolLayer(net,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool1',)   # output: (?, 14, 14, 32)
    # net = tl.layers.Conv2dLayer(net,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn2')     # output: (?, 14, 14, 64)
    # net = tl.layers.PoolLayer(net,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool2',)   # output: (?, 7, 7, 64)
    # Simplified conv API (the same with the above layers)

    # For MPR image network
    net_mpr = tl.layers.Conv2d(
        net_mpr,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn1_mpr')
    net_mpr = tl.layers.MaxPool2d(
        net_mpr, (2, 2), (2, 2), padding='SAME', name='pool1_mpr')
    net_mpr = tl.layers.Conv2d(
        net_mpr,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn2_mpr')
    net_mpr = tl.layers.MaxPool2d(
        net_mpr, (2, 2), (2, 2), padding='SAME', name='pool2_mpr')
    net_mpr = tl.layers.FlattenLayer(net_mpr, name='flatten_mpr')

    # For spectrogram image network
    net_spectrogram = tl.layers.Conv2d(
        net_spectrogram,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn1_spectrogram')
    net_spectrogram = tl.layers.MaxPool2d(
        net_spectrogram, (2, 2), (2, 2),
        padding='SAME',
        name='pool1_spectrogram')
    net_spectrogram = tl.layers.Conv2d(
        net_spectrogram,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn2_spectrogram')
    net_spectrogram = tl.layers.MaxPool2d(
        net_spectrogram, (2, 2), (2, 2),
        padding='SAME',
        name='pool2_spectrogram')
    net_spectrogram = tl.layers.FlattenLayer(
        net_spectrogram, name='flatten_spectrogram')

    # end of conv

    net = tl.layers.ConcatLayer(
        [net_mpr, net_spectrogram], name='concat_layer')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop1')
    net = tl.layers.DenseLayer(net, 256, act=tf.nn.relu, name='relu1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, 10, act=None, name='output')

    y = net.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 3200
    learning_rate = 0.0001
    print_freq = 10

    train_params = net.all_params
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        # Show train information every print_freq
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch,
                                               time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(
                    net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                    X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(
                    net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     tl.vis.CNN2d(net.all_params[0].eval(), second=10, saveable=True, name='cnn1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
            X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))


if __name__ == "__main__":
    main_test_cnn_layer()
