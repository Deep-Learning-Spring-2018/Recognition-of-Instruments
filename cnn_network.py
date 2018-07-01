#!/usr/bin/env python
# -*-coding:utf-8 -*-
#########################################################################
# File Name   :  cnn_network.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-06-29 17:16
# purpose     :
#########################################################################

import time

import tensorflow as tf
import tensorlayer as tl

import data_prepared

tf.logging.set_verbosity(tf.logging.DEBUG)


def load_datasets(image_type_1: str, image_type_2: str):
    """load the whole dataset to process multicolumn CNN
    :returns: TODO

    """
    if image_type_1 == 'mpr':
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = \
            data_prepared.load_mpr_dataset()
        channel_1 = 96
    elif image_type_1 == 'spectrogram':
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = \
            data_prepared.load_spectrogram_dataset()
        channel_1 = 16
    else:
        print("Not a valid image type")
        return

    if image_type_2 == 'mpr':
        X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = \
            data_prepared.load_mpr_dataset()
        channel_2 = 96
    elif image_type_2 == 'spectrogram':
        X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = \
            data_prepared.load_spectrogram_dataset()
        channel_2 = 16
    else:
        print("Not a valid image type")
        return

    return [X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1], \
           [X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2], \
        channel_1, channel_2


def MCNN_mpr_mpr():
    """
    :returns: TODO

    """
    dataset_1, dataset_2, channel_1, channel_2 = load_datasets('mpr', 'mpr')
    main_test_cnn_layer_mpr_spectrogram(dataset_1, dataset_2, channel_1,
                                        channel_2)


def MCNN_spectrogram_spectrogram():
    """
    :returns: TODO

    """
    dataset_1, dataset_2, channel_1, channel_2 = load_datasets(
        'spectrogram', 'spectrogram')
    main_test_cnn_layer_mpr_spectrogram(dataset_1, dataset_2, channel_1,
                                        channel_2)


def MCNN_mpr_spectrogram():
    """
    :returns: TODO

    """
    dataset_1, dataset_2, channel_1, channel_2 = load_datasets(
        'mpr', 'spectrogram')
    main_test_cnn_layer_mpr_spectrogram(dataset_1, dataset_2, channel_1,
                                        channel_2)


def main_test_cnn_layer_mpr_spectrogram(dataset_1, dataset_2, channel_1,
                                        channel_2):
    # **********************************************************
    # TODO
    # This Place is to Prepared data
    # :X_train_1: train_batch * 32 * 32 * 96  ndarray(float)
    # :X_train_2: train_batch * 32 * 32 * 16  ndarray(float)
    # :y_train_1: train_batch                 ndarray(int64)
    # :y_train_2: train_batch                 ndarray(int64)

    # :X_val_1: val_batch * 32 * 32 * 96    ndarray(float)
    # :X_val_2: val_batch * 32 * 32 * 16    ndarray(float)
    # :y_val_1: val_batch                   ndarray(int64)
    # :y_val_2: val_batch                   ndarray(int64)

    # :X_test_1: test_batch * 32 * 32 * 96   ndarray(float)
    # :X_test_2: test_batch * 32 * 32 * 16   ndarray(float)
    # :y_test_1: test_batch                  ndarray(int64)
    # :y_test_2: test_batch                  ndarray(int64)

    # (optional)
    # By conventional, val_batch : train_batch : val_batch = 1 : 3 : 1

    # ATTENTION: Please inplement the data output in the function:
    #   @see load_mpr_dataset()
    #   @see load_spectrogram_dataset()
    # These two functions are at the top of the cnn_network.py

    # Btw, Please specify the instrument types when handle with the data
    output_types = 13

    # **********************************************************
    # The following part is about the MCNN network

    # This is official mnist data
    # X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(
    #     shape=(-1, 28, 28, 1))

    # Unpack data list
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = dataset_1
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = dataset_2

    print(X_train_1.shape, y_train_1.shape)
    print(X_train_2.shape, y_train_2.shape)

    print(X_val_1.shape, y_val_1.shape)
    print(X_val_2.shape, y_val_2.shape)

    print(X_test_1.shape, y_test_1.shape)
    print(X_test_2.shape, y_test_2.shape)

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 16

    x_1 = tf.placeholder(
        tf.float32, shape=[batch_size, 32, 32,
                           channel_1])  # [batch_size, height, width, channels]

    x_2 = tf.placeholder(
        tf.float32, shape=[batch_size, 32, 32,
                           channel_2])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size])

    net_1 = tl.layers.InputLayer(x_1, name='input_1')
    net_2 = tl.layers.InputLayer(x_2, name='input_2')
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
    net_1 = tl.layers.Conv2d(
        net_1,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn1_1')
    net_1 = tl.layers.MaxPool2d(
        net_1, (2, 2), (2, 2), padding='SAME', name='pool1_1')
    net_1 = tl.layers.Conv2d(
        net_1,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn2_1')
    net_1 = tl.layers.MaxPool2d(
        net_1, (2, 2), (2, 2), padding='SAME', name='pool2_1')
    net_1 = tl.layers.FlattenLayer(net_1, name='flatten_1')

    # For spectrogram image network
    net_2 = tl.layers.Conv2d(
        net_2,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn1_2')
    net_2 = tl.layers.MaxPool2d(
        net_2, (2, 2), (2, 2), padding='SAME', name='pool1_2')
    net_2 = tl.layers.Conv2d(
        net_2,
        32, (5, 5), (1, 1),
        act=tf.nn.relu,
        padding='SAME',
        name='cnn2_2')
    net_2 = tl.layers.MaxPool2d(
        net_2, (2, 2), (2, 2), padding='SAME', name='pool2_2')
    net_2 = tl.layers.FlattenLayer(net_2, name='flatten_2')

    # end of conv

    net = tl.layers.ElementwiseLayer(
        [net_1, net_2], combine_fn=tf.add, name='concat_layer')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop1')
    net = tl.layers.DenseLayer(net, 256, act=tf.nn.relu, name='relu1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(
        net, output_types, act=tf.identity, name='output')

    y = net.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 400
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

    # Training parameter for one input

    # for epoch in range(n_epoch):
    #     start_time = time.time()
    #     for X_train_a, y_train_a in tl.iterate.minibatches(
    #             X_train, y_train, batch_size, shuffle=True):
    #         feed_dict = {x: X_train_a, y_: y_train_a}
    #         feed_dict.update(net.all_drop)  # enable noise layers
    #         sess.run(train_op, feed_dict=feed_dict)

    #     # Show train information every print_freq
    #     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
    #         print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch,
    #                                            time.time() - start_time))
    #         train_loss, train_acc, n_batch = 0, 0, 0
    #         for X_train_a, y_train_a in tl.iterate.minibatches(
    #                 X_train, y_train, batch_size, shuffle=True):
    #             dp_dict = tl.utils.dict_to_one(
    #                 net.all_drop)  # disable noise layers
    #             feed_dict = {x: X_train_a, y_: y_train_a}
    #             feed_dict.update(dp_dict)
    #             err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    #             train_loss += err
    #             train_acc += ac
    #             n_batch += 1
    #         print("   train loss: %f" % (train_loss / n_batch))
    #         print("   train acc: %f" % (train_acc / n_batch))
    #         val_loss, val_acc, n_batch = 0, 0, 0
    #         for X_val_a, y_val_a in tl.iterate.minibatches(
    #                 X_val, y_val, batch_size, shuffle=True):
    #             dp_dict = tl.utils.dict_to_one(
    #                 net.all_drop)  # disable noise layers
    #             feed_dict = {x: X_val_a, y_: y_val_a}
    #             feed_dict.update(dp_dict)
    #             err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    #             val_loss += err
    #             val_acc += ac
    #             n_batch += 1
    #         print("   val loss: %f" % (val_loss / n_batch))
    #         print("   val acc: %f" % (val_acc / n_batch))
    #         # try:
    #         #     tl.vis.CNN2d(net.all_params[0].eval(), second=10, saveable=True, name='cnn1_' + str(epoch + 1), fig_idx=2012)
    #         # except:  # pylint: disable=bare-except
    #         #     print("You should change vis.CNN(), if you want to save the feature images for different dataset")

    # print('Evaluation')
    # test_loss, test_acc, n_batch = 0, 0, 0
    # for X_test_a, y_test_a in tl.iterate.minibatches(
    #         X_test, y_test, batch_size, shuffle=True):
    #     dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
    #     feed_dict = {x: X_test_a, y_: y_test_a}
    #     feed_dict.update(dp_dict)
    #     err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    #     test_loss += err
    #     test_acc += ac
    #     n_batch += 1
    # print("   test loss: %f" % (test_loss / n_batch))
    # print("   test acc: %f" % (test_acc / n_batch))

    for epoch in range(n_epoch):
        start_time = time.time()
        # In fact y_train_a_1 is y_
        for [X_train_a_1, y_train_a_1],\
            [X_train_a_2, y_train_a_2] \
                in zip(
                    tl.iterate.minibatches(
                X_train_1, y_train_1,
                batch_size, shuffle=True),
                    tl.iterate.minibatches(
                X_train_2, y_train_2,
                batch_size, shuffle=True)):
            feed_dict = {x_1: X_train_a_1, x_2: X_train_a_2, y_: y_train_a_1}

            # enable noise layers when not output
            feed_dict.update(net.all_drop)

            # Run on the graph
            sess.run(train_op, feed_dict=feed_dict)

        # Show train information every print_freq
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch,
                                               time.time() - start_time))

            train_loss, train_acc, n_batch = 0, 0, 0

            for [X_train_a_1, y_train_a_1],\
                [X_train_a_2, y_train_a_2] \
                    in zip(
                        tl.iterate.minibatches(
                    X_train_1, y_train_1,
                    batch_size, shuffle=True),
                        tl.iterate.minibatches(
                    X_train_2, y_train_2,
                    batch_size, shuffle=True)):
                # disable noise layers
                dp_dict = tl.utils.dict_to_one(net.all_drop)

                feed_dict = {
                    x_1: X_train_a_1,
                    x_2: X_train_a_2,
                    y_: y_train_a_1
                }
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            # Now Start validation data process

            val_loss, val_acc, n_batch = 0, 0, 0
            for [X_val_a_1, X_val_a_2],\
                [y_val_a_1, y_val_a_2] \
                    in zip(
                        tl.iterate.minibatches(
                    X_val_1, y_val_1,
                    batch_size, shuffle=True),
                        tl.iterate.minibatches(
                    X_val_2, y_val_2,
                    batch_size, shuffle=True)):
                dp_dict = tl.utils.dict_to_one(
                    net.all_drop)  # disable noise layers
                feed_dict = {x_1: X_val_a_1, x_2: X_val_a_2, y_: y_val_a_1}
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
    for [X_test_a_1, y_test_a_1],\
        [X_test_a_2, y_test_a_2] \
            in zip(
                tl.iterate.minibatches(
            X_test_1, y_test_1,
            batch_size, shuffle=True),
                tl.iterate.minibatches(
            X_test_2, y_test_2,
            batch_size, shuffle=True)):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x_1: X_test_a_1, x_2: X_test_a_2, y_: y_test_a_1}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))


if __name__ == "__main__":
    MCNN_mpr_spectrogram()
