#!/usr/bin/env python
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  data_prepared.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-07-01 00:09
# purpose     :
#########################################################################

# TODO: Add description about .npy file structure
# **********************************************************
# About Spectrogram .npy file
# In `spectrogram.py` file
#
# @see spectrogram_generate()
# @see spectrogram_plot()
# @see spectrogram_image_save()
#
# Function Relationship:
# spectrogram_generate() invoke these two function
# :spectrogram_plot(): To plot && save image in the current directory,
# and the path can be set in this function.
# :spectrogram_image_save(): To generate .npy file in the current directory,
# and the path can be set in this function.
#
# :spectrogram_generate(): to prepared data and generate file

# About mpr .npy file
# In `mpr.py` file
#
# @see mpr_generate()
# @see mpr_plot()
# @see mpr_image_save()
#
# Function Relationship:
# mpr_generate() invoke these two function
# :mpr_plot(): To generate plot and save image in the current directory
# and the path can be set in this function.
# :mpr_image_save(): To generate .npy file in the current directory,
# and the path can be set in this function.
#
# :mpr_generate(): to prepared data and generate file


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
import numpy as np

output_types = 13
# TODO: The input y should be one-hot input i.e. batch*output_types ndarray, please check NN to make sure
# TODO: The number of instruments excluding flute is 13, please check NN to make sure
# TODO: shuffle_data should be called before loading data set, please check NN to make sure
label = np.load("label.npy")
mpr_x = np.load("mpr_x.npy")
spectrogram_x = np.load("spectrogram_x.npy")
batch_num = label.shape[0]


def load_mpr_dataset():
    """load the multilayer Recurrent Plot dataset
    :returns: X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1

    """
    train_num = int(batch_num * 3 / 5)
    test_val_num = int(batch_num / 5)
    X_train_1 = mpr_x[0:train_num]
    y_train_1 = label[0:train_num]
    X_val_1 = mpr_x[train_num:train_num + test_val_num]
    y_val_1 = label[train_num:train_num + test_val_num]
    X_test_1 = mpr_x[train_num + test_val_num:train_num + test_val_num * 2]
    y_test_1 = mpr_x[train_num + test_val_num:train_num + test_val_num * 2]
    return X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1


def load_spectrogram_dataset():
    """load the spectrogram datasets
    :returns: X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2

    """
    train_num = int(batch_num * 3 / 5)
    test_val_num = int(batch_num / 5)
    X_train_2 = spectrogram_x[0:train_num]
    y_train_2 = label[0:train_num]
    X_val_2 = spectrogram_x[train_num:train_num + test_val_num]
    y_val_2 = label[train_num:train_num + test_val_num]
    X_test_2 = spectrogram_x[train_num + test_val_num:train_num + test_val_num * 2]
    y_test_2 = spectrogram_x[train_num + test_val_num:train_num + test_val_num * 2]
    return X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2


def shuffle_data() -> None:
    """
    Randomly shuffles the data. Remember to call before loading data set
    :return: None
    """
    state = np.random.get_state()
    np.random.shuffle(label)
    np.random.set_state(state)
    np.random.shuffle(mpr_x)
    np.random.set_state(state)
    np.random.shuffle(spectrogram_x)
