#!/usr/bin/env python
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  mpr.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-06-29 00:38
# purpose     :
#########################################################################

import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from numba import jit


class mpr(object):
    """To calculate mpr image"""

    def __init__(self,
                 aiff_datas,
                 temporal_data_point_list,
                 temporal_point,
                 max_layers):
        """TODO: to be defined1.

        :aiff_datas: TODO
        :max_layers: TODO
        :temporal_point: TODO

        """
        self._aiff_datas = aiff_datas
        self._max_layers = max_layers
        self._temporal_point = temporal_point
        self._temporal_data_point_list = temporal_data_point_list

    def mpr_generate(self) -> None:
        """Some visualing test

        ::

        """

        # Now Start MPR plot generated
        # mpr_plot = audio_indexs * ndarray(8 * 6 * 32 * 32 * 2)
        mpr_image = []

        for audio_index in range(len(self._aiff_datas)):
            # self._aiff_datas[audio_index] is the current audio_file
            # self._temporal_data_point_list[audio_index] is
            #   the current temporal point
            mpr_matrices = []

            for temporal_index in self._temporal_data_point_list[audio_index]:
                mpr_layers = []
                for layer_index in range(1, self._max_layers + 1):

                    # Now Start 1d max pooling process
                    n_signal = self.n_calc(layer_index) - 5

                    max_pooling_1d_size = 2**(max(n_signal - 3, 0))

                    # ******************* Debug Log
                    # print(temporal_index)
                    # print(temporal_index + self.samples_calc(layer_index))
                    # print(self._aiff_datas)

                    signal_1d_after_max_pooling =  \
                        block_reduce(
                            self._aiff_datas[audio_index][temporal_index:temporal_index +
                                                          self.samples_calc(layer_index)],
                            (max_pooling_1d_size, 1), np.max)

                    # 2d RP generation
                    origin_2d_rp_two_track = rp_2d_calc(
                        signal_1d_after_max_pooling)

                    # 2d RP max pooling
                    max_pooling_2d_size = 2**(min(n_signal, 3))

                    signal_2d_after_max_pooling =  \
                        block_reduce(
                            origin_2d_rp_two_track,
                            (max_pooling_2d_size,
                             max_pooling_2d_size,
                             1), np.max)

                    # zero centering of RP image
                    final_rp = self.rp_centering(signal_2d_after_max_pooling)

                    # ******************* Debug Log
                    # print("max_pooling_2d_size")
                    # print(self._aiff_datas[audio_index]
                    #       [temporal_index:temporal_index +
                    #        self.samples_calc(layer_index)])
                    # print(signal_1d_after_max_pooling,
                    #       "signal_1d_after_max_pooling.shape")
                    # print(max_pooling_2d_size, "max_pooling_2d_size")
                    # print(origin_2d_rp_two_track, "origin_2d_rp_two_track")
                    # print(signal_2d_after_max_pooling,
                    #       "signal_2d_after_max_pooling.shape")
                    # print(final_rp, "final_rp")

                    mpr_layers.append(final_rp)

                mpr_matrices.append(np.array(mpr_layers))

            mpr_image.append(np.array(mpr_matrices))

            print("{} file is processed".format(audio_index + 1))

        self.mpr_plot(mpr_image)
        self.mpr_image_save(mpr_image)

    def mpr_plot(self, mpr_image, audio_index=0) -> None:
        """Plot mpr image for showing

        :mpr_image: TODO
        :returns: None

        """
        # Now plot for a test

        plt.gray()
        plt.figure(figsize=(16, 12))
        for points in range(self._temporal_point):
            for layers in range(self._max_layers):
                plt.subplot(self._temporal_point, self._max_layers,
                            points * self._max_layers + (layers + 1))
                plt.imshow(mpr_image[audio_index][points, layers, :, :, 0])
                plt.axis('off')
                plt.tight_layout()
                plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig('RP1.png')
        plt.close()

    def mpr_image_save(self, mpr_image):
        """Saving mpr_image to numpy file

        :mpr_image: TODO
        :returns: TODO

        """
        mpr_numpy_image = np.array(mpr_image)
        np.save('mpr_image.npz', mpr_numpy_image)

    def samples_calc(self, layer: int) -> int:
        """Calculate the layer samples

        :layer: TODO
        :returns: TODO

        """
        return 2**(5 + 2 * (self._max_layers - layer))

    def n_calc(self, layer: int) -> int:
        """Calculate the log_2(layer samples)

        :layer: TODO
        :returns: TODO

        """

        return 5 + 2 * (self._max_layers - layer)

    def rp_centering(self,
                     signal_2d_after_max_pooling: np.ndarray) -> np.ndarray:
        """To do zero centering process

        :signal_2d_after_max_pooling: np.ndarray: TODO
        :returns: TODO

        """
        square_2d_signal = np.sqrt(signal_2d_after_max_pooling)

        # Debug log
        # print(square_2d_signal.shape)

        return square_2d_signal - np.mean(square_2d_signal, (0, 1))


@jit(nopython=True, parallel=True)
def rp_2d_calc(signal_1d_after_max_pooling: np.ndarray) -> np.ndarray:
    """Generate 2d RP from 1d after max pooling

    :signal_1d_after_max_pooling: np.ndarray: TODO
    :returns: TODO

    The rp_2d_calc is not a method, because it need @jit for parallel.
    And jit should use nopython=True

    """
    sample_len, track = signal_1d_after_max_pooling.shape
    rp_2d_two_track = np.empty((sample_len, sample_len, track))
    for i in range(sample_len):
        for j in range(sample_len):
            for k in range(track):
                rp_2d_two_track[i, j, k] = \
                    abs(signal_1d_after_max_pooling[i][k] -
                        signal_1d_after_max_pooling[j][k])
    return rp_2d_two_track
