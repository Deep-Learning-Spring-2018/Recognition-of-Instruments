#!/usr/bin/env python
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  aiff_process.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-06-26 18:16
# purpose     :
#########################################################################

import glob

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numba import jit
from skimage.measure import block_reduce


class aiff(object):
    """Docstring for aiff. """

    def __init__(self, path: str) -> None:
        """Initial aiff resource file.

        :path: TODO

        Members
        :_aiff_dates: List of numpy array

        """
        self._max_layers = 6
        self._temporol_point = 8

        aiff_files = glob.glob(path + '/**/*.aif', recursive=True)
        aiff_datas = []
        samplerate_list = []
        std_samplerate = 44100

        # Read file
        for files in aiff_files:
            data, samplerate = sf.read(files)
            aiff_datas.append(data)
            samplerate_list.append(samplerate)

        # Assert expression
        for sample_rate in samplerate_list:
            assert sample_rate == std_samplerate, \
                "Sample rate is not {} Hz".format(std_samplerate)

        self._aiff_dates = aiff_datas
        self._samplerate = std_samplerate
        self.mpr_generate()

    def mpr_generate(self) -> None:
        """Some visualing test

        ::

        """
        temporal_data_point_list = []

        for datas in self._aiff_dates:
            samples = datas.shape[0]

            # The aiff file is long enough
            assert samples > self.samples_calc(1)

            last_temporol_data_point = samples - self.samples_calc(1) - 1

            temporal_data_point = [
                int(last_temporol_data_point / 2**(8 - i))
                for i in range(1, 8)
            ]
            temporal_data_point = [0] + temporal_data_point
            temporal_data_point_list.append(temporal_data_point)
            # print(temporal_data_point)

        # Now Start MPR plot generated
        # mpr_plot = audio_indexs * ndarray(8 * 6 * 32 * 32 * 2)
        mpr_image = []

        for audio_index in range(len(self._aiff_dates)):
            # self._aiff_dates[audio_index] is the current audio_file
            # temporal_data_point_list[audio_index] is
            #   the current temporol point
            mpr_matrices = []

            for temporal_index in temporal_data_point_list[audio_index]:
                mpr_layers = []
                for layer_index in range(1, self._max_layers + 1):

                    # Now Start 1d max pooling process
                    after_max_pooling_1d_size = \
                        2**(self.n_calc(layer_index) -
                            max(layer_index - 5 - 3, 0))

                    # ******************* Debug Log
                    # print(temporal_index)
                    # print(temporal_index + self.samples_calc(layer_index))
                    # print(self._aiff_dates)

                    signal_1d_after_max_pooling =  \
                        block_reduce(
                            self._aiff_dates[audio_index][temporal_index:temporal_index + \
                                        self.samples_calc(layer_index)],
                            (after_max_pooling_1d_size, 2), np.max)

                    # 2d RP generation
                    origin_2d_rp_two_track = self.rp_2d_calc(
                        signal_1d_after_max_pooling)
                    # 2d RP max pooling
                    after_max_pooling_2d_size = after_max_pooling_1d_size // 2**(
                        min(self.n_calc(layer_index), 3))

                    # ******************* Debug Log
                    # print("after_max_pooling_2d_size")
                    print(after_max_pooling_2d_size)

                    signal_2d_after_max_pooling =  \
                        block_reduce(
                            origin_2d_rp_two_track,
                            (after_max_pooling_2d_size,
                             after_max_pooling_2d_size,
                             2), np.max)

                    # zero centering of RP image
                    final_rp = self.rp_centering(signal_2d_after_max_pooling)

                    # Debug log
                    # print(final_rp.shape)

                    mpr_layers.append(final_rp)

                mpr_matrices.append(np.array(mpr_layers))

            mpr_image.append(np.array(mpr_matrices))

            # Now plot for a test

            # if audio_index == 0:
            #     plt.gray()
            #     for points in range(self._temporol_point):
            #         for layers in range(self._max_layers):
            #             plt.subplot(self._temporol_point, self._max_layers,
            #                         points * self._max_layers + (layers + 1))
            #             plt.plot(
            #                 mpr_image[audio_index][points, layers, :, :, 0])
            #     plt.savefig('RP1.png')
            #     plt.close()

            if audio_index == 0:
                plt.imshow(mpr_image[0][0, 0, :, :, 0], cmap=plt.get_cmap('gray'))
                print(mpr_image[0][0, 0, :, :, 0])
                plt.show()

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

    @jit
    def rp_2d_calc(self,
                   signal_1d_after_max_pooling: np.ndarray) -> np.ndarray:
        """Generate 2d RP from 1d after max pooling

        :signal_1d_after_max_pooling: np.ndarray: TODO
        :returns: TODO

        """
        sample_len, track = signal_1d_after_max_pooling.shape
        rp_2d_two_track = np.empty((sample_len, sample_len, track))
        for i in range(sample_len):
            for j in range(sample_len):
                for k in range(track):
                    rp_2d_two_track[i, j, k] = \
                        signal_1d_after_max_pooling[i][k] - \
                        signal_1d_after_max_pooling[j][k]
        return rp_2d_two_track

    @jit
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


if __name__ == "__main__":
    VIOLIN_PATH = "./resourse/violin"
    my_obj = aiff(VIOLIN_PATH)
    # data, samplerate = sf.read(
    #     "./resourse/violin/Violin.arco.ff.sulA.stereo/Violin.arco.ff.sulA.E5.stereo.aif"
    # )
    # plt.plot(data[:, 0], '-')
    # plt.show()
    # data.shape
    # three_data = np.array([data, data, data])
    # three_data.shape
    # np.mean(three_data, (0, 1)).shape
