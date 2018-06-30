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

import soundfile as sf

import mpr
import spectrogram


class aiff(object):
    """Docstring for aiff. """

    def __init__(self, path: str) -> None:
        """Initial aiff resource file.

        :path: TODO

        Members
        :_aiff_datas: List of numpy array

        """
        self._max_layers = 6
        self._temporal_point = 8

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

        self._aiff_datas = aiff_datas
        self._samplerate = std_samplerate
        self._temporal_point = 8

        # Calculate temporal point
        temporal_data_point_list = []

        for datas in self._aiff_datas:
            samples = datas.shape[0]

            # The aiff file is long enough
            assert samples > self.samples_calc(1)

            last_temporal_data_point = samples - self.samples_calc(1) - 1

            temporal_data_point = [
                int(last_temporal_data_point / 2**(8 - i))
                for i in range(8)
            ]
            temporal_data_point = temporal_data_point
            temporal_data_point_list.append(temporal_data_point)
            # print(temporal_data_point)

        self._temporal_data_point_list = temporal_data_point_list


    def mpr_process(self) -> None:
        """
        :returns: None

        """
        mpr_obj = mpr.mpr(self._aiff_datas, self._temporal_data_point_list,
                          self._temporal_point, self._max_layers)
        mpr_obj.mpr_generate()

    def spectrogram_process(self) -> None:
        """Processing spectrogram
        :returns: TODO

        """
        spectrogram_obj = spectrogram.spectrogram(self._aiff_datas, self._temporal_data_point_list, self._temporal_point)
        spectrogram_obj.spectrogram_generate()

    def samples_calc(self, layer: int) -> int:
        """Calculate the layer samples

        :layer: TODO
        :returns: TODO

        """
        return 2**(5 + 2 * (self._max_layers - layer))


if __name__ == "__main__":
    VIOLIN_PATH = "./resource/violin"
    my_obj = aiff(VIOLIN_PATH)
    my_obj.spectrogram_process()
    my_obj.mpr_process()
    # data, samplerate = sf.read(
    #     "./resourse/violin/Violin.arco.ff.sulA.stereo/Violin.arco.ff.sulA.E5.stereo.aif"
    # )
    # plt.plot(data[:, 0], '-')
    # plt.show()
    # data.shape
    # three_data = np.array([data, data, data])
    # three_data.shape
    # np.mean(three_data, (0, 1)).shape
