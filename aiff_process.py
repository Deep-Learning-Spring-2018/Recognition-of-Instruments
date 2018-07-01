#!/usr/bin/env python
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  aiff_process.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-06-26 18:16
# purpose     :
#########################################################################

from pathlib import Path

import numpy as np
import soundfile as sf

import mpr
import spectrogram


class aiff(object):
    """Docstring for aiff. """

    def __init__(self, path: Path):
        """Initial aiff resource file.

        :path: TODO

        Members
        :_aiff_datas: List of numpy array

        """
        self._max_layers = 6
        self._temporal_point = 8

        aiff_files = path.glob("*.aif")
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
                int(last_temporal_data_point / 2 ** (8 - i))
                for i in range(8)
            ]
            temporal_data_point = temporal_data_point
            temporal_data_point_list.append(temporal_data_point)
            # print(temporal_data_point)

        self._temporal_data_point_list = temporal_data_point_list

    def mpr_process(self):
        """
        :returns: None

        """
        mpr_obj = mpr.mpr(self._aiff_datas, self._temporal_data_point_list,
                          self._temporal_point, self._max_layers)
        return np.asarray(mpr_obj.mpr_generate())

    def spectrogram_process(self):
        """Processing spectrogram
        :returns: TODO

        """
        spectrogram_obj = spectrogram.spectrogram(self._aiff_datas, self._temporal_data_point_list,
                                                  self._temporal_point)
        return np.asarray(spectrogram_obj.spectrogram_generate())

    def samples_calc(self, layer: int) -> int:
        """Calculate the layer samples

        :layer: TODO
        :returns: TODO

        """
        return 2 ** (5 + 2 * (self._max_layers - layer))

    def get_spectrogram(self):
        pass

    def get_mrp(self):
        pass


if __name__ == "__main__":
    instruments = ['AltoSax', 'BassFlute', 'BbClarinet', 'Cello', 'EbClarinet', 'Horn', 'Oboe', 'SopSax',
                   'TenorTrombone', 'Trumpet', 'Viola', 'Violin']
    x_mpr = []
    x_spectrogram = []
    y = []
    for i, instrument in enumerate(instruments):
        print(instrument)
        instrument_path = Path("./resource") / instrument
        aiff_file = aiff(instrument_path)
        _mpr = aiff_file.mpr_process()
        batch_num = _mpr.shape[0]
        _mpr.transpose((0, 3, 4, 5, 1, 2))
        _mpr = _mpr.reshape((batch_num, 32, 32, 96))
        _spectrogram = aiff_file.spectrogram_process()
        _spectrogram.transpose((0, 2, 3, 4, 1))
        _spectrogram = _spectrogram.reshape((batch_num, 32, 32, 16))
        _label = np.zeros((len(instruments)))
        _label[i] = 1
        x_mpr += [_mpr[i] for i in range(batch_num)]
        x_spectrogram += [_spectrogram[i] for i in range(batch_num)]
        y += [_label for i in range(batch_num)]
    np.save("spectrogram_x", x_spectrogram)
    np.save("mpr_x", x_mpr)
    np.save("label", y)
    # data, samplerate = sf.read(
    #     "./resourse/violin/Violin.arco.ff.sulA.stereo/Violin.arco.ff.sulA.E5.stereo.aif"
    # )
    # plt.plot(data[:, 0], '-')
    # plt.show()
    # data.shape
    # three_data = np.array([data, data, data])
    # three_data.shape
    # np.mean(three_data, (0, 1)).shape
