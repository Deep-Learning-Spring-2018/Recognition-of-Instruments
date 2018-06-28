from unittest import TestCase, main

import numpy as np
import scipy.signal
import soundfile as sf
from scipy.signal import get_window


def generate_spectrogram(data,
                         acquisition_points,
                         sample_rate=44100,
                         hop=0.25,
                         resolution=32,
                         spectrogram_length=32,
                         channel_num=2):
    """

    :param data: Time series data, a x*channel_num numpy.ndarray
    :param sample_rate: Sample rate, float
    :param acquisition_points: An ascending list of int of length n, points to take spectrogram
    :param hop: percentage of hop (=1-percentage of overlap)
    :param resolution: number of frequencies
    :param spectrogram_length: number of spectrum in a single graph
    :param channel_num: number of channels
    :return: a len(acquisition_points)*resolution*spectrogram_length*channel_num numpy.ndarray, the spectrogram
    """
    ret = []
    for i in range(channel_num):
        channel_ret = []
        for j in range(len(acquisition_points)):
            window_length = resolution * 2
            f, t, sxx = scipy.signal.spectrogram(
                x=data[acquisition_points[j]:acquisition_points[j] + int(
                    window_length * (1 + (hop * len(acquisition_points) - 1))), i],
                fs=sample_rate,
                window=get_window(('tukey', .25), 64),
                nperseg=window_length,
                nfft=resolution * 2,
                noverlap=int(window_length * (1 - hop)),
                return_onesided=True,
                mode='magnitude'
            )
            channel_ret.append(np.resize(sxx, [resolution, spectrogram_length]))
        ret.append(channel_ret)
    return np.transpose(ret, [1, 2, 3, 0])


class SpectrogramTestCase(TestCase):
    def test_generate_spectrogram_shape(self):
        with open('/home/bill/Documents/curriculum/深度学习/code/test.aif', 'rb') as f:
            data, rate = sf.read(f)
        spectrum = generate_spectrogram(
            data=data,
            sample_rate=rate,
            acquisition_points=[128, 256, 512, 1024, 2048, 4096],
            hop=0.25,
            resolution=32,
            spectrogram_length=32
        )
        self.assertEqual(spectrum.shape, (6, 32, 32, 2))


if __name__ == '__main__':
    main()
