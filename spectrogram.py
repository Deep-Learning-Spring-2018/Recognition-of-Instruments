from unittest import TestCase, main

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
from scipy.signal import get_window


class spectrogram(object):
    """Main class to implement spectrogram"""

    def __init__(self, aiff_datas, temporal_data_point_list, temporal_point):
        """Prepared data for spectrogram

        :aiff_datas: TODO
        :temporal_data_point_list: TODO
        :temporal_point: TODO

        """
        self._aiff_datas = aiff_datas
        self._temporal_data_point_list = temporal_data_point_list
        self._temporal_point = temporal_point

    def spectrogram_generate(self):
        """Generate spectrum image and numpy file
        :returns: TODO

        """
        spectrogram_image = []

        for audio_index in range(len(self._aiff_datas)):
            spectrogram_image.append(
                spectrogram_single(
                    self._aiff_datas[audio_index],
                    self._temporal_data_point_list[audio_index]))

        self.spectrogram_plot(spectrogram_image)
        self.spectrogram_image_save(spectrogram_image)

    def spectrogram_plot(self, spectrogram_image, audio_index=0) -> None:
        """Plot spectrogram image for showing

        :spectrogram_image: TODO
        :returns: None

        """
        # Now plot for a test

        plt.figure(figsize=(4, 12))
        for points in range(self._temporal_point):
            plt.subplot(self._temporal_point, 1, points + 1)
            im = plt.imshow(spectrogram_image[audio_index][points, :, :, 0])
            plt.axis('off')
            plt.colorbar(im)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig('Spectrogram1.png')
        plt.close()

    def spectrogram_image_save(self, spectrogram_image):
        """Saving spectrogram_image to numpy file

        :spectrogram_image: TODO
        :returns: TODO

        """
        spectrogram_image_numpy_image = np.array(spectrogram_image)
        np.save('spectrogram_image.npz', spectrogram_image_numpy_image)


def spectrogram_single(data,
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
                x=data[acquisition_points[j]:acquisition_points[j] +
                       int(window_length *
                           (1 + hop * (spectrogram_length - 1))), i],
                fs=sample_rate,
                window=get_window(('tukey', .25), 64),
                nperseg=window_length,
                nfft=resolution * 2,
                noverlap=int(window_length * (1 - hop)),
                return_onesided=True)
            for x in range(resolution):
                for y in range(spectrogram_length):
                    sxx[x, y] = np.log(sxx[x, y])
            channel_ret.append(
                np.resize(sxx, [resolution, spectrogram_length]))
        ret.append(channel_ret)
    return np.transpose(ret, [1, 2, 3, 0])


class SpectrogramTestCase(TestCase):
    def test_generate_spectrogram_shape(self):
        with open('/home/bill/Documents/curriculum/深度学习/code/test.aif', 'rb') as f:
            data, rate = sf.read(f)
        spectrum = spectrogram_single(
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
