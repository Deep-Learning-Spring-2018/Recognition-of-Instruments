import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.signal import get_window
from skimage.measure import block_reduce


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
                spectrogram_single(self._aiff_datas[audio_index],
                                   self._temporal_data_point_list[audio_index],
                                   audio_index))

        return spectrogram_image

    def spectrogram_plot(self, spectrogram_image, audio_index=5) -> None:
        """Plot spectrogram image for showing

        :spectrogram_image: TODO
        :returns: None

        """
        # Now plot for a test

        plt.figure(figsize=(4, 12))
        for points in range(self._temporal_point):
            plt.subplot(self._temporal_point, 1, points + 1)
            im = plt.imshow(spectrogram_image[audio_index][points, :, :, 0], cmap=plt.get_cmap('jet'))
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
                       audio_index,
                       sample_rate=44100,
                       hop=0.25,
                       resolution=256,
                       spectrogram_length=256,
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
                       int(window_length * (1 + hop *
                                            (spectrogram_length - 1))), i],
                fs=sample_rate,
                window=get_window(('tukey', hop), resolution * 2),
                nperseg=window_length,
                nfft=resolution * 2,
                noverlap=int(window_length * (1 - hop)),
                return_onesided=True)

            # window_length = resolution * 2
            # f, t, sxx = scipy.signal.spectrogram(
            #     x=data[acquisition_points[j]:acquisition_points[j] +
            #            int(window_length * (1 + hop *
            #                                 (spectrogram_length - 1))), i],
            #     fs=sample_rate,
            #     return_onesided=True)
            sxx = np.log(sxx)
            sxx = (sxx - np.mean(sxx)) / np.std(sxx)
            channel_ret.append(
                np.resize(sxx, [resolution, spectrogram_length]))
        ret.append(channel_ret)

    ret = np.transpose(ret, [1, 2, 3, 0])

    # Average pooling process

    average_pooling_size = ret.shape[1] // 32
    ret = block_reduce(ret, (1, average_pooling_size, average_pooling_size, 1), np.average)
    print(ret.shape)

    return ret



