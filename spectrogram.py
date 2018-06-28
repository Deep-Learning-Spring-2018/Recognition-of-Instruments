from unittest import TestCase, main

import numpy as np
import scipy.signal
import soundfile as sf
from scipy.signal import get_window


def generate_spectrogram(data, sample_rate, acuisition_points, hop, resolution, spectrogram_length):
    ret = []
    for i in range(len(acuisition_points) - 1):
        total_length = acuisition_points[i + 1] - acuisition_points[i]
        window_length = 64
        f, t, sxx = scipy.signal.spectrogram(
            x=data[acuisition_points[i]:acuisition_points[i] + 560],
            fs=sample_rate,
            window=get_window(('tukey', .25), 64),
            nperseg=window_length,
            nfft=resolution * 2,
            noverlap=int(window_length * (1 - hop)),
            return_onesided=True,
            mode='magnitude'
        )
        ret.append(np.resize(sxx, [resolution, spectrogram_length]))
    return np.array(ret)


class SpectrogramTestCase(TestCase):
    def test_generate_spectrogram_shape(self):
        with open('/home/bill/Documents/curriculum/深度学习/code/test.aif', 'rb') as f:
            data, rate = sf.read(f)
        spectrum = generate_spectrogram(
            data=np.transpose(data)[0],
            sample_rate=rate,
            acuisition_points=[128, 256, 512, 1024, 2048, 4096],
            hop=0.25,
            resolution=32,
            spectrogram_length=32
        )
        self.assertEqual(spectrum.shape, (5, 32, 32))


if __name__ == '__main__':
    main()
