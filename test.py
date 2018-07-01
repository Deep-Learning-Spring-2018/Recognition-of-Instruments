from unittest import TestCase, main

import soundfile as sf

from spectrogram import spectrogram_single


class SpectrogramTestCase(TestCase):
    def test_generate_spectrogram_shape(self):
        with open('/home/bill/Documents/curriculum/深度学习/code/test.aif',
                  'rb') as f:
            data, rate = sf.read(f)
        spectrum = spectrogram_single(
            data=data,
            sample_rate=rate,
            acquisition_points=[128, 256, 512, 1024, 2048, 4096],
            hop=0.25,
            resolution=32,
            spectrogram_length=32)
        self.assertEqual(spectrum.shape, (6, 32, 32, 2))


if __name__ == '__main__':
    main()
