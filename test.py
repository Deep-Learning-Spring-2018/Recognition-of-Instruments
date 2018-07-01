from unittest import TestCase, main

import soundfile as sf

from data_prepared import *
from spectrogram import spectrogram_single


class SpectrogramTestCase(TestCase):
    def test_generate_spectrogram_shape(self):
        with open('/home/bill/Documents/curriculum/深度学习/code/test.aif',
                  'rb') as f:
            data, rate = sf.read(f)
        spectrum = spectrogram_single(data=data, acquisition_points=[128, 256, 512, 1024, 2048, 4096], sample_rate=rate,
                                      hop=0.25, resolution=32, spectrogram_length=32)
        self.assertEqual(spectrum.shape, (6, 32, 32, 2))

    def test_dataset_shape(self):
        shuffle_data()
        load_mpr_dataset()
        X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1 = load_mpr_dataset()
        X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2 = load_spectrogram_dataset()
        self.assertEqual(X_train_1.shape[1:], (32, 32, 96))
        self.assertEqual(y_train_1.shape[1:], (12,))


if __name__ == '__main__':
    main()
