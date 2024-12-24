import os
from .dataset_RGB import DataReader, NonRefDataReader


def get_data(img_dir, inp, tar, mode='train', ori=False, img_options=None):
    assert os.path.exists(img_dir)
    return DataReader(img_dir, inp, tar, mode, ori, img_options)


def get_data_nonref(img_dir, inp, mode='train', ori=False, img_options=None):
    assert os.path.exists(img_dir)
    return NonRefDataReader(img_dir, inp, mode, ori, img_options)

