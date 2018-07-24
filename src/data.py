import os
import ast
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence


class KagglePlanetSequence(Sequence):
    """
    """
    def __init__(self, file_path, data_path, im_size, batch_size, mode='train'):
        self.df = pd.read_csv(file_path)
        self.dp = data_path
        self.imsz = im_size
        self.bsz = batch_size
        self.mode = mode

    # Take labels and list of image locations in memory
        self.labels = [ast.literal_eval(v) for v in self.df['label'].values]
        self.im_list = [os.path.join(self.dp, im + 'jpg') for im in self.df['image_name'].values]

    def __len__(self):
        return len(self.df)

    def get_batch_labels(self, idx):
        if idx == self.__len__() / self.bsz:
            return self.labels[idx * self.bsz:]
        return np.array(self.labels[idx * self.bsz: (idx + 1 * self.bsz)])

    def get_batch_features(self, idx):
        if idx == self.__len__() / self.bsz:
            return [img_to_array(load_img(im, (self.imsz, self.imsz))) / 255. for im in self.im_list[idx * self.bsz:]]
        return np.array([img_to_array(load_img(im, (self.imsz, self.imsz))) / 255. for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y