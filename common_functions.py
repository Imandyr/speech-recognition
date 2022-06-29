# in this file contain common functions


# imports
import random
import time
import os
import numpy as np
from glob import glob
import shutil
import re
from importlib import reload
import pylab as p
import scipy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import pylab
import pandas as pd
from IPython import display
from jiwer import wer
import librosa
from PIL import Image
import imageio
import cv2
import skimage # scikit-image
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import preprocessing
from keras import Model
from keras import Input
from keras import optimizers
from keras import callbacks
from keras import applications
from keras import regularizers
from keras import initializers
from keras import activations
from keras import backend as K


# first image processing
def image_processing(raw_image_array, image_height, image_width, image_channels, rotate=False, grayscale=False, fill_image=False):
    image_array = np.asarray(raw_image_array).astype("float32")

    # turn image into grayscale, if grayscale=True
    if grayscale:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        image_array = np.asarray(image_array).astype("float32").reshape(image_array.shape[0], image_array.shape[1], image_channels)

    # image rotate, if rotate == True
    if rotate:
        image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        image_array = np.asarray(image_array).astype("float32").reshape(image_array.shape[0], image_array.shape[1], image_channels)

    # fill missing space with black color, if fill_image == True
    if fill_image:
        # addition blank pixels to image (make all image have same size)
        image_array = np.append(image_array, np.ones(shape=(image_height+image_width, image_array.shape[1], image_channels), dtype="float32"), axis=0)
        image_array = np.append(image_array, np.ones(shape=(image_array.shape[0], image_height+image_width, image_channels), dtype="float32"), axis=1)
        # crop unnecessary pixels
        image_array = image_array[:image_height, :image_width, :]
        image_array = np.asarray(image_array).astype("float32").reshape(image_height, image_width, image_channels)

    # resize image
    image_array = tf.image.resize(image_array, size=[image_height, image_width], method='bilinear', preserve_aspect_ratio=False, antialias=False,
                          name=None)

    # now image processing done
    image_array = np.asarray(image_array).astype("float32").reshape(image_height, image_width, image_channels)

    return image_array


# image standardization
def standardization(raw_image_array):
    raw_image_array = np.asarray(raw_image_array).astype("float32")
    mean, std = raw_image_array.mean(), raw_image_array.std()
    st_image = (raw_image_array - mean) / std
    st_image = np.asarray(st_image).astype("float32")
    return st_image


# image de-standardization in uint8 [0, 255]
def de_standardization(raw_image_array, rgb=False, bgr=False):
    raw_image_array = np.asarray(raw_image_array)
    # decrease image values, so that they become in the area [-1, 1]
    image_array = (raw_image_array - np.min(raw_image_array)) / (np.max(raw_image_array) - np.min(raw_image_array))
    # crop values in [-1, 1] to avoid errors
    image_array = np.clip(image_array, -1, 1)
    # conversion image array to uint8 using skimage.img_as_ubyte()
    image_array = skimage.util.img_as_ubyte(image_array)
    image_array = np.asarray(image_array).astype("uint8")
    # convert to rgb, if rgb == True
    if rgb:
        image_array = np.asarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)).astype("uint8")
    # convert to bgr, if bgr == True
    if bgr:
        image_array = np.asarray(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)).astype("uint8")
    # return
    return image_array


# create word-to-number and number-to-word dicts
# char_list - lost with all chars (["a","b","c"])
def dict_generation():
    char_list = ["BLANK", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
                 "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ", "<END>"]
    char_to_num = {}
    num_to_char = {}
    # append blank value, needed for ctc_loss function
    char_to_num.update({'UKN': -1})
    num_to_char.update({-1: 'UKN'})
    # iterate all letters and give him value in dict
    for c1, element in enumerate(char_list):
        char_to_num.update({str(element): int(c1)})
        num_to_char.update({int(c1): str(element)})
    # return
    return char_to_num, num_to_char


