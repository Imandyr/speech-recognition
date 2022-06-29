# imports
import common_functions

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


# base data generator
# char_list - list of all chars
# path_to_data - path to directory with all split data
# name_of_dataset - name of dataset "train"/"val"/"test"
# gen_iteration - number of complete iterations
# steps_per_iteration - number of steps per iteration
# batch_size - number of (images, text) per step
# max_len_text - len of text sequence from pad_sequences
# img_h, img_w, img_c - parameters of output image
def base_data_gen(path_to_data, name_of_dataset, gen_iteration, batch_size, max_len_text, img_h, img_w, img_c):
    char_to_num, num_to_char = common_functions.dict_generation()

    # iteration of iteration
    for iteration in range(gen_iteration):
        # load required csv file
        csv_data = pd.read_csv(f"{path_to_data}/{name_of_dataset}_data.csv", sep="|")
        images_names = np.asarray(csv_data[["File_name"]]).astype("str")
        images_texts = np.asarray(csv_data[["Text"]]).astype("str")
        csv_data = []
        # print(images_names)
        # print(images_texts)
        # shuffle
        indices = np.arange(images_names.shape[0])
        np.random.shuffle(indices)
        images_names = images_names[indices]
        images_texts = images_texts[indices]
        indices = []

        # lists of images and texts
        image_batch = []
        text_batch = []
        # count of data
        count_of_data = 0
        # iterate names
        for c1, file_name in enumerate(images_names):
            # work with image
            # create path to image
            image_path = f"{path_to_data}/{name_of_dataset}/{file_name[0]}"
            # load image
            image_array = preprocessing.image.load_img(image_path)
            image_array = preprocessing.image.img_to_array(image_array)
            # process image
            image_array = common_functions.image_processing(image_array, img_h, img_w, img_c, grayscale=True)
            # image standardization
            image_array = common_functions.standardization(image_array)

            # work with text
            # take text
            image_text = images_texts[c1, 0]
            # cleaning text of unnecessary characters
            image_text = re.sub(r"\n", "", image_text)
            image_text = re.sub(r"[^a-zA-Z ]", "", image_text)
            # swap capital letter to lower versions
            image_text = image_text.lower()

            # mapping characters to integers
            n_image_text = []
            for char in image_text:
                n_image_text.append(char_to_num[str(char)])
            # append special end character
            n_image_text.append(char_to_num["<END>"])
            # reshape and revert
            n_image_text = np.asarray(n_image_text).astype("float32").reshape(1, len(n_image_text))[:, ::-1]

            # make one len to all text
            n_image_text = preprocessing.sequence.pad_sequences(n_image_text, maxlen=max_len_text, value=char_to_num["BLANK"])
            # revert text to normal (this is necessary, because empty values should be at the end)
            n_image_text = np.asarray(n_image_text[::-1]).astype("int64").reshape(1, max_len_text)[:, ::-1]

            # append to batch
            image_batch.append(image_array)
            text_batch.append(n_image_text)
            count_of_data += 1

            # if count of (images, text) == batch_size, return batch of data
            if count_of_data == batch_size:
                # final preparation
                image_batch = np.asarray(image_batch).astype("float32").reshape(batch_size, img_h, img_w, img_c)
                text_batch = np.asarray(text_batch).astype("int64").reshape(batch_size, max_len_text)

                # return
                yield {"input_image": image_batch, "output_text": text_batch}

                # reset
                image_batch = []
                text_batch = []
                count_of_data = 0



