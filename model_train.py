# imports
import common_functions
from data_generator import base_data_gen
import transformer_model

import random
import time
import os
import numpy as np
import itertools
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


# params
path_to_split_data = "<your_path>"

global_image_height = 512
global_image_width = 128
global_image_channels = 1
batch_size = 20
iteration_count = 500
max_len_text = 190

# create dict
char_to_num, num_to_char = common_functions.dict_generation()

# create generators
train_gen = base_data_gen(path_to_split_data, "train", iteration_count, batch_size,
                          max_len_text, global_image_height, global_image_width, global_image_channels)
val_gen = base_data_gen(path_to_split_data, "val", iteration_count, batch_size,
                        max_len_text, global_image_height, global_image_width, global_image_channels)
test_gen = base_data_gen(path_to_split_data, "test", iteration_count, batch_size,
                         max_len_text, global_image_height, global_image_width, global_image_channels)

# build model
len_of_seq = 512
len_of_element = 64
model = transformer_model.build_model(transformer_layers_number=5, t_input_len=len_of_seq, t_embedding_dim=len_of_element,
                                      t_head_size=len_of_element, t_num_heads=2, t_ff_dim=128, t_dropout=0.25,
                                      reshape_form=(len_of_seq, 131072//len_of_seq), element_size=len_of_element, # 8192 32768 131072
                                      softmax_len=len(char_to_num), c_dropout=0.25, f_dropout=0.25, learning_rate=0.001,
                                      img_h=global_image_height, img_w=global_image_width, img_c=global_image_channels)
model.summary()

# callbacks
callbacks_list = [callbacks.ModelCheckpoint("speech_recognition_transformer_9.3.4.5_v3.h5", monitor='val_loss',
                                            save_best_only=True), # , save_weights_only=True
                  callbacks.EarlyStopping(monitor="val_loss", patience=10),
                  callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001),
                  callbacks.TensorBoard(log_dir='tensor_board_dir', write_images=True, write_graph=True,
                                        write_steps_per_second=True)
                  ]

# try to training model
history = model.fit(train_gen, steps_per_epoch=100, epochs=1000, validation_data=val_gen, validation_steps=20,
                    callbacks=callbacks_list)

# some results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss', 'validation_loss'])
plt.show()




