# imports
import common_functions
from data_generator import base_data_gen
import transformer_model

import random
import time
import os
import numpy as np
import jiwer
import cv2
from tensorflow import keras
from keras import models
from keras import preprocessing
from keras import Model
from keras import backend as K


# params
path_to_split_data = "<your_path>"

global_image_height = 512
global_image_width = 128
global_image_channels = 1
batch_size = 20
iteration_count = 500
max_len_text = 190

len_of_seq = 512
len_of_element = 64

# create dict
char_to_num, num_to_char = common_functions.dict_generation()

# create generators
train_gen = base_data_gen(path_to_split_data, "train", iteration_count, batch_size,
                          max_len_text, global_image_height, global_image_width, global_image_channels)
val_gen = base_data_gen(path_to_split_data, "val", iteration_count, batch_size,
                        max_len_text, global_image_height, global_image_width, global_image_channels)
test_gen = base_data_gen(path_to_split_data, "test", iteration_count, batch_size,
                         max_len_text, global_image_height, global_image_width, global_image_channels)


# load model
# (with custom layers)
model = models.load_model(filepath="speech_recognition_transformer_9.3.4.5_v3.h5",
                   custom_objects={
                       'TokenEmbedding': transformer_model.TokenEmbedding,
                       'SpeechFeatureEmbedding': transformer_model.SpeechFeatureEmbedding,
                       'TransformerEncoder': transformer_model.TransformerEncoder,
                       'CTCLayer': transformer_model.CTCLayer,
                                   }
                   )
model.summary()


# create a version of the model without input labels and output ctc losses
prediction_model = keras.models.Model(
    model.get_layer(name="input_image").input, model.get_layer(name="time_distributed").output
)
prediction_model.summary()


y_pred_list = []
y_real_list = []
# test of prediction
# iterate generator
for c1, input_data in enumerate(test_gen):
    for c2, image_array in enumerate(input_data["input_image"]):
        # generate prediction
        y_pred = prediction_model.predict(image_array.reshape(1, global_image_height, global_image_width, global_image_channels))
        # decode 600 outputs into 200 numbers
        y_pred = K.ctc_decode(y_pred, input_length=np.ones(shape=(1,)) * len_of_seq,
                                          greedy=True)
        # reshape
        y_pred = y_pred[0][0][:, 0:len_of_seq].numpy()
        # decode numbers to chars
        p_image_label = []
        for char in y_pred[0]:
            p_image_label.append(num_to_char[int(char)])
        # to string
        y_pred = ""
        for char in p_image_label:
            y_pred += "".join(str(char))
        y_pred_list.append(y_pred)

        # decode real text
        y_real = input_data["output_text"][c2]
        p_real_label = []
        for char in y_real:
            p_real_label.append(num_to_char[int(char)])
        # to string
        y_real = ""
        for char in p_real_label:
            y_real += "".join(str(char))
        y_real_list.append(y_real)

        # decode image
        dec_image_array = common_functions.de_standardization(image_array)

        # show
        print(f"prediction: {y_pred}")
        print(f"real:       {y_real}")
        cv2.imshow('Look', dec_image_array)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        time.sleep(0.01)

    # when all 1300 test images has gone
    if c1 == 65:
        # wer score (Calculate word error rate on all test data)
        wer_score = jiwer.wer(y_real_list, y_pred_list)
        print(f"wer score:  {wer_score}")
        break


