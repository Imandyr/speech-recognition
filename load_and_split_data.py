# in this script, we load audio files, create spectrogram, split and save them to train-val-test directories


# imports
import common_functions

import time
import os
import numpy as np
import shutil
import pandas as pd
import librosa
import imageio


# params
# image params
global_image_height = 512
global_image_width = 128
global_image_channels = 1
# data params
base_data_path = "<your_path>"
split_data_save_path = "<your_path>"


# a function that return a spectrogram of a single audio file
def load_and_process_file(path_to_file, img_h, img_w, img_c):
    # load audio values
    clip, sample_rate = librosa.load(path_to_file, sr=None)
    clip = np.asarray(clip).astype("float32")
    # create melspectrogram
    spectrogram = librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_fft=2040, hop_length=512)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # processing melspectrogram image
    spectrogram = common_functions.de_standardization(spectrogram)
    spectrogram = common_functions.image_processing(spectrogram, img_h, img_w, img_c, rotate=True, fill_image=True)
    spectrogram = common_functions.de_standardization(spectrogram)

    # return processed image
    return spectrogram


# function that converts all audio files into their spectrograms
def load_and_split_data(base_path, target_path, image_to_save_len, img_h, img_w, img_c):
    # try to delete previous files, because we overwrite them with new ones
    try:
        shutil.rmtree(f"{target_path}/train")
        shutil.rmtree(f"{target_path}/val")
        shutil.rmtree(f"{target_path}/test")
        time.sleep(5)
    except:
        pass

    # create directories
    os.makedirs(f"{target_path}/train", exist_ok=False)
    os.makedirs(f"{target_path}/val", exist_ok=False)
    os.makedirs(f"{target_path}/test", exist_ok=False)

    # create csv files for train\val\test datasets
    train_csv = open(target_path + "/train_data.csv", "w", encoding="utf-8")
    val_csv = open(target_path + "/val_data.csv", "w", encoding="utf-8")
    test_csv = open(target_path + "/test_data.csv", "w", encoding="utf-8")
    # write first string with columns names
    train_csv.write("File_name|Text" + "\n")
    val_csv.write("File_name|Text" + "\n")
    test_csv.write("File_name|Text" + "\n")


    # load csv with file names and values
    csv_data = pd.read_csv(f"{base_path}/metadata.csv", sep="|")
    print(csv_data)
    image_names = np.asarray(csv_data[["File_name"]]).astype("str")
    image_labels = np.asarray(csv_data[["Text_2"]]).astype("str")
    csv_data = []

    # arrays for data
    images_array = []
    labels_array = []
    # counts for data
    train_image_count = 1
    val_image_count = 1
    test_image_count = 1
    max_len_of_text = 0
    # process all data
    # iterate all file names
    for c1, image_name in enumerate(image_names):
        # create image path
        image_path = f"{base_path}/wavs/{image_name[0]}.wav"
        # create spectrogram image
        image_array = load_and_process_file(image_path, img_h, img_w, img_c)
        # take image text transcription
        label_array = image_labels[c1]
        # calculate max len of text
        if len(label_array[0]) > max_len_of_text:
            max_len_of_text = len(label_array[0])
            print(f"max_len: {max_len_of_text}")

        # append
        images_array.append(image_array)
        labels_array.append(label_array)

        # when we have image_to_save_len images
        if c1 % image_to_save_len == 0:
            print(c1)

            # to numpy arrays
            images_array = np.asarray(images_array).astype("uint8").reshape(len(images_array), img_h, img_w, img_c)
            labels_array = np.asarray(labels_array).astype("str")

            # shuffle
            indices = np.arange(images_array.shape[0])
            np.random.shuffle(indices)
            images_array = images_array[indices]
            labels_array = labels_array[indices]

            # split to datasets
            train_length = int(int(images_array.shape[0]) * 0.8)
            val_length = int(int(images_array.shape[0]) * 0.1)
            train_images = images_array[:train_length]
            val_images = images_array[train_length:train_length+val_length]
            test_images = images_array[train_length+val_length:]
            images_array = []
            train_labels = labels_array[:train_length]
            val_labels = labels_array[train_length:train_length+val_length]
            test_labels = labels_array[train_length+val_length:]
            labels_array = []

            # save images to .png, names and text to .csv
            for c1, train_image in enumerate(train_images):
                imageio.imwrite(f"{target_path}/train/spec_image_{train_image_count}.png", train_image)
                train_csv.write(f"spec_image_{train_image_count}.png|{train_labels[c1][0]}" + "\n")
                train_image_count += 1

            for c1, val_image in enumerate(val_images):
                imageio.imwrite(f"{target_path}/val/spec_image_{val_image_count}.png", val_image)
                val_csv.write(f"spec_image_{val_image_count}.png|{val_labels[c1][0]}" + "\n")
                val_image_count += 1

            for c1, test_image in enumerate(test_images):
                imageio.imwrite(f"{target_path}/test/spec_image_{test_image_count}.png", test_image)
                test_csv.write(f"spec_image_{test_image_count}.png|{test_labels[c1][0]}" + "\n")
                test_image_count += 1

            # re-create arrays
            images_array = []
            labels_array = []

    # close csv files
    train_csv.close()
    val_csv.close()
    test_csv.close()

    # write max len of text
    max_len_of_text_txt = open(target_path + "/max_len_of_text.txt", "w", encoding="utf-8")
    max_len_of_text_txt.write(str(max_len_of_text))
    max_len_of_text_txt.close()



# use it
load_and_split_data(base_data_path, split_data_save_path, 1000,
                    global_image_height, global_image_width, global_image_channels)
