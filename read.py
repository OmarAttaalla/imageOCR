from asyncio.windows_events import NULL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import os

from os.path import exists

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle

import math

Window = NULL

def pass_window(window):
    print("Passed Window")
    global Window
    Window = window

characters = set()

with open('characters.txt','rb') as f:
   characters = pickle.load(f)

class DataImage:
    def __init__(self, imagePath):
        self.LoadedImage = Image.open(imagePath)
        self.ImagePath = imagePath
        self.CroppedImages = []
        self.width, self.height = self.LoadedImage.size
    def Create_Crops(self, numChars, numLines, denseRead):
        totalWidth, totalHeight = self.LoadedImage.size
        numRows = numLines
        numColumns = numChars
        widthAdjustment = 0 # If we have a fractional width / height, we want to make this 1
        heightAdjustment = 0

        if denseRead:
            resizeDims = (551,917)
        else:
            resizeDims = (33,55)

        if (math.ceil(totalWidth / numChars) - (totalWidth / numChars) > 0):
            widthAdjustment = 1
        if (math.ceil(totalHeight / numLines) - (totalHeight / numLines) > 0):
            heightAdjustment = 1
            
        self.widths = math.ceil(totalWidth / numChars)
        self.heights = math.ceil(totalHeight / numLines)
        self.numRows = numRows
        self.numColumns = numColumns
        for i in range (numRows):
            for q in range(numColumns):
                left = q*(self.widths - widthAdjustment)
                top = i*(self.heights - heightAdjustment)
                right = left + self.widths
                bottom = top + self.heights
                newCrop = self.LoadedImage.crop((left,top,right,bottom))
                newCrop = newCrop.resize(resizeDims)
                newCrop.save("TempCrops\\" + str(i) + "-" + str(q) + ".png")
                newDataImage = DataImage("TempCrops\\" + str(i) + "-" + str(q) + ".png")
                self.CroppedImages.append(newDataImage)
                if Window:
                    Window['Progess'].update("Creating Crop: " + str(q + i*100) + '/' + str(numRows * numColumns))

labels = np.load("Labels.npy", allow_pickle=True)
max_length = max([len(label) for label in labels])


def encode_single_sample(DataImage):
    # 1. Read image 
    img_path = DataImage.ImagePath
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [DataImage.height, DataImage.width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 7. Return a dict as our model is expecting two inputs
    return img
# Mapping characters to integers

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)



def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        print(res, "-- res")
        if res == '[UNK]':
            res = 'v'
        output_text.append(res)
    return output_text


def start_read(imagedir, numChars, numLines, Dense_Read):
    nline = 0
    totalText = ""

    numChars = int(numChars)
    numLines = int(numLines)

    if Window:
        Window['Progess'].update("Loading Image...")
    MainImage = DataImage(imagedir)

    if Window:
        Window['Progess'].update("Creating Crops...")
    MainImage.Create_Crops(numChars, numLines, Dense_Read)

    if Window:
        Window['Progess'].update("Loading Model...")

    if Dense_Read: #use Dense NN Model
        newmodel = keras.models.load_model("nnModel.h5", compile=False)
    else:
        newmodel = keras.models.load_model("nnModel.h5", compile=False)

    for p in range(len(MainImage.CroppedImages)):
            print(p, " -- P")
            if Window:
                Window['Progess'].update("Reading Characters: " + str(p) + "/" + str(len(MainImage.CroppedImages)))
            imageToRead = encode_single_sample(MainImage.CroppedImages[p])
            print(MainImage.CroppedImages[p].ImagePath, "-- Path")
            imageToRead = np.expand_dims(imageToRead, axis=0)
            pred = newmodel.predict(imageToRead)
            totalText = totalText + decode_prediction(pred)[0]
            nline = nline + 1
            if (nline >= MainImage.numColumns):
                nline = 0
                totalText = totalText + "\n"

    text_file = open("NNResults5.txt", "w")
    if Window:
        Window['Progess'].update("Creating Text File...")
    text_file.write(totalText)
    text_file.close()
    if Window:
        Window['Progess'].update("Text File Created at: " + os.path.abspath(os.getcwd()) + "\\" + "NNResults5.txt")
    return True



        
