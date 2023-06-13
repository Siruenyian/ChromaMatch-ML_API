import requests
from flask import Flask, render_template, jsonify, request, redirect
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

MODEL_COLOR = keras.models.load_model('Model/color_to_tone.h5')
MODEL_PIC = keras.models.load_model('Model/pic_to_tone.h5')

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


# basic endpoint for /
@app.route('/')
def hello():
    return "hello, this is test"


# independent endpoint for color->tone, season
@app.route('/color', methods=['POST'])
def color_controller():
    # get color in hex format from input
    rgb1 = request.form.get('rgb1')[1:]
    rgb2 = request.form.get('rgb2')[1:]
    rgb3 = request.form.get('rgb3')[1:]
    rgb4 = request.form.get('rgb4')[1:]
    # concat and to 2d array and convert to hex
    col_rgb = [hex_to_rgb(rgb1)+hex_to_rgb(rgb2) +
               hex_to_rgb(rgb3)+hex_to_rgb(rgb4)]
    # this is unnecessary but okay-ish if you want to format
    # np.reshape(col_rgb, (1, -1)).astype(float)

    # make predcition
    tone_col, season_col = color_to_tone(col_rgb)
    # construct response dictionary
    response_obj = {'tone': tone_col, 'season': season_col}

    return jsonify(response_obj)


# independent endpoint for image->tone
@app.route('/tone', methods=['POST'])
def tone_controller():
    # get file from request
    image = request.files['gambar']
    # make prediction
    image_prediction = image_to_tone(image)
    # construct response dictionary
    response_obj = {'tone': image_prediction}
    return jsonify(response_obj)


# compound endpoint for image->tone & color->tone,season
@app.route('/predict', methods=['POST'])
def prediction_controller():
    # get files from request
    image = request.files['gambar']
    rgb1 = request.form.get('rgb1')[1:]
    rgb2 = request.form.get('rgb2')[1:]
    rgb3 = request.form.get('rgb3')[1:]
    rgb4 = request.form.get('rgb4')[1:]
    # concat and to 2d array and convert to hex
    col_rgb = [hex_to_rgb(rgb1)+hex_to_rgb(rgb2) +
               hex_to_rgb(rgb3)+hex_to_rgb(rgb4)]
    # make prediction
    tone_img = image_to_tone(image)
    tone_col, season_col = color_to_tone(col_rgb)
    # construct response dictionary
    response_obj = {'tone_from_image': tone_img,
                    'tone_from_color': tone_col, 'season_from_color': season_col}

    return jsonify(response_obj)


# function to convert hex to rgb
def hex_to_rgb(hex_value):
    return list(int(hex_value[i:i+2], 16) for i in (0, 2, 4))


# function to parse tone from prediction array
def parse_tone(index):
    if index == 0:
        return "white"
    elif index == 1:
        return "olive"
    elif index == 2:
        return "darkbrown"


# function to parse season from prediction array
def parse_season(index):
    if index == 0:
        return "summer"
    elif index == 1:
        return "winter"
    elif index == 2:
        return "autumn"
    elif index == 3:
        return "spring"


# utility function to make prediction image->tone
def image_to_tone(image_file):
    # open file
    img = Image.open(image_file)
    # resize image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    # expand dimentions of array to fit the model
    img_array = np.expand_dims(img_array, axis=0)
    # /255 to normalize image
    img_array = img_array/255
    # make prediction from loaded model
    val = MODEL_PIC.predict(img_array)
    img.close()
    # directly parse prediction
    if np.argmax(val) == 0:
        return "Olive"
    elif np.argmax(val) == 1:
        return "White"
    else:
        return "Brown"


# utility function to make prediction color->tone, season
def color_to_tone(col_rgb):
    # make prediction from rgb color
    predictions = MODEL_COLOR.predict(col_rgb)
    # parse prediction using utility function
    skin_tone = parse_tone(predictions[:, :3].argmax(axis=1))
    season = parse_season(predictions[:, 3:].argmax(axis=1))
    # return results
    return skin_tone, season


# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0')
