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


@app.route('/')
def hello():
    return "hello, this is test"


@app.route('/color', methods=['POST'])
def color_controller():
    rgb1 = hex_to_rgb(request.form.get('rgb1')[1:])
    rgb2 = hex_to_rgb(request.form.get('rgb2')[1:])
    rgb3 = hex_to_rgb(request.form.get('rgb3')[1:])
    rgb4 = hex_to_rgb(request.form.get('rgb4')[1:])
    # to 2d array
    col_rgb = [rgb1+rgb2+rgb3+rgb4]
    # this is unnecessary but okay-ish if you want to format
    # np.reshape(col_rgb, (1, -1)).astype(float)

    predictions = MODEL_COLOR.predict(col_rgb)
    skin_tone = parse_tone(predictions[:, :3].argmax(axis=1))
    season = parse_season(predictions[:, 3:].argmax(axis=1))
    print(skin_tone)
    print(season)
    response_obj = {'tone': skin_tone, 'season': season}

    return jsonify(response_obj)


@app.route('/tone', methods=['POST'])
def tone_controller():
    image = request.files['gambar']

    response_obj = {'tone': predict_image(image)}

    return jsonify(response_obj)


def hex_to_rgb(hex_value):
    return list(int(hex_value[i:i+2], 16) for i in (0, 2, 4))


def parse_tone(index):
    if index == 0:
        return "white"
    elif index == 1:
        return "olive"
    elif index == 2:
        return "darkbrown"


def parse_season(index):

    if index == 0:
        return "summer"
    elif index == 1:
        return "winter"
    elif index == 2:
        return "autumn"
    elif index == 3:
        return "spring"


def predict_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    print(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255
    val = MODEL_PIC.predict(img_array)
    if np.argmax(val) == 0:
        return "Olive"
    elif np.argmax(val) == 1:
        return "White"
    else:
        return "Brown"


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
