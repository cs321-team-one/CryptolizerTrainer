# for you automatically.
# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request, jsonify
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
# for matrix math
import numpy as np
# for importing our keras model
import keras.models
# for regular expressions, saves time dealing with string data
import re

import sys
import os
from lib.functions import char_preproc_x
from model.load import *

app = Flask(__name__)
global model, graph

# initialize these variables
model, graph = init()


# SINGLE PREDICTION:
# model = keras.models.load_model(MODEL_PATH)
#
# title = "Tiny US soft drinks firm changes name to cash"
# to_predict = [title]
#
# print(title)
#
# x = functions.char_preproc_x(to_predict)
#
# y = model.predict(x)
#
# y = [1 if a[0] >= a[1] else 0 for a in y]
# print(y)


@app.route('/')
def index():
    return "hello world"


@app.route('/news', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        x = char_preproc_x(data)

        print(data)

        with graph.as_default():
            prediction = model.predict(x)
            response = np.argmax(prediction, axis=1).tolist()
            print(response)
            return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 4444))
    app.run(host='0.0.0.0', port=port)
