# for you automatically.
# requests are objects that flask handles (get set post, etc)
import requests
from flask import Flask, render_template, request, jsonify
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
# for matrix math
import numpy as np
# for importing our keras model
import keras.models
# for regular expressions, saves time dealing with string data
import re
import pandas as pd
import sys
import os
from lib.functions import char_preproc_x
from model.load import *
import json

app = Flask(__name__)
global news_model, price_model, price_scaler, graph

# initialize these variables
news_model, price_model, price_scaler, graph = init()


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
def predict_news():
    if request.method == 'POST':
        data = request.get_json()
        x = char_preproc_x(data)

        print(data)

        with graph.as_default():
            prediction = news_model.predict(x)
            response = np.argmax(prediction, axis=1).tolist()
            print(response)
            return jsonify(response)


def load_new_data(ticker='BTC'):
    js = requests.get(f'https://min-api.cryptocompare.com/data/histominute?'
                      f'fsym={ticker}'
                      f'&tsym=USD'
                      f'&limit=300'
                      f'&aggregate=3'
                      f'&e=CCCAGG')
    d = json.loads(js.text)
    df = pd.DataFrame(d['Data'])
    return df


@app.route('/price', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        past_window = 256
        data = request.get_json()

        print(f'Prediction price data for {data}')
        if 'ticker' not in data:
            ticker = 'BTC'
        else:
            ticker = data['ticker']

        df = load_new_data(ticker)
        columns = ['close']
        df = df.loc[:, columns]

        df = price_scaler.transform(df)
        df_np = np.array(df)
        df_np = df_np[len(df) - past_window:]
        df_np = np.expand_dims(df_np, axis=0)
        predicted_b = price_scaler.inverse_transform(price_model.predict(df_np))

        response = predicted_b[0].tolist()
        return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 4444))
    app.run(host='0.0.0.0', port=port)
