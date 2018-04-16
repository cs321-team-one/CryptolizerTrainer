import json
import os
import sys
from pathlib import Path

import h5py
import pandas as pd
import requests
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


PARENT_DIR = Path(os.path.realpath(__file__)).resolve().parents[0]

sys.path.append(str(PARENT_DIR))
sys.path.append(str(Path(os.path.realpath(__file__)).resolve().parents[1]))

from price_trainer import get_data

MODEL_OUTPATH = str(PARENT_DIR / 'data/bitcoin2015to2017_close_GRU')
H5_FILEPATH = str(PARENT_DIR / 'data/bitcoin2015to2017_close.h5')
SCALER_FILEPATH = str(PARENT_DIR / 'data/scaler.save')

OVERWRITE_MODEL = False
REFRESH_DATA = False

PAST_WINDOW = 256
FUTURE_WINDOW = 16

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_ai(h5_data_path):
    with h5py.File(''.join([h5_data_path]), 'r') as hf:
        datas = hf['inputs'].value
        labels = hf['outputs'].value

    step_size = datas.shape[1]
    units = 50
    batch_size = 8
    nb_features = datas.shape[2]
    epochs = 20
    output_size = 16
    # split training validation
    training_size = int(0.8 * datas.shape[0])
    training_datas = datas[:training_size, :]
    training_labels = labels[:training_size, :, 0]
    validation_datas = datas[training_size:, :]
    validation_labels = labels[training_size:, :, 0]

    # build model
    model = Sequential()
    model.add(GRU(units=units, input_shape=(step_size, nb_features), return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(Activation('relu'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size,
              validation_data=(validation_datas, validation_labels), epochs=epochs)

    model.save(MODEL_OUTPATH)


def load_new_data(
        url='https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=300&aggregate=3&e=CCCAGG'):
    js = requests.get(url)
    d = json.loads(js.text)
    df = pd.DataFrame(d['Data'])
    return df


def main():
    if not os.path.exists(SCALER_FILEPATH) or REFRESH_DATA:
        get_data.get_data(REFRESH_DATA, h5_outpath=H5_FILEPATH)

        with h5py.File(''.join([H5_FILEPATH]), 'r') as hf:
            original_datas = hf['original_datas'].value

        scaler = MinMaxScaler()
        scaler.fit(original_datas[:, 0].reshape(-1, 1))
        joblib.dump(scaler, SCALER_FILEPATH)

    else:
        print('Loading saved scaler')
        scaler = joblib.load(SCALER_FILEPATH)

    if OVERWRITE_MODEL:
        train_ai(H5_FILEPATH)

    model = load_model(str(PARENT_DIR / 'data/GRN_model'))

    df = load_new_data()
    columns = ['close']
    df = df.loc[:, columns]
    df = scaler.transform(df)

    df_np = np.array(df)[len(df) - PAST_WINDOW:]
    df_np = np.expand_dims(df_np, axis=0)

    predicted_b = scaler.inverse_transform(model.predict(df_np))
    print(predicted_b)


if __name__ == "__main__":
    main()
