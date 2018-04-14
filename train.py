import argparse
import os

import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Embedding, Dropout, Conv1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import io_functions
from lib.get_training_data import get_training_data

import functions

# settings ---------------------
# ------------------------------

EMBEDDING = True
TYPE = 'embedding' if EMBEDDING else 'standard'
MODELPATH = 'models/char-conv-' + TYPE + '-{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5'
FILTERS = 500
LR = 0.0001 if EMBEDDING else 0.00001
EPOCHS = 75
BATCH_SIZE = 32
EARLY_STOPPING, PATIENCE = True, 30
KEEP_OLD_MODEL = False
REGULARIZATION = 0.05
BEST_WEIGHTS_PATH = './trained_weights.hdf5'

CONV = [
    {'filters': 500, 'kernel': 16, 'strides': 1, 'padding': 'same', 'reg': REGULARIZATION, 'pool': 2},
    {'filters': 500, 'kernel': 8, 'strides': 1, 'padding': 'same', 'reg': REGULARIZATION, 'pool': 2},
    {'filters': 500, 'kernel': 16, 'strides': 1, 'padding': 'same', 'reg': 0, 'pool': ''}
]


def create_model():
    if EMBEDDING:
        inputlayer = Input(shape=(250,))
        network = Embedding(70, 16, input_length=250)(inputlayer)

    else:
        inputlayer = Input(shape=(250, 70))
        network = inputlayer

    # convolutional layers ---------
    # ------------------------------

    for C in CONV:

        # conv layer
        network = Conv1D(filters=C['filters'], kernel_size=C['kernel'],
                         strides=C['strides'], padding=C['padding'], activation='relu',
                         kernel_regularizer=regularizers.l2(C['reg']))(network)

        if type(C['pool']) != int:
            continue

        # pooling layer
        network = MaxPooling1D(C['pool'])(network)

    # fully connected --------------
    # ------------------------------
    network = Flatten()(network)
    network = Dense(1024, activation='relu')(network)
    network = Dropout(0.1)(network)

    # output
    ypred = Dense(2, activation='softmax')(network)
    optimizer = RMSprop(lr=LR)

    model = Model(inputs=inputlayer, outputs=ypred)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def train_api(data_input, output, x_column, y_column, keep_old_model):
    if not isinstance(data_input, pd.DataFrame):
        infile = data_input[0].name
        df = io_functions.read_data(infile)
    else:
        df = data_input

    df = df[[x_column, y_column]]

    df = df.dropna().drop_duplicates(subset=x_column)

    data = functions.char_preproc(df[x_column], df[y_column], binarize=not EMBEDDING)

    if os.path.exists(output) and keep_old_model:
        model = keras.models.load_model(output)
    else:
        model = create_model()
    print(model.summary())

    # early stopping
    estopping = EarlyStopping(monitor='val_acc', patience=PATIENCE)
    save_best_model = keras.callbacks.ModelCheckpoint(BEST_WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # fit and run ------------------
    # ------------------------------
    try:
        model.fit(data.x_train,
                  data.y_train,
                  validation_data=(data.x_test, data.y_test),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  verbose=2,
                  callbacks=[estopping, save_best_model] if EARLY_STOPPING else [save_best_model])

    except KeyboardInterrupt:
        print("training stopped")
        exit(1)

    model.load_weights(BEST_WEIGHTS_PATH)

    evaluation = model.evaluate(data.x_test, data.y_test)
    print(f'Loss: {evaluation[0]}')
    print(f'Accuracy: {evaluation[1]}')
    model.save(output)

    os.remove(BEST_WEIGHTS_PATH)
    return evaluation


def main():
    # parser = argparse.ArgumentParser(description='Train on data from a .csv/.xlsx/.json file.')
    # parser.add_argument('input', nargs=1, type=argparse.FileType('r'), help='The file that contains a list of data '
    #                                                                         'with URLs that would be used for '
    #                                                                         'classification.')
    # parser.add_argument('-o', '--output', help='The output filepath for the trained model.', required=True)
    # parser.add_argument('-x', '--x-column', help='The column name that corresponds to the URLs of the websites.',
    #                     required=True)
    # parser.add_argument('-y', '--y-column', help='The column name that corresponds to the infringing values.',
    #                     required=True)
    #
    # args = parser.parse_args()
    # train_api(args.input, args.output, args.x_column, args.y_column, KEEP_OLD_MODEL)
    training_data = get_training_data()
    train_api(training_data, './cached_data/news_model.h5', 'title', 'price_change', False)


if __name__ == "__main__":
    main()
