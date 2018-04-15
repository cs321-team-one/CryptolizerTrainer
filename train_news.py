import os

import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Embedding, Dropout, Conv1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import RMSprop

import functions
from lib.get_training_data import get_training_data

FILTERS = 500
LR = 0.0001
EPOCHS = 25
BATCH_SIZE = 16
EARLY_STOPPING, PATIENCE = True, 20
KEEP_OLD_MODEL = False
REGULARIZATION = 0.20

CACHE_PATH = './cached_data'
MODEL_PATH = f'{CACHE_PATH}/news_model.h5'
BEST_WEIGHTS_PATH = f'{CACHE_PATH}/trained_weights.hdf5'


def create_model():
    input_layer = Input(shape=(250,))
    network = Embedding(70, 16, input_length=250)(input_layer)
    network = Conv1D(filters=500, kernel_size=16,
                     strides=1, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.20))(network)
    network = MaxPooling1D(2)(network)
    network = Flatten()(network)
    network = Dense(1024, activation='relu')(network)
    network = Dropout(0.1)(network)
    y_prediction = Dense(2, activation='softmax')(network)
    optimizer = RMSprop(lr=LR)
    model = Model(inputs=input_layer, outputs=y_prediction)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def train_api(data_input, output, x_column, y_column, keep_old_model):
    df = data_input

    df = df[[x_column, y_column]]

    df = df.dropna().drop_duplicates(subset=x_column)

    data = functions.char_preproc(df[x_column], df[y_column], binarize=False)

    if os.path.exists(output) and keep_old_model:
        model = keras.models.load_model(output)
    else:
        model = create_model()
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE)
    save_best_model = keras.callbacks.ModelCheckpoint(BEST_WEIGHTS_PATH,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='auto')

    try:
        model.fit(data.x_train,
                  data.y_train,
                  validation_data=(data.x_test, data.y_test),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  verbose=2,
                  callbacks=[early_stopping, save_best_model] if EARLY_STOPPING else [save_best_model])

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
    training_data = get_training_data()
    train_api(training_data, MODEL_PATH, 'title', 'price_change', False)


if __name__ == "__main__":
    main()
