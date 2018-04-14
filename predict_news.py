import keras
import numpy as np

import functions
from lib.get_training_data import get_training_data

INFRINGING_THRESHOLD = 0.5

CACHE_PATH = './cached_data'
MODEL_PATH = f'{CACHE_PATH}/news_model.h5'

X_COLUMN = 'title'
Y_COLUMN = 'price_change'


def main():
    df = get_training_data()

    model = keras.models.load_model(MODEL_PATH)
    x = functions.char_preproc_x(df[X_COLUMN])
    y = model.predict(x)

    y = [1 if a[0] >= a[1] else 0 for a in y]

    df[Y_COLUMN] = np.array(y)

    print(df)


main()
