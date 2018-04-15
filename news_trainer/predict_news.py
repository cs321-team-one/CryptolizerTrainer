import keras
import numpy as np
from pathlib import Path
import sys
import os
import json
import pandas as pd

ROOT = Path(os.path.realpath(__file__)).resolve().parents[1]
sys.path.append(str(ROOT))

from lib import functions
from lib.get_training_data import get_training_data

INFRINGING_THRESHOLD = 0.5

ROOT = Path(os.path.realpath(__file__)).resolve().parents[1]
CACHE_PATH = ROOT / 'cached_data'
# MODEL_PATH = str(ROOT / 'model' / 'news_model.h5')
MODEL_PATH = str(CACHE_PATH / 'news_model.h5')
CACHE_PATH = str(CACHE_PATH)

X_COLUMN = 'text'
Y_COLUMN = 'price_change'


def main():
    # BATCH PREDICTION:
    df = get_training_data(cached_data_path=Path(CACHE_PATH), overwrite_data=False)
    model = keras.models.load_model(MODEL_PATH)
    x = functions.char_preproc_x(df[X_COLUMN])
    y = model.predict(x)
    y = [1 if a[0] >= a[1] else 0 for a in y]
    df[Y_COLUMN] = np.array(y)

    print(df)


main()
