import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import os
from sklearn.externals import joblib


def init():
    root = Path(os.path.realpath(__file__)).resolve().parents[0]
    news_model_path = str(root / 'news_model.h5')
    news_model = load_model(news_model_path)
    print(f"Loaded News Model from disk: {news_model_path}")

    price_model_path = str(root / 'price_model.h5')
    price_model = load_model(price_model_path)
    print(f'Loaded Price Model from disk: {price_model_path}')

    price_scaler_path = str(root / 'scaler.save')
    price_scaler = joblib.load(price_scaler_path)
    print(f'Loaded Price Scaler from disk: {price_scaler_path}')

    graph = tf.get_default_graph()

    return news_model, price_model, price_scaler, graph
