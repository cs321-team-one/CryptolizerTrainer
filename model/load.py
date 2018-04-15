import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import os


def init():
    root = Path(os.path.realpath(__file__)).resolve().parents[0]
    model_path = str(root / 'news_model.h5')
    loaded_model = load_model(model_path)
    print(f"Loaded Model from disk: {model_path}")

    graph = tf.get_default_graph()

    return loaded_model, graph
