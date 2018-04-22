import pandas as pd
import numpy as np
import re
import string
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer


def cleanup_str(st, numbers=False):
    if type(st) == bytes:
        try:
            st = st.decode('utf-8').strip().lower()
        except UnicodeDecodeError:
            print('unicode error: {}'.format(st))

    if numbers:
        keep = set(string.ascii_lowercase + string.digits + string.punctuation + ' ')
    else:
        keep = set(string.ascii_lowercase + string.punctuation + ' ')

    # clean string
    st = ''.join(x if x in keep else ' ' for x in str(st))
    # rem multiple spaces
    st = re.sub(' +', ' ', st)

    return st


# mapper: cleanup a pd column or list of strings
def cleanup_col(col, numbers=False):
    col = map(lambda x: cleanup_str(x, numbers=numbers), col)
    return list(col)


def binarize_tokenized(x, vocab_len):
    binarizer = LabelBinarizer()
    binarizer.fit(range(vocab_len))
    x = np.array([binarizer.transform(x) for x in x])

    return x


def char_preproc_x(x, vocab_len=70, binarize=False):
    # -----------------------------
    # preproc x's------------------

    # cleanup
    x = cleanup_col(x, numbers=True)
    # split in arrays of characters
    char_arrs = [[x for x in y] for y in x]

    # tokenize
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(char_arrs)

    # token sequences
    seq = tokenizer.texts_to_sequences(x)

    # pad to same length
    seq = pad_sequences(seq, maxlen=250, padding='post', truncating='post', value=0)

    # make to on-hot
    if binarize:
        x = binarize_tokenized(seq, vocab_len)
    else:
        x = seq
    return x


def char_preproc(x, y, vocab_len=70, binarize=False):
    # -----------------------------
    # preproc x's------------------

    # cleanup
    x = cleanup_col(x, numbers=True)
    # split in arrays of characters
    char_arrs = [[x for x in y] for y in x]

    # tokenize
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(char_arrs)

    # token sequences
    seq = tokenizer.texts_to_sequences(x)

    # pad to same length
    seq = pad_sequences(seq, maxlen=250, padding='post', truncating='post', value=0)

    # make to on-hot
    if binarize:
        x = binarize_tokenized(seq, vocab_len)
    else:
        x = seq

    # ----------------------------
    # preproce y's and return data

    # one-hot encode y's
    y = np.array([[1, 0] if x == 1 else [0, 1] for x in y])

    # generate and return final dataset
    data = Dataset(x, y, shuffle=False, testsize=0.1)

    return data


def load_processed_data(filepath, load=False, binarize=True):
    table = None

    if os.path.isfile('data/processed/data-ready.pkl') and load:
        print("data exists - loading")

        with open('data/processed/data-ready.pkl', 'rb') as file:
            data = pickle.load(file)
    else:
        print("reading raw data and preprocessing..")
        table = pd.read_json(filepath)
        table = table[pd.notnull(table['infringing_human'])]
        data = char_preproc(table.text, table.infringing_human, 70, binarize)

        with open('data/processed/data-ready.pkl', 'wb') as file:
            pickle.dump(data, file)

    return data, table


class Dataset:
    def __init__(self, x, y=None, testsize=0.2, shuffle=False):

        lend = len(x)

        if testsize is None:
            self.x_data = x
            if y is not None:
                self.y_data = y

            print('Single dataset of size {}'.format(lend))
        else:
            if shuffle:
                si = np.random.permutation(np.arange(lend))
                x = x[si]
                y = y[si]
                self.si = si

            if type(testsize) == int:
                testindex = testsize
            else:
                testindex = int(testsize * lend)

            self.x_train = x[testindex:]
            self.x_test = x[:testindex]
            self.y_train = y[testindex:]
            self.y_test = y[:testindex]
            self.testindex = testindex

            print('Train size: {}, test size {}'.format(len(self.y_train), len(self.y_test)))
