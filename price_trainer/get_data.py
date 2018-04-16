
import json
import numpy as np
import os
import pandas as pd
import requests

DATA_PATH = 'data/bitcoin2015to2017.csv'
H5_FILEPATH = 'data/bitcoin2015to2017_close.h5'


class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, N, K, sliding_window=True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)

            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1, 1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


def download_data():
    # connect to poloniex's API
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1356998100&end=9999999999&period=300'

    # parse json returned from the API to Pandas DF
    js = requests.get(url)
    d = json.loads(js.text)
    df = pd.DataFrame(d)

    original_columns = [u'close', u'date', u'high', u'low', u'open']
    new_columns = ['Close', 'Timestamp', 'High', 'Low', 'Open']
    df = df.loc[:, original_columns]
    df.columns = new_columns
    df.to_csv(DATA_PATH, index=None)


def process_data(csv_file, past_window, future_window):
    columns = ['Close']
    df = pd.read_csv(csv_file)
    time_stamps = df['Timestamp']
    df = df.loc[:, columns]

    original_df = pd.read_csv(csv_file).loc[:, columns]
    file_name = H5_FILEPATH

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # normalization
    for c in columns:
        df[c] = scaler.fit_transform(df[c].values.reshape(-1, 1))

    # Features are input sample dimensions(channels)
    A = np.array(df)[:, None, :]
    original_A = np.array(original_df)[:, None, :]
    time_stamps = np.array(time_stamps)[:, None, None]

    # Make samples of temporal sequences of pricing data (channel)
    ps = PastSampler(past_window, future_window, sliding_window=False)
    B, Y = ps.transform(A)
    input_times, output_times = ps.transform(time_stamps)
    original_B, original_Y = ps.transform(original_A)

    import h5py
    with h5py.File(file_name, 'w') as f:
        f.create_dataset("inputs", data=B)
        f.create_dataset('outputs', data=Y)
        f.create_dataset("input_times", data=input_times)
        f.create_dataset('output_times', data=output_times)
        f.create_dataset("original_datas", data=np.array(original_df))
        f.create_dataset('original_inputs', data=original_B)
        f.create_dataset('original_outputs', data=original_Y)


def get_data(overwrite=False, past_window=256, future_window=16, h5_outpath=H5_FILEPATH):
    if not os.path.exists(DATA_PATH) or overwrite:
        download_data()

    if not os.path.exists(h5_outpath) or overwrite:
        process_data(DATA_PATH, past_window, future_window)
