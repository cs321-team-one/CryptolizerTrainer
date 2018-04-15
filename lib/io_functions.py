import os
import pandas as pd


def load_csv_xlsx(filepath):
    ext = os.path.splitext(filepath)[1]

    if ext == '.xlsx':
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise FileNotFoundError('This program can only read a .xlsx/.csv/.json file.')

    return df


def read_data(filepath):
    """
    Reads data from a file (xlsx/csv/json) and extracts inner links within websites
    :param filepath:
    :return:
    """
    json_path = os.path.splitext(filepath)[0] + '.json'

    if filepath == json_path:
        df = pd.read_json(json_path)
    else:
        df = load_csv_xlsx(filepath)

    return df


def write_file(json_data, filename):
    with open(filename, 'w') as f:
        f.write(json_data)
