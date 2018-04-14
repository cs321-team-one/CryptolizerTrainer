import argparse
import os
import pandas as pd
import keras
import numpy as np

import functions
import io_functions

INFRINGING_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description='Predict URLs from the file as infringing or not.')
    parser.add_argument('input', nargs=1, type=argparse.FileType('r'), help='The file that contains a list of data '
                                                                            'with URLs that would be used for '
                                                                            'classification.')
    parser.add_argument('-o', '--output', help='The output filename.', required=True)
    parser.add_argument('-m', '--model', help='The filepath of the trained model.', required=True)
    parser.add_argument('-x', '--x-column', help='The column/field name that corresponds to the desired column of the '
                                                 'data to use for prediction.',
                        required=True)

    args = parser.parse_args()
    infile = args.input[0].name
    outfile = args.output
    modelpath = args.model

    df = io_functions.read_data(infile)

    model = keras.models.load_model(modelpath)
    x = functions.char_preproc_x(df[args.x_column])
    y = model.predict(x)

    y = [1 if a[0] >= a[1] else 0 for a in y]

    df['infringing_robot'] = np.array(y)
    ext = os.path.splitext(args.output)[1]

    if ext == '.xlsx':
        writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
    else:
        if ext == '.csv':
            out_data = df.to_csv()
        else:
            out_data = df.to_json(orient='records')
        io_functions.write_file(out_data, outfile)


main()
