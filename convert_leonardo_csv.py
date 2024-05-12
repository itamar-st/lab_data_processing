import pandas as pd
import numpy as np


def convert_leonardo_csv(file_path):
    df = pd.read_csv(file_path)
    while True:
        df['diff'] = df['Dev1/port0/line7 S output'] - df['Tone S output']
        df['diff2'] = df['diff'] - df['diff'].shift(1)
        diff = df['diff2']
        indices1 = diff[diff < -2].index
        indices2 = diff[diff > 2].index
        indices = indices1.tolist() + indices2.tolist()
        # for index, row in df.iterrows():
        #     if abs(row['diff2']) > 2:
        #         df.at[index, 'Tone S output'] = float('nan')  # or use "" for an empty string

        # Shift indices to account for the added rows
        # indices += np.arange(len(indices))

        # Insert blank rows only for 'Tone S output' column
        if not indices:
            break
        idx = indices[0]
        if idx in indices1:
            col_to_change = 'Tone S output'
            correct_column = 'Dev1/port0/line7 S output'
            amount_of_change = -0.5
        else:
            col_to_change = 'Dev1/port0/line7 S output'
            correct_column = 'Tone S output'
            amount_of_change = 0.5
        df.loc[idx + 1:, col_to_change] = df.loc[idx:, col_to_change].shift(1)
        df.at[idx, col_to_change] = df.at[idx, correct_column] + amount_of_change  # Set the current index value to NaN

    print(df)

if __name__ == '__main__':
    path = "C:\\Users\\itama\\Downloads\\test_data_02.csv"
    convert_leonardo_csv(path)

