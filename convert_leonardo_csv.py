import pandas as pd
import numpy as np
from collections import Counter
from main import add_trial_num_to_raw_data, lickport_processing


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

    unique_reward_times = (df['Dev1/port0/line7 F output'] - df['Dev1/port0/line7 S output']).round(4).unique().tolist()
    if len(unique_reward_times) > 2:
        print(f"unique_reward_times : {unique_reward_times}")

    big_reward_val = 30
    small_reward_val = 8

    small_trial_time_df = pd.DataFrame({
        'timestamp': df['Trial name: 1Tone_small started'].dropna(),
        'reward_size': [small_reward_val] * df['Trial name: 1Tone_small started'].count()
        # List of 1's with the same length as the DataFrame
    })
    big_trial_time_df = pd.DataFrame({
        'timestamp': df['Trial name: 2ToneLarge started'].dropna(),
        'reward_size': [big_reward_val] * df['Trial name: 2ToneLarge started'].count()
        # List of 1's with the same length as the DataFrame
    })
    # Concatenate the DataFrames
    TrialTimeline_df = pd.concat([small_trial_time_df, big_trial_time_df])

    # Sort by 'start'
    TrialTimeline_df = TrialTimeline_df.sort_values(by='timestamp').reset_index(drop=True)
    TrialTimeline_df['trial_num'] = range(1, len(TrialTimeline_df) + 1)

    reward_df = pd.DataFrame({
        'timestamp_reward_start': df['Dev1/port0/line7 S output'].dropna(),
        'timestamp_reward_end': df['Dev1/port0/line7 F output'].dropna(),
        'reward_size': TrialTimeline_df['reward_size'],
        'trial_num': TrialTimeline_df['trial_num']
    })

    lickport_df = pd.DataFrame({
        'timestamp': df['Dev1/ai13 G input'],
    })
    lickport_df = lickport_df.dropna()
    lickport_df = add_trial_num_to_raw_data(lickport_df, TrialTimeline_df)

    print(df)
    lickport_df = pd.merge(lickport_df, TrialTimeline_df, on='trial_num')
    lickport_df['lickport_signal'] = [1] * lickport_df['trial_num'].count()

    # Duplicate the original DataFrame, but shift everything down by one position
    duplicated_df = lickport_df.copy()
    duplicated_df['lickport_signal'] = 0  # Set 'lickport_signal' to 0 for duplicated rows

    # Adjust the index to interleave, original will have integer, duplicates will have half values
    original_index = lickport_df.index
    duplicated_index = original_index + 0.5

    duplicated_df.index = duplicated_index[:-1]  # Exclude the last to avoid exceeding the original df's length

    # Concatenate the original and duplicated DataFrame
    result_df = pd.concat([lickport_df, duplicated_df]).sort_index().reset_index(drop=True)

    reward_time_range = reward_df['timestamp_reward_start'].values.tolist()
    bins = [float('-inf'), 0, 200, float('inf')]
    group_labels = ['below 0%', 'between 0%-100%', 'above 100%']
    lickport_processing(pd.DataFrame(), bins, group_labels, lickport_df, pd.DataFrame(), TrialTimeline_df,
                        reward_time_range, reward_df)


if __name__ == '__main__':
    path = "C:\\Users\\itama\\Downloads\\test_data_02.csv"
    convert_leonardo_csv(path)
