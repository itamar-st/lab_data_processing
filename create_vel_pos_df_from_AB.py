import os
import pandas as pd
import json


def create_vel_pos_df_from_AB(path_of_directory):
    AB_lickport_record_df, Reward_df, TrialTimeline_df = create_df(
        path_of_directory)
    # File name for CSV
    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge
    AB_lickport_record_df = add_trial_num_to_raw_data(AB_lickport_record_df, TrialTimeline_df)

    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']

    AB_lickport_record_df['lickport_signal'] = AB_lickport_record_df['lickport_signal'].round(decimals=0)
    AB_lickport_record_df['A_signal'] = AB_lickport_record_df['A_signal'].round(decimals=0)
    AB_lickport_record_df['B_signal'] = AB_lickport_record_df['B_signal'].round(decimals=0)
    AB_lickport_record_df.loc[AB_lickport_record_df["lickport_signal"] >= 1, "lickport_signal"] = 1
    AB_lickport_record_df.loc[AB_lickport_record_df["A_signal"] >= 1, "A_signal"] = 1
    AB_lickport_record_df.loc[AB_lickport_record_df["B_signal"] >= 1, "B_signal"] = 1

    # create the vel and pos df from an existing file or create from the A_B data
    create_vel_pos_df(AB_lickport_record_df, Reward_df, TrialTimeline_df, path_of_directory)


def create_vel_pos_df(AB_lickport_record_df, Reward_df, TrialTimeline_df, path_of_directory):
    # Check if the file already exists
    vel_pos_file_path = path_of_directory + "\\vel_pos_from_AB.csv"
    vel_pos_with_ITI_file_path = path_of_directory + "\\vel_pos_from_AB_with_ITI.csv"
    if not os.path.exists(vel_pos_file_path) and not os.path.exists(vel_pos_with_ITI_file_path):
        # get the velocity and position by the A_B data
        vel_from_AB_df_with_ITI = extract_vel_pos_from_AB(AB_lickport_record_df)
        vel_from_AB_df_with_ITI['trial_num'] = vel_from_AB_df_with_ITI['trial_num'].astype(int)
        # remove the data that of the ITI
        vel_from_AB_df, TrialTimeline_df = remove_ITI_data(vel_from_AB_df_with_ITI, TrialTimeline_df, Reward_df)
        # Save the DataFrame to a CSV file
        vel_from_AB_df.to_csv(vel_pos_file_path, index=False)
        vel_from_AB_df_with_ITI.to_csv(vel_pos_with_ITI_file_path, index=False)
    else:
        vel_from_AB_df = pd.read_csv(vel_pos_file_path)
        vel_from_AB_df_with_ITI = pd.read_csv(vel_pos_with_ITI_file_path)
    vel_from_AB_df['trial_num'] = vel_from_AB_df['trial_num'].astype(int)
    vel_from_AB_df_with_ITI['trial_num'] = vel_from_AB_df_with_ITI['trial_num'].astype(int)
    return TrialTimeline_df, vel_from_AB_df, vel_from_AB_df_with_ITI


def add_trial_num_to_raw_data(AB_lickport_record_df, TrialTimeline_df):
    intervals = list(zip(TrialTimeline_df['timestamp'].iloc[:-1], TrialTimeline_df['timestamp'].iloc[1:]))
    bins = [item for sublist in intervals for item in sublist]
    # Assign trial_num based on conditions
    AB_lickport_record_df['trial_num'] = pd.cut(AB_lickport_record_df['timestamp'], bins=bins,
                                                labels=TrialTimeline_df['trial_num'].iloc[:-1], duplicates='drop')
    # change the first one from null
    AB_lickport_record_df['trial_num'][0] = 1
    # complete the final trial rows
    last_trial_num = TrialTimeline_df['trial_num'].iloc[-1]
    AB_lickport_record_df['trial_num'] = AB_lickport_record_df['trial_num'].cat.add_categories(last_trial_num)
    AB_lickport_record_df['trial_num'].fillna(last_trial_num, inplace=True)

    return AB_lickport_record_df


def remove_ITI_data(df, TrialTimeline_df, Reward_df):
    trials_times = TrialTimeline_df['timestamp'].values.tolist()
    reward_times = Reward_df['timestamp_reward_start'].values.tolist()

    group_by_trials = df.groupby('trial_num')
    AB_without_ITI = pd.DataFrame()  # Initialize an empty DataFrame

    for trial_index, (group_identifier, group) in enumerate(group_by_trials):
        try:
            # Filter the group for the current trial without ITI
            trial_without_ITI = group.loc[
                (group['timestamp_x'] >= trials_times[trial_index]) &
                (group['timestamp_x'] < reward_times[trial_index])
                ]
            # make it start from 0 till the track length
            min_position = trial_without_ITI['position'].iloc[0]
            trial_without_ITI.loc[:, 'position'] -= min_position

            AB_without_ITI = pd.concat([AB_without_ITI, trial_without_ITI], axis=0)
        # started trial without finishing it
        except IndexError:
            print(f"didn't calculate vel-pos for trial index: {trial_index}")
            # Remove rows from TrialTimeline_df based on the condition
            TrialTimeline_df = TrialTimeline_df[TrialTimeline_df['trial_num'] != group_identifier]

    # Reset the index of the concatenated DataFrame
    AB_without_ITI.reset_index(drop=True, inplace=True)
    return AB_without_ITI, TrialTimeline_df


def extract_vel_pos_from_AB(AB_lickport_record_df):
    # Group by every 20 rows and calculate the number of changes and get the first timestamp in each group
    sec_worth_samples = 2000
    number_of_samples = 40  # todo : turn back to 2000/200
    position = [0]
    avg_vel_per_slit_passed = 59.84 * (
            sec_worth_samples / number_of_samples) / 1024  # 59.84 cm Perimeter, 1024 slits, 100 ms=10th of a sec
    vel_from_AB_df = AB_lickport_record_df.groupby(AB_lickport_record_df.index // number_of_samples).apply(
        lambda group: pd.Series({
            'Avg_velocity1': (group['A_signal'].diff() == 1).sum() * avg_vel_per_slit_passed * calculate_direction
            (group),
            'position': calculate_position_for_trial(group, group['trial_num'].iloc[0] - 1, position),
            'timestamp_x': group['timestamp'].iloc[0],
            'trial_num': group['trial_num'].iloc[0]
        }))
    #  todo: remove the direction calculation outside the scripe and perform only once - save to csv the new vel
    vel_from_AB_df['Avg_velocity'] = vel_from_AB_df['Avg_velocity1'].rolling(window=5, min_periods=1).mean()
    return vel_from_AB_df


def calculate_direction(group):
    prevA = group['A_signal'].iloc[0]  # Previous A channel value
    x = 0  # Encoder direction (0 = clockwise, 1 = counterclockwise)
    # Loop through input channels
    for _, row in group.iterrows():
        currA = row['A_signal']
        currB = row['B_signal']

        if abs(currA - prevA) == 1 and currA == 1:  # A channel has changed
            if abs(currA - currB) == 1:  # Encoder is turning in one direction
                # Set direction to clockwise
                return 1
            else:  # Encoder is turning in the other direction
                # Set direction to counterclockwise
                return -1

        prevA = currA
    return x


def calculate_position_for_trial(lickport_trial_merged_df_with_zeros, trial_num, position):
    counter = 0
    prev_A = True
    prev_B = True

    # Iterate over each row
    for j in range(lickport_trial_merged_df_with_zeros.shape[0]):
        a_state = lickport_trial_merged_df_with_zeros['A_signal'].iloc[j]
        b_state = lickport_trial_merged_df_with_zeros['B_signal'].iloc[j]

        if b_state != prev_B:
            counter += (b_state - prev_B) * (1 if a_state else -1)
        elif a_state != prev_A:
            counter += (a_state - prev_A) * (-1 if b_state else 1)

        prev_A = a_state
        prev_B = b_state
    distance_passed = (counter / 1024) * 59.84 / 4
    position[0] += distance_passed

    return position[0]


def create_df(path_of_directory):
    pd.set_option('display.max_columns', None)
    TrialTimeline_df = pd.read_csv(path_of_directory + "\\TrialTimeline.csv")
    Reward_df = pd.read_csv(path_of_directory + "\\Reward.csv")
    # AB_lickport_record_df = pd.read_csv(path_of_directory + "\\A-B_leakport_record.csv")
    AB_lickport_record_df = pd.read_csv(path_of_directory + "\\Raw_A-B_leakport_record.csv")
    return AB_lickport_record_df, Reward_df, TrialTimeline_df


if __name__ == '__main__':
    for entry in os.scandir("C:\\Users\\itama\\Desktop\\virmen_purple"):
        if entry.is_dir():
            session_folder = entry.path
            create_vel_pos_df_from_AB(session_folder)
