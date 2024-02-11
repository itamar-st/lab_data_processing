import os

from functools import partial
from statistics import mean

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter import messagebox
import csv


def post_processing(path_of_directory, percentage_from_start, percentage_from_end, remove_outliers):
    AB_lickport_record_df, Reward_df, TrialTimeline_df, config_json, sound_df, velocity_df, stats_df, vel_from_AB_df = create_df(
        path_of_directory)
    # File name for CSV

    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge

    AB_lickport_record_df = add_trial_num_to_raw_data(AB_lickport_record_df, TrialTimeline_df)

    init_amount_of_trials = TrialTimeline_df.shape[0]
    trial_num_bottom_threshold = int(
        init_amount_of_trials * (percentage_from_start / 100))  # precentage to num of trials
    trial_num_top_threshold = int(init_amount_of_trials * (percentage_from_end / 100))
    bins = [float('-inf'), trial_num_bottom_threshold, trial_num_top_threshold, float('inf')]
    group_labels = [f'below {percentage_from_start}%', f'between {percentage_from_start}%-{percentage_from_end}%',
                    f'above {percentage_from_end}%']

    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']
    if remove_outliers:  # todo:switch back vel_from_AB_df
        AB_lickport_record_df, Reward_df, TrialTimeline_df, velocity_df, vel_from_AB_df = outliers_removal(AB_lickport_record_df,
                                                                                           Reward_df, TrialTimeline_df,
                                                                                           sound_df, velocity_df, vel_from_AB_df)

    trial_length_results, stats_df = trial_length_processing(stats_df, TrialTimeline_df, bins, group_labels)

    AB_lickport_record_df['lickport_signal'] = AB_lickport_record_df['lickport_signal'].round(decimals=0)
    AB_lickport_record_df['A_signal'] = AB_lickport_record_df['A_signal'].round(decimals=0)
    AB_lickport_record_df['B_signal'] = AB_lickport_record_df['B_signal'].round(decimals=0)
    AB_lickport_record_df.loc[AB_lickport_record_df["lickport_signal"] >= 1, "lickport_signal"] = 1
    AB_lickport_record_df.loc[AB_lickport_record_df["A_signal"] >= 1, "A_signal"] = 1
    AB_lickport_record_df.loc[AB_lickport_record_df["B_signal"] >= 1, "B_signal"] = 1
    lickport_trial_merged_df_with_zeros = pd.merge(AB_lickport_record_df, TrialTimeline_df, on='trial_num')
    # only start lickport activation and finish
    lickport_start_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['lickport_signal'] == 1) &
        (lickport_trial_merged_df_with_zeros['lickport_signal'].shift(1) == 0)]
    lickport_end_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['lickport_signal'] == 0) &
        (lickport_trial_merged_df_with_zeros['lickport_signal'].shift(1) == 1)]


    global trials_time_range
    trials_time_range = TrialTimeline_df['timestamp'].values.tolist()
    global reward_time_range
    reward_time_range = Reward_df['timestamp_reward_start'].values.tolist()

    # Check if the file already exists
    file_path = "vel_pos_from_AB.csv"
    if not os.path.exists(file_path):
        # get the velocity and position by the A_B data
        vel_from_AB_df = extract_vel_pos_from_AB(AB_lickport_record_df, vel_from_AB_df)
        # remove the data that of the ITI
        vel_from_AB_df = remove_ITI_data(vel_from_AB_df)

        # Save the DataFrame to a CSV file
        vel_from_AB_df.to_csv(file_path, index=False)

    # calculate_position(lickport_trial_merged_df_with_zeros, reward_time_range, trials_time_range, vel_from_AB_df)

    lickport_results, stats_df = lickport_processing(stats_df, bins, group_labels, lickport_trial_merged_df_with_zeros,
                                                     TrialTimeline_df, reward_time_range)

    velocity_trial_merged_df = pd.merge(vel_from_AB_df, TrialTimeline_df, on='trial_num')  #todo:switch back
    # velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')

    velocity_results, stats_df = velocity_processing(stats_df, bins, group_labels, velocity_trial_merged_df,
                                                     config_json)

    # create_formatted_file(Reward_df, TrialTimeline_df, lickport_trial_merged_df_with_zeros,
    #                       config_json, lickport_end_df, lickport_start_df,
    #                       path_of_directory, sound_df)

    stats_df.to_csv(path_of_directory + "\\data_for_stats.csv", float_format='%.4f',
                    index=False)  # write the dataframe into a csv

    # plot_velocity_over_position(config_json, velocity_trial_merged_df, 'velocity over position')
    plt.show()
    # all the results from the processing and the number of trials in the session
    final_amount_of_trials = TrialTimeline_df.shape[0]  # without the outliers
    result_dict = {**trial_length_results, **lickport_results, **velocity_results,
                   **{"number of trials": final_amount_of_trials}}

    return result_dict

def calculate_position(lickport_trial_merged_df_with_zeros, reward_time_range, trials_time_range, vel_from_AB_df):
    all_positions = []
    for i in range(0, len(trials_time_range)):
        counter = 0
        prev_A = True
        prev_B = True
        encoder_st = 0
        a_last_state = 0

        AB_without_ITI = lickport_trial_merged_df_with_zeros.loc[
            (lickport_trial_merged_df_with_zeros['timestamp_x'] >= trials_time_range[i])
            & (lickport_trial_merged_df_with_zeros['timestamp_x'] < reward_time_range[i])]

        hundred_ms_window_rows = 200  # num of rows to get 100 ms worth of data

        # Group the DataFrame by a custom function based on the row index
        grouped = AB_without_ITI.iloc[1:].groupby(lambda x: x // hundred_ms_window_rows)

        # Iterate over each group
        for group_index, group in grouped:
            for j in range(group.shape[0]):
                a_state = group['A_signal'].iloc[j]
                b_state = group['B_signal'].iloc[j]

                if b_state != prev_B:
                    counter += (b_state - prev_B) * (1 if a_state else -1)
                elif a_state != prev_A:
                    counter += (a_state - prev_A) * (-1 if b_state else 1)

                prev_A = a_state
                prev_B = b_state
            curr_position = (counter / 1024) * 59.84 / 4

            all_positions.append(curr_position)
    print(f"aaaa mean = {mean(all_positions)}")
    print(f"aaaa std = {numpy.std(all_positions)}")
    # position_df = pd.DataFrame({'position': all_positions})
    vel_from_AB_df['position'] = all_positions

def calculate_position_for_trial(lickport_trial_merged_df_with_zeros, trial_num, position):
    counter = 0
    prev_A = True
    prev_B = True
    # Iterate over each group
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


def extract_vel_pos_from_AB(AB_lickport_record_df, vel_from_AB_df):
    # Group by every 20 rows and calculate the number of changes and get the first timestamp in each group
    sec_worth_samples = 2000
    number_of_samples = 200
    position = [0]
    avg_vel_per_slit_passed = 59.84 * 10 / 1024  # 59.84 cm Perimeter, 1024 slits, 100 ms=10th of a sec
    vel_from_AB_df = AB_lickport_record_df.groupby(AB_lickport_record_df.index // 200).apply(
        lambda group: pd.Series({
            'Avg_velocity1': (group['A_signal'].diff() == 1).sum() * avg_vel_per_slit_passed * calculate_direction
            (group),
            'position': calculate_position_for_trial(group, group['trial_num'].iloc[0]-1, position),
            'timestamp_x': group['timestamp'].iloc[0],
            'trial_num': group['trial_num'].iloc[0]
        }))
    #  todo: remove the direction calculation outside the scripe and perform only once - save to csv the new vel
    vel_from_AB_df['Avg_velocity'] = vel_from_AB_df['Avg_velocity1'].rolling(window=5).mean()
    return vel_from_AB_df


def calculate_direction3(group):
    prevA = group['A_signal'].iloc[0]  # Previous A channel value
    x = 0  # Encoder direction (0 = clockwise, 1 = counterclockwise)

    # Loop through input channels
    for _, row in group.iterrows():
        currA = row['A_signal']
        currB = row['B_signal']

        if currA != prevA:  # A channel has changed
            if currB != currA:  # Encoder is turning in one direction
                x = 1  # Set direction to clockwise
                break
            else:  # Encoder is turning in the other direction
                x = -1  # Set direction to counterclockwise
                break

        prevA = currA

    # print("Encoder direction:", x)
    return x


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
    if group.shape[0] != 200:
        print(group.shape[0])
    return x


def calculate_direction2(AB_lickport_record_df):
    # Calculate the differences between consecutive A and B values
    diff_A = AB_lickport_record_df['A_signal'].diff()
    # diff_B = AB_lickport_record_df['B_signal'].diff()

    # Find the first index where A has changed significantly
    change_index = (abs(diff_A) == 1) & (AB_lickport_record_df['A_signal'] == 1)
    change_index = change_index.idxmax()

    if not change_index == 0:
        # Determine direction based on the difference between A and B at the change point
        x = np.sign(AB_lickport_record_df['A_signal'].iat[change_index] - AB_lickport_record_df['B_signal'].iat[change_index])

        # print("Encoder direction:", x)
        return x
    else:
        print("No significant change in A channel.")


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


def create_df(path_of_directory):
    pd.set_option('display.max_columns', None)
    stats_df = pd.DataFrame()  # for calculating stats
    TrialTimeline_df = pd.read_csv(path_of_directory + "\\TrialTimeline.csv")
    Reward_df = pd.read_csv(path_of_directory + "\\Reward.csv")
    # AB_lickport_record_df = pd.read_csv(path_of_directory + "\\A-B_leakport_record.csv")
    AB_lickport_record_df = pd.read_csv(path_of_directory + "\\Raw_A-B_leakport_record.csv")
    velocity_df = pd.read_csv(path_of_directory + "\\velocity.csv")
    vel_from_AB_df = pd.DataFrame()
    sound_df = pd.read_csv(path_of_directory + "\\SoundGiven.csv")
    config_file = open(path_of_directory + "\\config.json")
    config_json = json.load(config_file)
    config_file.close()
    return AB_lickport_record_df, Reward_df, TrialTimeline_df, config_json, sound_df, velocity_df, stats_df, vel_from_AB_df


def outliers_removal(AB_lickport_record_df, Reward_df, TrialTimeline_df, sound_df, velocity_df, vel_from_AB_df):
    trial_length_std = TrialTimeline_df['trial_length'].std()
    trial_length_mean = TrialTimeline_df['trial_length'].mean()
    threshold = 2.5
    abnormal_trial_length = trial_length_mean + threshold * trial_length_std
    abnormal_rows = TrialTimeline_df['trial_length'] > abnormal_trial_length
    abnormal_trial_nums = TrialTimeline_df.loc[abnormal_rows, 'trial_num'].unique()
    print(f"abnormal trials: {abnormal_trial_nums}")
    TrialTimeline_df = TrialTimeline_df[~TrialTimeline_df['trial_num'].isin(abnormal_trial_nums)]
    AB_lickport_record_df = AB_lickport_record_df[~AB_lickport_record_df['trial_num'].isin(abnormal_trial_nums)]
    Reward_df = Reward_df[~Reward_df['trial_num'].isin(abnormal_trial_nums)]
    velocity_df = velocity_df[~velocity_df['trial_num'].isin(abnormal_trial_nums)]
    # vel_from_AB_df = vel_from_AB_df[~vel_from_AB_df['trial_num'].isin(abnormal_trial_nums)] todo:fix
    sound_df = sound_df[~sound_df['trial_num'].isin(abnormal_trial_nums)]
    return AB_lickport_record_df, Reward_df, TrialTimeline_df, velocity_df, vel_from_AB_df


def plot_velocity_over_position(stats_df, config_json, velocity_trial_merged_df, title, label, graph_color, ax=None):
    velocity_by_position = []
    std_by_position = []
    position_segments = np.linspace(0, int(config_json['db_distance_to_run']), 60, endpoint=True)  # todo *2 instaid of 60
    # all_velocity_by_position_plot(label, position_segments, title, velocity_trial_merged_df)

    # get the mean and std of every trial
    for i in range(len(position_segments) - 1):
        # divide the data by the position of the mouse
        data_by_position = velocity_trial_merged_df.loc[
            (velocity_trial_merged_df['position'] >= position_segments[i])
            & (velocity_trial_merged_df['position'] <= position_segments[i + 1])]
        mean_velocity_by_position = data_by_position['Avg_velocity'].mean()
        std_velocity_by_position = data_by_position['Avg_velocity'].std()
        velocity_by_position.append(mean_velocity_by_position)
        std_by_position.append(std_velocity_by_position)
    # set the points in the middle of the section of the speed range
    position_bins = [(position_segments[i] + position_segments[i + 1]) / 2 for i in range(len(position_segments) - 1)]
    # ax.plot(position_bins,
    #         velocity_by_position,
    #         linestyle='--',
    #         marker='o',
    #         color=graph_color,
    #         label=f'mean velocity {label}')
    # scatter plot with its std
    ax.errorbar(position_bins, velocity_by_position, yerr=std_by_position,
                linestyle='--', marker='o', color=graph_color,
                label=f'mean velocity {label} ± std')
    ax.fill_between(position_bins,
                    np.array(velocity_by_position) - np.array(std_by_position),
                    np.array(velocity_by_position) + np.array(std_by_position),
                    color=graph_color, alpha=0.4, label=f'Error Range {label}')
    # Add text annotations for each bar at the top
    # for x, y, value in zip(position_bins, velocity_by_position, velocity_by_position):
    #     ax.text(x, y, f'{value:.2f}', ha='center', va='bottom', color='black', fontsize=8)

    df = pd.DataFrame()
    df[title + f" reward size {label} :position"] = position_bins
    df[title + f" reward size {label} :velocity"] = velocity_by_position
    df.reset_index(drop=True, inplace=True)

    stats_df = pd.concat([stats_df, df], axis=1)

    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Mean Velocity')
    ax.legend()

    return stats_df


def all_velocity_by_position_plot(label, position_segments, title, velocity_trial_merged_df):
    grouped_by_trials = velocity_trial_merged_df.groupby(velocity_trial_merged_df['trial_num'])
    velocity_trial_merged_df.plot(kind='scatter', x='position', y='Avg_velocity',
                                  title=f'velocity over position all data {title} {label}',
                                  label=f'velocity',
                                  color='pink')
    # trial_velocity_by_position = []
    # all_buffers = []
    # # devide the length of the trace
    # for j, group in enumerate(grouped_by_trials):
    #     for i in range(len(position_segments) - 1):
    #         # divide the data by the position of the mouse
    #         data_by_position = group.loc[
    #             (group['position'] >= position_segments[i])
    #             & (group['position'] <= position_segments[i + 1])]
    #         mean_velocity_by_position = data_by_position['Avg_velocity'].mean()
    #         trial_velocity_by_position.append(mean_velocity_by_position)
    #     all_buffers.append(trial_velocity_by_position)
    # num_of_buffers = len(all_buffers)
    # for n, s in enumerate(all_buffers):
    #     s['order'] = num_of_buffers - n
    #     s.plot(kind='scatter', x='position', y='order', ax=ax3, s=5)


def calc_licks_around_time_event(stats_df, lickport_trial_merged_df_with_zeros, reward_time_range, title):
    all_buffers = []
    length_of_buff = 4  # time buffer around the start of the trial/reward
    # separate the data around each start of a trial
    for i in range(1, len(reward_time_range) - 1):
        buffer_around_trial = lickport_trial_merged_df_with_zeros.loc[
            (lickport_trial_merged_df_with_zeros['timestamp_x'] >= reward_time_range[i] - length_of_buff)
            & (lickport_trial_merged_df_with_zeros['timestamp_x'] <= reward_time_range[i] + length_of_buff)]
        if not buffer_around_trial.empty:
            # decrease the first timestamp so all will start from 0
            buffer_around_trial.loc[:, 'timestamp_x'] = buffer_around_trial['timestamp_x'] - \
                                                        buffer_around_trial['timestamp_x'].iloc[0]
            # take only the activation of the lickport
            buffer_around_trial = buffer_around_trial[(buffer_around_trial['lickport_signal'] == 1) &
                                                      (buffer_around_trial['lickport_signal'].shift(1) == 0)]

            all_buffers.append(buffer_around_trial)
    stats_df = plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title)
    return stats_df


def plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title):
    lick_fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
    # Plot each DataFrame in a loop, vertically spaced
    num_of_buffers = len(all_buffers)

    for i, s in enumerate(all_buffers):
        s['lickport_signal'] = s['lickport_signal'] + num_of_buffers - i
        s.plot(kind='scatter', x='timestamp_x', y='lickport_signal', ax=ax3, s=5)
    ax3.axvline(x=length_of_buff, color='red', linestyle='--')
    ax3.set_title(title + ' -- Licks over time')
    ax3.set_xlabel('time')
    ax3.set_ylabel('start licking')

    all_licks = pd.concat(all_buffers)
    hist_title = f"lickport {length_of_buff} sec around the start of the reward"
    histogram_plot = all_licks['timestamp_x'].plot(kind='hist',
                                                   bins=100,
                                                   ax=ax4,
                                                   label='licks',
                                                   title=hist_title)
    frequencies = get_frequencies(histogram_plot)
    df = pd.DataFrame({title + " frequencies": frequencies})
    stats_df = pd.concat([stats_df, df], axis=1)

    ax4.axvline(x=length_of_buff, color='red', linestyle='--', label='reward start')
    ax4.legend()
    ax3.legend()
    plt.tight_layout()

    return stats_df


def get_frequencies(histogram_plot):
    # Get the patches (bars) from the histogram plot
    patches = histogram_plot.patches
    # Extract the frequencies from the height of each patch
    frequencies = [patch.get_height() for patch in patches]
    return frequencies


def create_formatted_file(Reward_df, TrialTimeline_df, lickport_trial_merged_df_with_zeros, config_json,
                          lickport_end_df, lickport_start_df,
                          path_of_directory, sound_df):
    formatted_file_name = path_of_directory + "\\formatted.csv"
    A_start_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['A_signal'] == 1) &
        (lickport_trial_merged_df_with_zeros['A_signal'].shift(1) == 0)]
    A_end_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['A_signal'] == 0) &
        (lickport_trial_merged_df_with_zeros['A_signal'].shift(1) == 1)]
    B_start_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['B_signal'] == 1) &
        (lickport_trial_merged_df_with_zeros['B_signal'].shift(1) == 0)]
    B_end_df = lickport_trial_merged_df_with_zeros[
        (lickport_trial_merged_df_with_zeros['B_signal'] == 0) &
        (lickport_trial_merged_df_with_zeros['B_signal'].shift(1) == 1)]
    formatted_df = pd.DataFrame([])
    lickport_start_df.reset_index(drop=True, inplace=True)
    lickport_end_df.reset_index(drop=True, inplace=True)
    A_start_df.reset_index(drop=True, inplace=True)
    A_end_df.reset_index(drop=True, inplace=True)
    B_start_df.reset_index(drop=True, inplace=True)
    B_end_df.reset_index(drop=True, inplace=True)
    formatted_df['A_start'] = A_start_df['timestamp_x']
    formatted_df['A_end'] = A_end_df['timestamp_x']
    formatted_df['B_start'] = B_start_df['timestamp_x']
    formatted_df['B_end'] = B_end_df['timestamp_x']
    formatted_df['lick start'] = lickport_start_df['timestamp_x']
    formatted_df['lick end'] = lickport_end_df['timestamp_x']
    formatted_df['trial start'] = TrialTimeline_df['timestamp']
    formatted_df['trial end'] = Reward_df['timestamp_reward_start']
    formatted_df['trial length'] = TrialTimeline_df['trial_length'].round(2)
    formatted_df['sound start'] = sound_df['timestamp']
    formatted_df['sound end'] = sound_df.loc[:, 'timestamp'] + 0.5
    formatted_df['reward start'] = Reward_df['timestamp_reward_start']
    formatted_df['reward end'] = Reward_df['timestamp_reward_end']
    formatted_df['reward size'] = Reward_df['reward_size']
    formatted_df['black room start'] = formatted_df['reward start'] + int(config_json['db_leakport_room_break'])
    formatted_df['black room end'] = formatted_df['black room start'] + int(config_json['db_black_room_break'])
    formatted_df.to_csv(formatted_file_name)


def lickport_processing(stats_df, bins, group_labels, lickport_trial_merged_df_with_zeros, TrialTimeline_df,
                        reward_time_range):
    grouped_lickport_by_trial_precentage = lickport_trial_merged_df_with_zeros.groupby(
        pd.cut(lickport_trial_merged_df_with_zeros['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    results = {}
    print("sum of lickport activations by reward:")
    for condition, group in grouped_lickport_by_trial_precentage:
        if not group.empty:  # for not creating empty figures
            print(f"\t{condition}:")
            grouped_by_reward_type = group.groupby('reward_size')
            colors = ['blue', 'yellow']
            fig, ax = plt.subplots()
            for i, (condition2, reward_group) in enumerate(grouped_by_reward_type):
                sum_of_licks = reward_group['lickport_signal'].sum()
                results["lickport activation" + str(condition) + str(condition2)] = sum_of_licks
                print(f"\t\tCondition- reward size {condition2}: {sum_of_licks}")
                title = f'lickport of trials {condition} Reward Size {condition2}'
                stats_df = calc_licks_around_time_event(stats_df, reward_group, reward_time_range, title)
                # take only the activation of the lickport
                reward_group = reward_group[(reward_group['lickport_signal'] == 1) &
                                            (reward_group['lickport_signal'].shift(1) == 0)]
                reward_group.plot(kind='scatter', x='timestamp_x', y='lickport_signal',
                                  title=f'lickport of trials {condition}%', label=f'Reward Size {condition2}', ax=ax,
                                  color=colors[i])
                df = pd.DataFrame()
                df[title + ' :timestamp'] = reward_group['timestamp_x']
                df.reset_index(drop=True, inplace=True)
                stats_df = pd.concat([stats_df, df], axis=1)
            for timestamp in TrialTimeline_df['timestamp']:
                ax.axvline(x=timestamp, color='red', linestyle='--')
            stats_df = calc_licks_around_time_event(stats_df, lickport_trial_merged_df_with_zeros, reward_time_range,
                                                    f"{str(condition)} all reward sizes")

            print()
    print(f"all reward sizes:\n{grouped_lickport_by_trial_precentage['lickport_signal'].sum()}")
    print("\n\n")

    return results, stats_df


def velocity_processing(stats_df, bins, group_labels, velocity_trial_merged_df, config_json):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Filter 'Avg_velocity' without rows equal to 0f
    velocity_without_zeros = velocity_trial_merged_df.loc[velocity_trial_merged_df['Avg_velocity'] != 0]
    velocity_trial_merged_df_without_ITI = remove_ITI_data(velocity_trial_merged_df)
    non_zero_velocity_without_ITI = remove_ITI_data(velocity_without_zeros)
    # Calculate rolling average
    non_zero_velocity_without_ITI.plot(x='timestamp_x', y='Avg_velocity', ax=ax1, label='Avg_velocity')
    # Set title and labels
    ax1.set_title('Velocity over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Avg_velocity')
    ax1.legend()

    results = {}

    grouped_velocity_by_trial_precentage = non_zero_velocity_without_ITI.groupby(
        pd.cut(non_zero_velocity_without_ITI['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("average velocity by reward:")
    for condition, group in grouped_velocity_by_trial_precentage:
        if not group.empty:  # for not creating empty figures
            print(f"\t{condition}:")
            grouped_by_reward_type = group.groupby('reward_size')
            colors = ['blue', 'green']
            fig, ax = plt.subplots()
            for i, (condition2, reward_group) in enumerate(grouped_by_reward_type):
                results = calc_movement_during_session(results, velocity_trial_merged_df_without_ITI,
                                                       reward_group, condition, condition2)
                mean_vel = reward_group['Avg_velocity'].mean()
                results["avg velocity " + str(condition) + str(condition2)] = mean_vel
                print(
                    f"\t\tCondition- reward size {condition2}: {mean_vel}")  # considers all the data points, including all the 0's
                title = f'velocity of trials {condition}, reward size {condition2}'
                reward_group.plot(x='timestamp_x', y='Avg_velocity',
                                  title=title)
                df = pd.DataFrame()
                df[title + ' :timestamp'] = reward_group['timestamp_x']
                df[title + ' :Avg_velocity'] = reward_group['Avg_velocity']
                df.reset_index(drop=True, inplace=True)
                stats_df = pd.concat([stats_df, df], axis=1)

                stats_df = plot_velocity_over_position(stats_df, config_json, reward_group,
                                                       f'velocity over position {condition} (without 0s)', label=condition2,
                                                       graph_color=colors[i], ax=ax)
                plt.figure()
                histogram_plot = reward_group['Avg_velocity'].plot(kind='hist',
                                                                   bins=50,
                                                                   title=f"speed Distribution: trials {condition}, reward size {condition2}",
                                                                   xlabel='speed(cm/sec)',
                                                                   ylabel='Frequency')
                frequencies = get_frequencies(histogram_plot)
                histogram_df = pd.DataFrame({title: frequencies})
                stats_df = pd.concat([stats_df, histogram_df], axis=1)

            print()
            plt.figure()
            # plot for all the reward types and all trials
            all_reward_vel_hist = group['Avg_velocity'].plot(kind='hist',
                                                             bins=50,
                                                             title=f"speed Distribution: trials {condition}, all reward",
                                                             xlabel='speed(cm/sec)',
                                                             ylabel='Frequency')
            frequencies = get_frequencies(all_reward_vel_hist)
            histogram_df = pd.DataFrame({"all rewards speed Distribution": frequencies})
            histogram_df.reset_index(drop=True, inplace=True)
            stats_df = pd.concat([stats_df, histogram_df], axis=1)
    print(f"all reward sizes:\n{grouped_velocity_by_trial_precentage['Avg_velocity'].mean()}")
    print("\n\n")
    return results, stats_df


def remove_ITI_data(df):
    conditions1 = [
        (df['timestamp_x'] >= start) & (df['timestamp_x'] < end)
        for start, end in zip(trials_time_range, reward_time_range)
    ]
    # Combine the conditions using logical OR
    is_in_range1 = pd.concat(conditions1, axis=1).any(axis=1)
    # Filter the DataFrame based on the combined condition
    df_without_ITI = df[is_in_range1]
    # Calculate the minimum position value for each trial_num group
    min_positions = df_without_ITI.groupby('trial_num')['position'].transform('min')

    # Subtract the minimum position value from the position column
    df_without_ITI['position'] -= min_positions
    return df_without_ITI


def calc_movement_during_session(results, velocity_trial_merged_df, velocity_without_zeros, trial_prec, reward_size):
    movement_during_trial = []
    trials = []
    num_of_samples = velocity_trial_merged_df.shape[0]  # todo : check why graph and print inconsistent
    non_zero_samples = velocity_without_zeros.shape[0]
    print(f"percentage of movement for trials {trial_prec} reward size {reward_size}: {(non_zero_samples/ num_of_samples)*100} %")
    results["percentage of movement for small reward"] = (non_zero_samples / num_of_samples)*100
    group_by_trial_with_zero = velocity_trial_merged_df.groupby('trial_num')
    group_by_trial_without_zero = velocity_without_zeros.groupby('trial_num')
    for trial_num, group_without_zero in group_by_trial_without_zero:
        group_with_zero = group_by_trial_with_zero.get_group(trial_num)

        # percentage of movement during the trial
        movement_during_trial.append((group_without_zero.shape[0] / group_with_zero.shape[0])*100)
        trials.append(trial_num)
    plt.figure()
    plt.scatter(x=trials, y=movement_during_trial, color='orange')
    plt.plot(trials, movement_during_trial, linestyle='-', color='gray')
    plt.title(f'% of movement per trial, trials {trial_prec} reward size: {reward_size}')
    plt.xlabel('Trials')
    plt.ylabel('% of Movement')
    plt.legend(['% of Movement'])
    return results


def trial_length_processing(stats_df, TrialTimeline_df, bins, group_labels):
    grouped_by_trial_precentage = TrialTimeline_df.groupby(
        pd.cut(TrialTimeline_df['trial_num'], bins=bins, labels=group_labels), observed=False)
    print("average trial length by reward:")
    results = {}
    for condition, group in grouped_by_trial_precentage:
        if not group.empty:  # for not creating empty figures
            print(f"\t{condition}:")
            grouped_by_reward_type = group.groupby('reward_size')
            fig, ax = plt.subplots()
            for condition2, reward_group in grouped_by_reward_type:
                mean_trial_length = reward_group['trial_length'].mean()
                results["trial length " + str(condition) + str(condition2)] = mean_trial_length
                print(f"\t\tCondition- reward size {condition2}: {mean_trial_length}")
                print(f"\t\tnumber of trials: {reward_group['trial_num'].count()}")
                title = f'Length of Trials {condition}, Reward Size {condition2}'
                # Plot each reward group's data on the same axis
                reward_group.plot(x='trial_num', y='trial_length', ax=ax,
                                  label=title)

                df = pd.DataFrame()
                df[title + ' :trial_num'] = reward_group['trial_num']
                df[title + ' :trial_length'] = reward_group['trial_length']
                df.reset_index(drop=True, inplace=True)
                stats_df = pd.concat([stats_df, df], axis=1)

            # Set title and labels for the plot
            ax.set_title(f'trials length over trials: {condition}')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Trial Length')
            ax.legend()
            print("\n\n")

            plt.figure()
            hist_plot = TrialTimeline_df['trial_length'].plot(kind='hist',
                                                              bins=100,
                                                              title=f"Trial {condition} Length Distribution",
                                                              xlabel='Trial Length(sec)',
                                                              ylabel='Frequency')
            frequencies = get_frequencies(hist_plot)
            histogram_df = pd.DataFrame({f"trial {condition} all rewards length Distribution": frequencies})
            histogram_df.reset_index(drop=True, inplace=True)
            stats_df = pd.concat([stats_df, histogram_df], axis=1)

    return results, stats_df


def create_gui():
    def run_post_processing():
        path = entry_path.get()
        start = int(entry_start.get())
        end = int(entry_end.get())
        remove_outliers = var_outliers.get()
        run_on_all_sessions = var_all_sessions.get()
        all_session_post_proc(path, start, end, remove_outliers, run_on_all_sessions)
        root.destroy()  # Close the window after clicking the button

    root = tk.Tk()
    root.title("Post Processing Parameters")
    root.geometry("500x250")  # Set initial window size

    # Path
    label_path = tk.Label(root, text="Enter Path:")
    label_path.pack()
    entry_path = tk.Entry(root, width=80)  # Set width of the entry field
    entry_path.pack()
    entry_path.insert(0, "C:\\Users\\itama\\Desktop\\Virmen_Blue\\05-Feb-2024 141858 Blue_42_DavidParadigm")

    # Start
    label_start = tk.Label(root, text="percentage from the start:")
    label_start.pack()
    entry_start = tk.Entry(root)
    entry_start.pack()
    entry_start.insert(0, "0")

    # End
    label_end = tk.Label(root, text="percentage from the start:")
    label_end.pack()
    entry_end = tk.Entry(root)
    entry_end.pack()
    entry_end.insert(0, "100")

    # Remove Outliers
    var_outliers = tk.BooleanVar()
    checkbox_outliers = tk.Checkbutton(root, text="Remove Outliers", variable=var_outliers)
    checkbox_outliers.pack()

    # one session or several
    var_all_sessions = tk.BooleanVar()
    checkbox_all_sessions = tk.Checkbutton(root, text="run on all sessions?", variable=var_all_sessions)
    checkbox_all_sessions.pack()
    # Button to run post_processing function
    button_run = tk.Button(root, text="Run Post Processing", command=run_post_processing)
    button_run.pack()

    root.mainloop()


def write_end_results(folder_path, file_dir, dict):
    # folder_path = os.path.dirname(file_path)
    # Open the CSV file in write mode
    with open(folder_path + "\\end_results.csv", 'a', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)

        # Write header (optional, depending on your needs)
        csv_writer.writerow([file_dir, file_dir + ' Value'])

        # Write each key-value pair as a row
        for key, value in dict.items():
            csv_writer.writerow([key, value])


def all_session_post_proc(path, start, end, remove_outliers, run_on_all_sessions):
    results_file = open(path + "\\end_results.csv", 'w', newline='')
    results_file.close()
    if run_on_all_sessions:
        all_results = []
        for entry in os.scandir(path):
            if entry.is_dir():
                session_folder = entry.path
                session_results = post_processing(session_folder, start, end, remove_outliers)
                all_results.append(session_results)
                write_end_results(path, session_folder, session_results)
                print(all_results)
    else:
        try:
            session_results = post_processing(path, start, end, remove_outliers)
            write_end_results(path, path, session_results)
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
    print(f'Data has been written to {path}')


if __name__ == '__main__':
    # Start the GUI
    create_gui()
