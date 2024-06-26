import os

from functools import partial
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter import messagebox
import csv
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from tabulate import tabulate


def post_processing(path_of_directory, percentage_from_start, percentage_from_end, remove_outliers):
    global config_json
    AB_lickport_record_df, Reward_df, TrialTimeline_df, config_json, sound_df, velocity_df, stats_df = create_df(
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

    # create the vel and pos df from an existing file or create from the A_B data
    TrialTimeline_df, vel_from_AB_df, vel_from_AB_df_with_ITI = create_vel_pos_df(AB_lickport_record_df, Reward_df,
                                                                                  TrialTimeline_df, path_of_directory)
    # remove outliars from session
    if remove_outliers:  # todo:switch back vel_from_AB_df
        AB_lickport_record_df, Reward_df, TrialTimeline_df, sound_df, velocity_df, vel_from_AB_df, vel_from_AB_df_with_ITI = outliers_removal(
            AB_lickport_record_df,
            Reward_df, TrialTimeline_df,
            sound_df, velocity_df, vel_from_AB_df, vel_from_AB_df_with_ITI)

    trial_length_results, stats_df = trial_length_processing(stats_df, TrialTimeline_df, bins, group_labels)

    global trials_time_range
    trials_time_range = TrialTimeline_df['timestamp'].values.tolist()
    global reward_time_range
    reward_time_range = Reward_df['timestamp_reward_start'].values.tolist()

    # calculate_position(lickport_trial_merged_df_with_zeros, reward_time_range, trials_time_range, vel_from_AB_df)

    velocity_trial_merged_df = pd.merge(vel_from_AB_df, TrialTimeline_df,
                                        on='trial_num')  # todo:switch back for old sessions
    velocity_trial_merged_with_ITI_df = pd.merge(vel_from_AB_df_with_ITI, TrialTimeline_df,
                                                 on='trial_num')  # todo:switch back for old sessions

    lickport_results, stats_df = lickport_processing(stats_df, bins, group_labels, lickport_trial_merged_df_with_zeros,
                                                     velocity_trial_merged_df, TrialTimeline_df, reward_time_range,
                                                     Reward_df)

    # velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')

    velocity_results, stats_df = velocity_processing(stats_df, bins, group_labels, velocity_trial_merged_df,
                                                     velocity_trial_merged_with_ITI_df, config_json)

    # create_formatted_file(Reward_df, TrialTimeline_df, lickport_trial_merged_df_with_zeros,
    #                       config_json, lickport_end_df, lickport_start_df,
    #                       path_of_directory, sound_df)

    stats_df.to_csv(path_of_directory + "\\data_for_stats.csv", float_format='%.4f',
                    index=False)  # write the dataframe into a csv

    plt.show()
    # all the results from the processing and the number of trials in the session
    final_amount_of_trials = TrialTimeline_df.shape[0]  # without the outliers
    result_dict = {**trial_length_results, **lickport_results, **velocity_results,
                   **{"number of trials": final_amount_of_trials}}

    return result_dict


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


def trial_duration_respectively_to_previous(TrialTimeline_df, stats_df):
    '''Previous trial effect'''
    Previous_large = []
    Previous_small = []

    Previous_large_Current_Small = []  # Befor the trail (Previous) Now the trial is small (Current)
    Previous_large_Current_Large = []
    Previous_small_Current_Small = []
    Previous_small_Current_Large = []
    index_without_oitliers = []
    for i in TrialTimeline_df['trial_num']:
        if i not in TrialTimeline_df.index:
            continue  # Skip missing indexes - Outliers
        if TrialTimeline_df.loc[i - 1, 'reward_size'] < 10:
            Previous_small.append(float(TrialTimeline_df.loc[i, 'trial_length']))  # Calculation of the duration
            if TrialTimeline_df.loc[i, 'reward_size'] < 10:
                Previous_small_Current_Small.append(float(TrialTimeline_df.loc[i, 'trial_length']))
            elif TrialTimeline_df.loc[i, 'reward_size'] > 15:
                Previous_small_Current_Large.append(float(TrialTimeline_df.loc[i, 'trial_length']))

        elif TrialTimeline_df.loc[i - 1, 'reward_size'] > 15:
            Previous_large.append((float(TrialTimeline_df.loc[i, 'trial_length'])))
            if TrialTimeline_df.loc[i, 'reward_size'] < 10:
                Previous_large_Current_Small.append(float(TrialTimeline_df.loc[i, 'trial_length']))
            elif TrialTimeline_df.loc[i, 'reward_size'] > 15:
                Previous_large_Current_Large.append(float(TrialTimeline_df.loc[i, 'trial_length']))

    if Previous_large[0] == 0:
        Previous_large.remove(0)
    if Previous_small[0] == 0:
        Previous_small.remove(0)

    # Calculate means
    mean_previous_small = np.mean(Previous_small)
    mean_previous_large = np.mean(Previous_large)
    mean_current_small = np.mean(Previous_large_Current_Small + Previous_small_Current_Small)
    mean_current_large = np.mean(Previous_large_Current_Large + Previous_small_Current_Large)
    # Calculate medians
    median_previous_small = np.median(Previous_small)
    median_previous_large = np.median(Previous_large)
    median_current_small = np.median(Previous_large_Current_Small + Previous_small_Current_Small)
    median_current_large = np.median(Previous_large_Current_Large + Previous_small_Current_Large)

    table = [
        ["Previous Trial Size", "Current Trial Size (Small)", "Current Trial Size (Large)", ""],
        ["Small", mean(Previous_small_Current_Small), mean(Previous_small_Current_Large),
         f"Mean: {mean_previous_small:.2f}"],
        ["Large", mean(Previous_large_Current_Small), mean(Previous_large_Current_Large),
         f"Mean: {mean_previous_large:.2f}"],
        ["", f"Mean: {mean_current_small:.2f}", f"Mean: {mean_current_large:.2f}"]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    print()
    table_medians = [
        ["Previous Trial Size", "Current Trial Size (Small)", "Current Trial Size (Large)", ""],
        ["Small", f"Median: {np.median(Previous_small_Current_Small):.2f}",
         f"Median: {np.median(Previous_small_Current_Large):.2f}", f"Median: {median_previous_small:.2f}"],
        ["Large", f"Median: {np.median(Previous_large_Current_Small):.2f}",
         f"Median: {np.median(Previous_large_Current_Large):.2f}", f"Median: {median_previous_large:.2f}"],
        ["", f"Median: {median_current_small:.2f}", f"Median: {median_current_large:.2f}", ""]]

    print(tabulate(table_medians, headers="firstrow", tablefmt="grid"))
    print()

    # Find the maximum length among the arrays
    max_length = max(len(Previous_large_Current_Small), len(Previous_large_Current_Large),
                     len(Previous_small_Current_Small), len(Previous_small_Current_Large))

    # Pad the arrays with NaN values to make them equal in length
    Previous_large_Current_Small += [np.nan] * (max_length - len(Previous_large_Current_Small))
    Previous_large_Current_Large += [np.nan] * (max_length - len(Previous_large_Current_Large))
    Previous_small_Current_Small += [np.nan] * (max_length - len(Previous_small_Current_Small))
    Previous_small_Current_Large += [np.nan] * (max_length - len(Previous_small_Current_Large))

    # Create a DataFrame
    df = pd.DataFrame()
    df["Previous_large_Current_Small"] = Previous_large_Current_Small
    df["Previous_large_Current_Large"] = Previous_large_Current_Large
    df["Previous_small_Current_Small"] = Previous_small_Current_Small
    df["Previous_small_Current_Large"] = Previous_small_Current_Large

    # Reset the index of df
    df.reset_index(drop=True, inplace=True)

    # Concatenate stats_df and df along the columns (axis=1)
    stats_df = pd.concat([stats_df, df], axis=1)

    return stats_df


# todo: delete?
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
    print(f"aaaa std = {np.std(all_positions)}")
    # position_df = pd.DataFrame({'position': all_positions})
    vel_from_AB_df['position'] = all_positions


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
    sound_df = pd.read_csv(path_of_directory + "\\SoundGiven.csv")
    config_file = open(path_of_directory + "\\config.json")
    config_json = json.load(config_file)
    config_file.close()
    return AB_lickport_record_df, Reward_df, TrialTimeline_df, config_json, sound_df, velocity_df, stats_df


def outliers_removal(AB_lickport_record_df, Reward_df, TrialTimeline_df, sound_df, velocity_df, vel_from_AB_df,
                     vel_from_AB_df_with_ITI):
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
    vel_from_AB_df = vel_from_AB_df[~vel_from_AB_df['trial_num'].isin(abnormal_trial_nums)]
    vel_from_AB_df_with_ITI = vel_from_AB_df_with_ITI[~vel_from_AB_df_with_ITI['trial_num'].isin(abnormal_trial_nums)]
    sound_df = sound_df[~sound_df['trial_num'].isin(abnormal_trial_nums)]
    return AB_lickport_record_df, Reward_df, TrialTimeline_df, sound_df, velocity_df, vel_from_AB_df, vel_from_AB_df_with_ITI


# def plot_velocity_over_position_all_trials(stats_df, config_json, velocity_trial_merged_df, title, label, graph_color, ax=None):
#     for trial in

def plot_raster_vel_over_pos(stats_df, vel_by_pos_all_trials, position_bins, title):
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

    # Define a color palette using a colormap, e.g., 'viridis'
    color_palette = plt.cm.viridis(np.linspace(0, 1, len(vel_by_pos_all_trials)))

    # Plot each set of velocities in a loop, using position_bins as the x-values
    for i, velocities in enumerate(vel_by_pos_all_trials):
        # Remove the pos bins which we didn't arrive in the trial
        position_bins_for_curr_trial = position_bins[:len(velocities)]
        # Use color from the color palette
        color = color_palette[i]
        plt.scatter(position_bins_for_curr_trial, velocities, s=5, color=color)  # s is the marker size
        plt.plot(position_bins_for_curr_trial, velocities, alpha=0.5, color=color)
    legend_handles = []
    num_segments = 5  # Adjust based on how many segments you want to show in the legend
    for i in range(num_segments):
        color = plt.cm.viridis(i / num_segments)  # Getting the color corresponding to the segment
        label = f"{i * (100/num_segments)}% - {(i + 1) * (100/num_segments)}%"  # Adjust label as needed for your segments
        patch = mpatches.Patch(color=color, label=label)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, title="trial percentage", bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Velocity')



def plot_velocity_over_position(stats_df, config_json, velocity_trial_merged_df, title, label, graph_color, ax=None):
    # Create figure with 2 subplots (2 rows, 1 column)
    # combined_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    mean_vel_by_pos_all_trials = []
    std_vel_by_pos_all_trials = []
    vel_by_pos_all_trials = [[] for _ in range(int(config_json['db_amount_trials']))]
    position_segments = np.linspace(0, int(config_json['db_distance_to_run']), 60,
                                    endpoint=True)  # todo *2 instaid of 60

    # get the mean and std of every trial
    for i in range(len(position_segments) - 1):
        # divide the data by the position of the mouse
        data_by_position = velocity_trial_merged_df.loc[
            (velocity_trial_merged_df['position'] >= position_segments[i])
            & (velocity_trial_merged_df['position'] <= position_segments[i + 1])]
        # Group by 'trial_num' and calculate the mean 'Avg_velocity' for each group
        grouped_by_trial = data_by_position.groupby('trial_num').mean()

        # # Append the mean velocities for each trial to the corresponding inner list in vel_by_pos_all_trials
        # vel_by_pos_all_trials[i].extend(grouped_by_trial)
        for trial_num, row in grouped_by_trial.iterrows():
            # Assuming trial_num is 0-indexed; if it's 1-indexed, you might not need to adjust it
            # If trial_num starts from 1 or another number, adjust accordingly
            index = trial_num - 1  # Adjust based on your trial_num starting value
            avg_velocity = row['Avg_velocity']  # Get the mean Avg_velocity for this trial

            # Add the avg_velocity to the corresponding inner list based on trial_num
            # Check if the index is within bounds of your list
            if 0 <= index < len(vel_by_pos_all_trials):
                vel_by_pos_all_trials[index].append(avg_velocity)
        # vel_by_pos_all_trials.append(data_by_position['Avg_velocity'])
        mean_velocity_by_position = data_by_position['Avg_velocity'].mean()
        std_velocity_by_position = data_by_position['Avg_velocity'].std()
        mean_vel_by_pos_all_trials.append(mean_velocity_by_position)
        std_vel_by_pos_all_trials.append(std_velocity_by_position)
    # remove empty lists
    vel_by_pos_all_trials = [inner_list for inner_list in vel_by_pos_all_trials if inner_list]
    # set the points in the middle of the section of the speed range
    position_bins = [(position_segments[i] + position_segments[i + 1]) / 2 for i in range(len(position_segments) - 1)]

    plot_raster_vel_over_pos(stats_df, vel_by_pos_all_trials, position_bins,
                             f"raster of velocity over position -- reward size: {label}")
    # scatter plot with its std
    ax.errorbar(position_bins, mean_vel_by_pos_all_trials, yerr=std_vel_by_pos_all_trials,
                linestyle='--', marker='o', color=graph_color,
                label=f'mean velocity {label} ± std')
    ax.fill_between(position_bins,
                    np.array(mean_vel_by_pos_all_trials) - np.array(std_vel_by_pos_all_trials),
                    np.array(mean_vel_by_pos_all_trials) + np.array(std_vel_by_pos_all_trials),
                    color=graph_color, alpha=0.4, label=f'Error Range {label}')
    # Add text annotations for each bar at the top
    # for x, y, value in zip(position_bins, velocity_by_position, velocity_by_position):
    #     ax.text(x, y, f'{value:.2f}', ha='center', va='bottom', color='black', fontsize=8)

    # for i, velocities in enumerate(vel_by_pos_all_trials):
    #     # Plot each velocity data point for the segment
    #     x_values = [position_bins[i]] * len(velocities)
    #     ax2.scatter(x_values, velocities, color=graph_color, alpha=0.6, s=10)  # Adjust size as needed
    #     ax2.plot(x_values, y_values, marker='', linestyle='-', color='gray', alpha=0.5)  # Adjust color/alpha as needed

    df = pd.DataFrame()
    df[title + f" reward size {label} :position"] = position_bins
    df[title + f" reward size {label} :velocity"] = mean_vel_by_pos_all_trials
    df.reset_index(drop=True, inplace=True)

    stats_df = pd.concat([stats_df, df], axis=1)

    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Mean Velocity')
    ax.legend()

    return stats_df


def calc_licks_around_time_event(stats_df, lickport_trial_merged_df_with_zeros, reward_times, title, all_buffers):
    length_of_buff = 4  # time buffer around the start of the trial/reward
    if all_buffers is None:
        all_buffers = []
        # separate the data around each reward of a trial
        for i in range(len(reward_times)):
            buffer_around_trial = lickport_trial_merged_df_with_zeros.loc[
                (lickport_trial_merged_df_with_zeros['timestamp_x'] >= reward_times[i] - length_of_buff)
                & (lickport_trial_merged_df_with_zeros['timestamp_x'] <= reward_times[i] + length_of_buff)]
            if not buffer_around_trial.empty:
                # normalize the timestamp to start from 0
                buffer_around_trial.loc[:, 'timestamp_x'] = buffer_around_trial['timestamp_x'] - reward_times[i]
                # take only the activation of the lickport
                buffer_around_trial = buffer_around_trial[(buffer_around_trial['lickport_signal'] == 1) &
                                                          (buffer_around_trial['lickport_signal'].shift(1) == 0)]

                all_buffers.append(buffer_around_trial)

    stats_df = plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title, 'timestamp_x')
    return stats_df, all_buffers


# TODO: finish

def calc_licks_around_position_event(stats_df, lickport_trial_merged_df_with_zeros, reward_times,
                                     vel_pos_df, title, all_buffers):
    track_length = int(config_json['db_distance_to_run'])
    length_of_buff = 20  # position buffer around the start of the reward
    start_location = track_length - length_of_buff
    if all_buffers is None:
        all_buffers = []
        # Perform an asof merge with forward fill to handle NaN values for the last positions
        lickport_with_pos = pd.merge_asof(lickport_trial_merged_df_with_zeros, vel_pos_df[['timestamp_x', 'position']],
                                          on='timestamp_x', direction='forward').fillna(method='ffill')

        # take only the activation of the lickport
        buffer_around_trial = lickport_with_pos[(lickport_with_pos['lickport_signal'] == 1) &
                                                (lickport_with_pos['lickport_signal'].shift(1) == 0)]
        # separate the data around each reward of a trial
        buffer_around_trial = buffer_around_trial.loc[
            (buffer_around_trial['position'] >= start_location)]
        if not buffer_around_trial.empty:
            # decrease the first timestamp so all will start from 0
            buffer_around_trial.loc[:, 'timestamp_x'] = buffer_around_trial['timestamp_x'] - \
                                                        buffer_around_trial['timestamp_x'].iloc[0]

            all_buffers.append(buffer_around_trial)
    fig, ax = plt.subplots()
    title = f"{title} -- licks over position, {length_of_buff} cm before reward"
    # Now use the ax parameter to specify where to plot
    histogram_plot = buffer_around_trial['position'].plot(kind='hist',
                                                          bins=40,
                                                          ax=ax,  # This ensures the plot goes to the new figure
                                                          label='licks',
                                                          title=title,
                                                          color='pink')
    ax.legend()
    plt.tight_layout()
    frequencies = get_frequencies(histogram_plot)  # todo: change to use the plot_lick_around_time_event func
    df = pd.DataFrame({title + " frequencies": frequencies})
    stats_df = pd.concat([stats_df, df], axis=1)
    # stats_df = plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title, 'position')
    return stats_df


def plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title, x_axis):
    # Function to get the first value of the "trial num" column
    def get_first_trial_num(df):
        return df['trial_num'].iloc[0]

    # Sort the array of dataframes by the first value of the "trial num" column
    all_buffers = sorted(all_buffers, key=get_first_trial_num)
    # Update to 3 rows, 1 column for subplots
    lick_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))  # 3 rows, 1 column
    # Plot each DataFrame in a loop, vertically spaced, on the first subplot
    for i, s in enumerate(all_buffers):
        s['location_in_scatter'] = i
        s.plot(kind='scatter', x=x_axis, y='location_in_scatter', ax=ax1, s=5)
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_title(title + ' -- Licks over time')
    ax1.set_xlabel('time')
    ax1.set_ylabel('start licking')

    # Second subplot for the calculated PDF histogram
    all_licks = pd.concat(all_buffers)
    hist_title = f"lickport {length_of_buff} sec around the start of the reward"
    bin_width = 0.1  # Fine granularity for more bars
    min_time, max_time = -length_of_buff, length_of_buff
    bins = np.arange(min_time, max_time + bin_width, bin_width)
    counts, bin_edges = np.histogram(all_licks[x_axis], bins=bins)
    probabilities = counts / counts.sum()  # Convert counts to probabilities
    ax2.bar(bin_edges[:-1], probabilities, width=bin_width, align='edge')
    ax2.axvline(x=0, color='red', linestyle='--', label='reward start')
    ax2.set_ylabel('Probability')
    ax2.legend()

    # histogram_plot = all_licks['timestamp_x'].plot(kind='hist',
    #                                                bins=100,
    #                                                label='licks',
    #                                                title=hist_title,
    #                                                color='green')

    # frequencies = get_frequencies(histogram_plot) # todo: return the lines
    df = pd.DataFrame({title + " probabilities": probabilities})
    stats_df = pd.concat([stats_df, df], axis=1)

    # Third subplot for the additional histogram with 100 bins
    ax3.set_ylim([0, 120])
    ax3.hist(all_licks[x_axis], bins=100, label='licks', color='green', alpha=0.6)
    ax3.set_title(hist_title)
    ax3.axvline(x=0, color='red', linestyle='--', label='reward start')
    ax3.set_ylabel('Amount')
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


# Function to process lickport data and generate statistics and plots.
def lickport_processing(stats_df, bins, group_labels, lickport_trial_merged_df_with_zeros, velocity_trial_merged_df,
                        TrialTimeline_df,
                        reward_time_range, Reward_df):
    # Group the lickport data by trial percentage using the specified bins and labels.
    grouped_lickport_by_trial_precentage = lickport_trial_merged_df_with_zeros.groupby(
        pd.cut(lickport_trial_merged_df_with_zeros['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    results = {}
    # all the trials' buffer
    all_buffers_time = None
    all_buffers_position = None
    overall_buff = []

    print("sum of lickport activations by reward:")
    # Iterate over each group (condition) in the grouped data.
    for condition, group in grouped_lickport_by_trial_precentage:
        if not group.empty:  # Check to avoid processing empty groups.
            print(f"\t{condition}:")
            # Further group the data by reward size within each trial percentage group.
            grouped_by_reward_type = group.groupby('reward_size')
            colors = ['blue', 'yellow']
            fig, ax = plt.subplots()
            # Iterate over each reward size group within the current condition.
            for i, (condition2, reward_group) in enumerate(grouped_by_reward_type):
                sum_of_licks = reward_group['lickport_signal'].sum()
                results["lickport activation" + str(condition) + str(condition2)] = sum_of_licks
                print(f"\t\tCondition- reward size {condition2}: {sum_of_licks}")
                title = f'lickport of trials {condition} Reward Size {condition2}'

                # Get reward times for the current reward size group.
                reward_times_by_group = Reward_df.loc[(Reward_df['reward_size'] == condition2)]
                reward_times_by_group = reward_times_by_group['timestamp_reward_start'].values.tolist()
                # Calculate licks around reward time / position for the current reward group.
                stats_df, buff = calc_licks_around_time_event(stats_df, reward_group, reward_times_by_group, title,
                                                              all_buffers_time)
                # run position base analysis only if the df exists
                if not velocity_trial_merged_df.empty:
                    stats_df = calc_licks_around_position_event(stats_df, reward_group, reward_times_by_group,
                                                                velocity_trial_merged_df, title,
                                                                all_buffers_position)
                overall_buff.extend(buff)  # Add the current buffer to the overall buffer list.
                # Get only the activation of the lickport.
                reward_group = reward_group[(reward_group['lickport_signal'] == 1) &
                                            (reward_group['lickport_signal'].shift(1) == 0)]
                # Plot lickport activations for the current reward size group.
                reward_group.plot(kind='scatter', x='timestamp_x', y='lickport_signal',
                                  title=f'lickport of trials {condition}%', label=f'Reward Size {condition2}', ax=ax,
                                  color=colors[i])
                df = pd.DataFrame()
                df[title + ' :timestamp'] = reward_group['timestamp_x']
                df.reset_index(drop=True, inplace=True)
                stats_df = pd.concat([stats_df, df], axis=1)
            # Plot vertical lines for each timestamp in the TrialTimeline DataFrame.
            for timestamp in TrialTimeline_df['timestamp']:
                ax.axvline(x=timestamp, color='red', linestyle='--')
            # Calculate licks around time events for the entire group (all reward sizes combined).
            stats_df, buff = calc_licks_around_time_event(stats_df, group, reward_time_range,
                                                          f"{str(condition)} all reward sizes", overall_buff)

            print()
    print(f"all reward sizes:\n{grouped_lickport_by_trial_precentage['lickport_signal'].sum()}")
    print("\n\n")

    return results, stats_df  # Return the results dictionary and the updated stats DataFrame.


def velocity_processing(stats_df, bins, group_labels, velocity_trial_merged_df, velocity_trial_merged_with_ITI_df,
                        config_json):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    # Filter 'Avg_velocity' without rows equal to 0f
    velocity_without_zeros = velocity_trial_merged_df.loc[
        (0.5 < velocity_trial_merged_df['Avg_velocity']) | (velocity_trial_merged_df['Avg_velocity'] < -0.5)]

    velocity_with_ITI_without_zeros = velocity_trial_merged_with_ITI_df.loc[
        (0.5 < velocity_trial_merged_with_ITI_df['Avg_velocity']) | (
                    velocity_trial_merged_with_ITI_df['Avg_velocity'] < -0.5)]

    plot_velocity_over_time(ax1, velocity_with_ITI_without_zeros, 'Velocity over Time')

    results = {}

    grouped_velocity_by_trial_precentage = velocity_without_zeros.groupby(
        pd.cut(velocity_without_zeros['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("average velocity by reward:")
    for condition, group in grouped_velocity_by_trial_precentage:
        if not group.empty:  # for not creating empty figures
            print(f"\t{condition}:")
            grouped_by_reward_type = group.groupby('reward_size')
            colors = ['blue', 'green']
            fig, ax = plt.subplots()
            for i, (condition2, reward_group) in enumerate(grouped_by_reward_type):
                results, stats_df = calc_movement_during_session(stats_df, results, velocity_trial_merged_df,
                                                                 reward_group, condition, condition2)
                mean_vel = reward_group['Avg_velocity'].mean().round(4)
                median_vel = reward_group['Avg_velocity'].median().round(4)
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
                                                       f'velocity over position {condition} (without 0s)',
                                                       label=condition2,
                                                       graph_color=colors[i], ax=ax)
                # histogram of speed PDF
                vel_hist_title = f"Velocity PDF: Trials {condition}, Reward Size {condition2}"
                hist_values = plot_vel_PDF_hist(vel_hist_title, reward_group, mean_vel, median_vel)

                histogram_df = pd.DataFrame({title: hist_values})
                stats_df = pd.concat([stats_df, histogram_df], axis=1)

            print()
            # plot for all the reward types and all trials
            vel_hist_title = f"Velocity PDF: trials {condition}, all reward"
            mean_vel_all_reward = group['Avg_velocity'].mean().round(4)
            median_vel_all_reward = group['Avg_velocity'].median().round(4)
            hist_values_all_reward = plot_vel_PDF_hist(vel_hist_title, group, mean_vel_all_reward,
                                                       median_vel_all_reward)  # todo: add ral median and mean

            histogram_df = pd.DataFrame({vel_hist_title : hist_values_all_reward})
            histogram_df.reset_index(drop=True, inplace=True)
            stats_df = pd.concat([stats_df, histogram_df], axis=1)
            print(f"all reward sizes:\n{mean_vel_all_reward}")
            print("\n\n")
    return results, stats_df


def plot_vel_PDF_hist(title, reward_group, mean_vel, median_vel):
    plt.figure()
    # Get the histogram data
    hist_values, bin_edges, _ = plt.hist(reward_group['Avg_velocity'], bins=50, density=True)
    # Calculate the width of each bin
    bin_width = bin_edges[1] - bin_edges[0]
    # Scale the histogram values by the bin width
    hist_values_scaled = hist_values * bin_width
    # Plot the scaled histogram
    plt.bar(bin_edges[:-1], hist_values_scaled, width=bin_width, align='edge')

    plt.axvline(x=mean_vel, color='black', linestyle='--', label=f'mean = {mean_vel}')
    plt.axvline(x=median_vel, color='red', linestyle='--', label=f'median = {median_vel}')
    # Set title and labels
    # extra_var_patch = mpatches.Patch(colorlabel=f'Median = {median_vel}')

    plt.title(title)
    plt.xlabel('Speed (cm/sec)')
    plt.ylabel('Probability')
    plt.legend()
    # plt.legend(handles=[mpatches.Patch(color='red', linestyle='--', label='Threshold'), extra_var_patch])
    return hist_values


def plot_velocity_over_time(ax1, velocity_df, title):
    # Find unique reward sizes and their min/max
    unique_rewards = velocity_df['reward_size'].unique()
    big_reward_val = unique_rewards.max()
    small_reward_val = unique_rewards.min()

    # Define color map
    color_map = {small_reward_val: 'green', big_reward_val: 'blue'}

    # Plot each reward size category separately
    for reward_size, color in color_map.items():
        # Filter dataframe by reward_size
        df_filtered = velocity_df[velocity_df['reward_size'] == reward_size]
        # Plot scatter for the filtered dataframe
        ax1.scatter(df_filtered['timestamp_x'], df_filtered['Avg_velocity'], c=color, s=5,
                    label=f'Reward Size: {reward_size}')

    # Add velocity trend line
    ax1.plot(velocity_df['timestamp_x'], velocity_df['Avg_velocity'], color='gray', alpha=0.5, label='Velocity Trend')

    # Adding vertical lines for trials and rewards
    # trials_time_range and reward_time_range need to be defined or passed to the function
    ax1.axvline(x=trials_time_range[0], color='red', linestyle='--', label='Trial Start')
    ax1.axvline(x=reward_time_range[0], color='black', linestyle='--', label='Reward')
    for timestamp in trials_time_range[1:]:
        ax1.axvline(x=timestamp, color='red', linestyle='--')
    for timestamp in reward_time_range[1:]:
        ax1.axvline(x=timestamp, color='black', linestyle='--')

    # Setting title and labels
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Avg Velocity')

    # Create legend
    ax1.legend()


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


# calculate the % of movement in the trials and the stops durations
def calc_movement_during_session(stats_df, results, velocity_trial_merged_df, velocity_without_zeros, trial_prec,
                                 reward_size):
    movement_during_trial = []
    trials = []
    num_of_samples = 0
    non_zero_samples = 0
    all_trial_stops = []
    all_trials_stop_positions = []
    mean_vel_between_stops = []
    max_vel_between_stops = []
    group_by_trial_with_zero = velocity_trial_merged_df.groupby('trial_num')
    group_by_trial_without_zero = velocity_without_zeros.groupby('trial_num')
    # for each trial get the samples including 0's and excluding 0's
    for trial_num, group_without_zero in group_by_trial_without_zero:
        group_with_zero = group_by_trial_with_zero.get_group(trial_num)
        # fill all_trial_stops and all_trials_stop_positions with the stops of each trial
        get_trial_stops(all_trial_stops, all_trials_stop_positions, mean_vel_between_stops, max_vel_between_stops,
                        group_with_zero)
        # percentage of movement during the trial
        movement_during_trial.append((group_without_zero.shape[0] / group_with_zero.shape[0]) * 100)
        trials.append(trial_num)
        num_of_samples += group_with_zero.shape[0]
        non_zero_samples += group_without_zero.shape[0]

    # Flatten the nested list all_trials_stop_positions into a single list
    flattened_positions = [position for sublist in all_trials_stop_positions for position in sublist]
    # Create the DataFrame with the flattened list
    title = f"trial stops over position {trial_prec} reward size {reward_size}"
    df = pd.DataFrame()
    df[title + ' :position'] = flattened_positions  # Use the flattened list here
    df.reset_index(drop=True, inplace=True)
    # Concatenate the new DataFrame to stats_df along columns
    stats_df = pd.concat([stats_df, df], axis=1)
    # todo add to stats
    # plot the mean/max velocity between stops
    plot_vel_between_stops_heatmap(all_trials_stop_positions, mean_vel_between_stops,
                                   f"trial mean vel between stops over position {trial_prec} reward size {reward_size}")
    plot_vel_between_stops_heatmap(all_trials_stop_positions, max_vel_between_stops,
                                   f"trial max vel between stops over position {trial_prec} reward size {reward_size}")
    # plot stop duration
    plot_stops_heatmap(all_trial_stops, all_trials_stop_positions, title)
    print(
        f"percentage of movement for trials {trial_prec} reward size {reward_size}: {(non_zero_samples / num_of_samples) * 100} %")
    results["percentage of movement for small reward"] = (non_zero_samples / num_of_samples) * 100
    plt.figure()
    plt.scatter(x=trials, y=movement_during_trial, color='orange')
    plt.plot(trials, movement_during_trial, linestyle='-', color='gray')
    plt.title(f'% of movement per trial, trials {trial_prec} reward size: {reward_size}')
    plt.xlabel('Trials')
    plt.ylabel('% of Movement')
    plt.legend(['% of Movement'])
    return results, stats_df


def plot_vel_between_stops_heatmap(all_trials_stop_positions, mean_vel_between_stops, title):
    num_of_trials = len(all_trials_stop_positions)
    fig, ax = plt.subplots()

    # Assuming mean_vel_between_stops is structured similarly to all_trials_stop_positions
    # Normalize the velocity values for color mapping
    flattened_velocities = [vel for sublist in mean_vel_between_stops for vel in sublist]
    if flattened_velocities:
        min_vel_value = np.min(flattened_velocities)
        max_vel_value = np.max(flattened_velocities)
    else:
        min_vel_value = 0
        max_vel_value = 0

    norm = plt.Normalize(min_vel_value, max_vel_value)
    cmap = plt.get_cmap('coolwarm')  # Using 'coolwarm' colormap

    for i, (trial_positions, trial_velocities) in enumerate(zip(all_trials_stop_positions, mean_vel_between_stops)):
        points = np.array([trial_positions, [i] * len(trial_positions)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # plot the stop position points
        if len(trial_positions) > 1:
            ax.scatter(trial_positions, [i] * len(trial_positions), color='black', edgecolor='none', s=20)
        # Create a LineCollection from the segments
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
        # Set the LineCollection color using the trial_velocities
        lc.set_array(np.array(trial_velocities))
        # Add the LineCollection to the plot
        ax.add_collection(lc)

    # Add a colorbar to the plot to show the mapping from velocity values to colors
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Velocity (cm/s)')

    # Optional: Customize the plot with titles and labels
    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Trial')
    ax.autoscale()
    ax.set_ylim(-0.5, num_of_trials)
    ax.set_xlim([-0.5, int(config_json['db_distance_to_run']) * 1.2])


# create a heatmap for the raster plot of the stops in each trial in the session (big/small) over their position
def plot_stops_heatmap(all_trial_stops, all_trials_stop_positions, title):
    # Assuming all_trials_stop_positions is a nested list where each inner list contains position values
    num_of_trials = len(all_trials_stop_positions)
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()
    scatter = None
    for i, (trial_positions, trial_stops) in enumerate(zip(all_trials_stop_positions, all_trial_stops)):
        y_values = [i] * len(trial_positions)  # Adjust y-values so each trial is on a different row

        # Normalize the stop values for color mapping
        flattened_stops = [stop for sublist in all_trial_stops for stop in sublist]

        # if we have stops in trial
        if flattened_stops:
            min_stop_value = np.min(flattened_stops)
            max_stop_value = np.max(flattened_stops)
        else:
            min_stop_value = 0
            max_stop_value = 0

        # Use these min and max values for normalization
        norm = plt.Normalize(min_stop_value, max_stop_value)
        colors = [(0, 'lightblue'), (1, 'red')]  # (position in the colormap, color)
        # Create the colormap
        cmap_name = 'custom_lightblue_red'
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

        # Plot the positions with colors based on corresponding stop values
        scatter = ax.scatter(trial_positions, y_values, c=trial_stops, cmap=custom_cmap, norm=norm,
                             s=20)  # 's' controls the size of the points
    # Add a colorbar to the plot to show the mapping from stop values to colors
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Stop duration')
    # Optional: Customize the plot with titles and labels
    ax.set_title(title)
    ax.set_xlabel('Position')
    ax.set_ylabel('Trial')
    # cosmetic
    ax.set_ylim([0, num_of_trials])
    ax.set_xlim([-0.5, int(config_json['db_distance_to_run']) * 1.2])


def get_trial_stops(all_trial_stops, all_trials_stop_positions, mean_vel_between_stops, max_vel_between_stops,
                    group_with_zero):
    zero_vel_samples = group_with_zero[
        (group_with_zero['Avg_velocity'] > -0.5) & (group_with_zero['Avg_velocity'] < 0.5)].copy()
    # Calculate the time differences between consecutive timestamps
    zero_vel_samples['time_diff'] = zero_vel_samples['timestamp_x'].diff()
    # Identify the start of each stop sequence by marking large time differences
    # Assuming any large gap indicates a new stop period; adjust 'threshold' as needed
    threshold = 0.5  # length threshold for indication of a stop
    zero_vel_samples['new_sequence'] = zero_vel_samples['time_diff'] > threshold
    # Cumulatively sum the 'new_sequence' to create unique group identifiers for each stop sequence
    zero_vel_samples['sequence_id'] = zero_vel_samples['new_sequence'].cumsum()
    # Perform the aggregation to sum 'time_diff' and get the first 'position' for each 'sequence_id' group
    aggregated = zero_vel_samples.groupby('sequence_id').agg({
        'time_diff': 'sum',
        'position': 'first'  # Get the first position value in each group
    })
    stop_durations = aggregated['time_diff']

    # The 'position' of the first row in each group is now also available
    first_positions = aggregated['position']

    all_trial_stops.append(stop_durations.tolist())
    all_trials_stop_positions.append(first_positions.tolist())

    # Reset the index of the DataFrame to ensure it's sequentially indexed
    zero_vel_samples_reset = zero_vel_samples.reset_index()

    # Find indices where new_sequence is True in the reset DataFrame
    new_sequence_indices = zero_vel_samples_reset[zero_vel_samples_reset['new_sequence']].index

    # Initialize lists to store the timestamps and mean velocities
    last_false_timestamps = []
    true_timestamps = []
    mean_velocities = []  # List to store mean velocities for each sequence
    max_velocities = []
    # Iterate through indices to find corresponding timestamp_x values and calculate mean velocities
    for idx in new_sequence_indices:
        # Get the timestamp_x value for the current True index
        true_timestamp = zero_vel_samples_reset.loc[idx, 'timestamp_x']
        true_timestamps.append(true_timestamp)

        # Determine the start index for calculating the mean velocity
        if idx > 0:
            prev_idx = idx - 1
            last_false_timestamp = zero_vel_samples_reset.loc[prev_idx, 'timestamp_x']
            last_false_timestamps.append(last_false_timestamp)
            # Calculate the mean velocity for the current sequence
            # Ensure to filter by the range from the last False to the current True
            mean_velocity = group_with_zero[(group_with_zero['timestamp_x'] > last_false_timestamp) & (
                    group_with_zero['timestamp_x'] < true_timestamp)]['Avg_velocity'].mean()
            max_velocity = group_with_zero[(group_with_zero['timestamp_x'] > last_false_timestamp) & (
                    group_with_zero['timestamp_x'] < true_timestamp)]['Avg_velocity'].max()

            mean_velocities.append(mean_velocity)  # Append the calculated mean velocity to the list
            max_velocities.append(max_velocity)  # Append the calculated mean velocity to the list

    mean_vel_between_stops.append(mean_velocities)
    max_vel_between_stops.append(max_velocities)


def trial_length_processing(stats_df, TrialTimeline_df, bins, group_labels):
    grouped_by_trial_precentage = TrialTimeline_df.groupby(
        pd.cut(TrialTimeline_df['trial_num'], bins=bins, labels=group_labels), observed=False)
    print("average trial length by reward:")
    results = {}
    for condition, group in grouped_by_trial_precentage:
        if not group.empty:  # for not creating empty figures
            print(f"\t{condition}:")
            print('Previous Trial Information: ')
            stats_df = trial_duration_respectively_to_previous(TrialTimeline_df, stats_df)
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
            # hist_plot = TrialTimeline_df['trial_length'].plot(kind='hist',
            #                                                   bins=100,
            #                                                   title=f"Trial {condition} Length Distribution",
            #                                                   xlabel='Trial Length(sec)',
            #                                                   ylabel='Frequency')

            # Identify the unique reward sizes
            unique_rewards = TrialTimeline_df['reward_size'].unique()
            big_reward_val = unique_rewards.max()
            small_reward_val = unique_rewards.min()
            # Prepare the data for each reward size
            big_reward_trials = TrialTimeline_df[TrialTimeline_df['reward_size'] == big_reward_val]['trial_length']
            small_reward_trials = TrialTimeline_df[TrialTimeline_df['reward_size'] == small_reward_val]['trial_length']

            # Determine common bins for both datasets to align them properly
            bins = np.histogram_bin_edges(TrialTimeline_df['trial_length'], bins=50)

            # Plot stacked histogram
            n, bins, patches = plt.hist([big_reward_trials, small_reward_trials], bins=bins, stacked=True,
                                        color=['red', 'green'],
                                        label=[f'Reward Size: {big_reward_val}', f'Reward Size: {small_reward_val}'])

            # Add plot title and labels
            plt.title(f"Trial {condition} Length Distribution")
            plt.xlabel('Trial Length (sec)')
            plt.ylabel('Frequency')

            # Add legend to differentiate between the two reward sizes
            plt.legend()

            frequencies_big = [patch.get_height() for patch in patches[0]]
            frequencies_small = [patch.get_height() for patch in patches[1]]
            histogram_df = pd.DataFrame(
                {f"length Distribution trial {condition} rewards size: {big_reward_val}": frequencies_big,
                 f"length Distribution trial {condition} rewards size: {small_reward_val}": frequencies_small}
            )
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
    entry_path.insert(0, "C:\\Users\\itama\\Desktop\\virmen_purple\\03-Mar-2024 124643 Purple1_24_DavidParadigm")

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
