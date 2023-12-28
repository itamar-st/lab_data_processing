import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def post_processing(path_of_directory, percentage_from_start, percentage_from_end):
    pd.set_option('display.max_columns', None)

    TrialTimeline_df = pd.read_csv(path_of_directory + "\\TrialTimeline.csv")
    Reward_df = pd.read_csv(path_of_directory + "\\Reward.csv")
    AB_lickport_record_df = pd.read_csv(path_of_directory + "\\A-B_leakport_record.csv")
    velocity_df = pd.read_csv(path_of_directory + "\\velocity.csv")
    sound_df = pd.read_csv(path_of_directory + "\\SoundGiven.csv")
    config_file = open(path_of_directory + "\\config.json")
    config_json = json.load(config_file)
    config_file.close()
    # File name for CSV

    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge
    trial_num_bottom_threshold = int(
        TrialTimeline_df.shape[0] * (percentage_from_start / 100))  # precentage to num of trials
    trial_num_top_threshold = int(TrialTimeline_df.shape[0] * (percentage_from_end / 100))
    bins = [float('-inf'), trial_num_bottom_threshold, trial_num_top_threshold, float('inf')]
    group_labels = [f'below {percentage_from_start}%', f'between {percentage_from_start}%-{percentage_from_end}%',
                    f'above {percentage_from_end}%']

    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']

    trial_length_processing(TrialTimeline_df, bins, group_labels)

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

    # create_formatted_file(Reward_df, TrialTimeline_df, lickport_trial_merged_df_with_zeros, config_json, lickport_end_df, lickport_start_df,
    #                       path_of_directory, sound_df)

    # take all rows without 0
    lickport_trial_merged_df = lickport_trial_merged_df_with_zeros[
        lickport_trial_merged_df_with_zeros['lickport_signal'] != 0]
    # lickport_processing(bins, group_labels, lickport_start_df)

    velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')
    velocity_trial_merged_df['Rolling_Avg_Last_2'] = velocity_trial_merged_df['Avg_velocity'].rolling(2).mean().shift(
        -1)
    velocity_processing(bins, group_labels, velocity_trial_merged_df, config_json)

    velocity_trial_merged_df.plot(x='timestamp_x', y=['Avg_velocity', 'Rolling_Avg_Last_2'],
                                  title='velocity over time')
    TrialTimeline_df.plot(x='trial_num', y='trial_length',
                          title='trials length over trials')

    fig, ax = plt.subplots()
    lickport_start_df.plot(kind='scatter',
                           x='timestamp_x',
                           y='lickport_signal',
                           title='lickport with start of trials',
                           c=lickport_start_df['reward_size'],
                           cmap='viridis',
                           ax=ax,
                           label='Lickport Signal')
    for timestamp in TrialTimeline_df['timestamp']:
        ax.axvline(x=timestamp, color='red', linestyle='--')
    # ax.axvline(label='Trial Start', color='r', linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(2, 0.5))  # Position the legend outside the plot

    trials_time_range = TrialTimeline_df['timestamp'].values.tolist()
    all_tracks = []
    reward_time_range = Reward_df['timestamp_reward_start'].values.tolist()

    grouped_by_trial = lickport_trial_merged_df_with_zeros.groupby(
        lickport_trial_merged_df_with_zeros['trial_num'])
    # for i in range(1, len(reward_time_range) - 1):
    #     group = lickport_trial_merged_df_with_zeros.loc[
    #         (lickport_trial_merged_df_with_zeros['timestamp_x'] >= trials_time_range[i])
    #         & (lickport_trial_merged_df_with_zeros['timestamp_x'] <= reward_time_range[i])]
    #     slits_passed = group[
    #         (group['B_signal'] == 0) &
    #         (group['B_signal'].shift(1) == 1)]
    #     slits_returnd = group[
    #         (group['B_signal'] == 0) &
    #         (group['B_signal'].shift(1) == 1)]
    #     differences = group['timestamp_x'] - group['timestamp_x'].shift(1)
    #     filtered_differences = differences[differences > 0.01011]
    #     # View all the values of the differences that satisfy the condition
    #     num_of_slits_passed = len(slits_passed)
    #     print(i, num_of_slits_passed)
    # if trial_num==68:
    #     print(trial_num, num_of_slits_passed)

    all_buffers = []
    length_of_buff = 4  # time buffer around the start of the trial/reward
    # separate the data around each start of a trial
    for i in range(1, len(reward_time_range) - 1):
        buffer_around_trial = lickport_trial_merged_df_with_zeros.loc[
            (lickport_trial_merged_df_with_zeros['timestamp_x'] >= reward_time_range[i] - length_of_buff)
            & (lickport_trial_merged_df_with_zeros['timestamp_x'] <= reward_time_range[i] + length_of_buff)]
        # decrease the first timestamp so all will start from 0
        buffer_around_trial.loc[:, 'timestamp_x'] = buffer_around_trial['timestamp_x'] - \
                                                    buffer_around_trial['timestamp_x'].iloc[0]
        # take only the activation of the lickport
        buffer_around_trial = buffer_around_trial[(buffer_around_trial['lickport_signal'] == 1) &
                                                  (buffer_around_trial['lickport_signal'].shift(1) == 0)]
        if buffer_around_trial.empty:
            print(f"no data for the buffer around trial number {i}")
        else:
            all_buffers.append(buffer_around_trial)

    plot_lick_around_time_event(all_buffers, length_of_buff)
    plot_velocity_over_position(config_json, velocity_trial_merged_df, 'velocity over position')
    plt.show()


def plot_velocity_over_position(config_json, velocity_trial_merged_df, title):
    velocity_by_position = []
    position_segments = np.linspace(0, int(config_json['db_distance_to_run']), 60, endpoint=True)
    for i in range(len(position_segments) - 1):
        data_by_position = velocity_trial_merged_df.loc[
            (velocity_trial_merged_df['position'] >= position_segments[i])
            & (velocity_trial_merged_df['position'] <= position_segments[i + 1])]
        mean_velocity_by_position = data_by_position['Avg_velocity'].mean()
        velocity_by_position.append(mean_velocity_by_position)
    plt.figure()
    position_bins = [(position_segments[i] + position_segments[i + 1]) / 2 for i in range(len(position_segments) - 1)]
    plt.plot(position_bins,
             velocity_by_position,
             linestyle='--',
             marker='o',
             color='red',
             mec='blue',
             label='mean velocity')
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Mean Velocity')
    plt.legend()


def plot_lick_around_distance_event(all_buffers, length_of_buff):
    lick_fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
    # Plot each DataFrame in a loop, vertically spaced
    num_of_buffers = len(all_buffers)

    for i, s in enumerate(all_buffers):
        s['lickport_signal'] = s['lickport_signal'] + num_of_buffers - i
        s.plot(kind='scatter', x='timestamp_x', y='lickport_signal', ax=ax3, s=5)
    ax3.axvline(x=length_of_buff, color='red', linestyle='--')
    ax3.set_title('Licks over time')
    ax3.set_xlabel('time')
    ax3.set_ylabel('start licking')

    all_licks = pd.concat(all_buffers)
    all_licks['timestamp_x'].plot(kind='hist',
                                  bins=100,
                                  ax=ax4,
                                  label='licks',
                                  title=f"lickport {length_of_buff} sec around the start of the reward")
    ax4.axvline(x=length_of_buff, color='red', linestyle='--', label='reward start')
    ax4.legend()
    ax3.legend()
    plt.tight_layout()


def plot_lick_around_time_event(all_buffers, length_of_buff):
    lick_fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
    # Plot each DataFrame in a loop, vertically spaced
    num_of_buffers = len(all_buffers)

    for i, s in enumerate(all_buffers):
        s['lickport_signal'] = s['lickport_signal'] + num_of_buffers - i
        s.plot(kind='scatter', x='timestamp_x', y='lickport_signal', ax=ax3, s=5)
    ax3.axvline(x=length_of_buff, color='red', linestyle='--')
    ax3.set_title('Licks over time')
    ax3.set_xlabel('time')
    ax3.set_ylabel('start licking')

    all_licks = pd.concat(all_buffers)
    all_licks['timestamp_x'].plot(kind='hist',
                                  bins=100,
                                  ax=ax4,
                                  label='licks',
                                  title=f"lickport {length_of_buff} sec around the start of the reward")
    ax4.axvline(x=length_of_buff, color='red', linestyle='--', label='reward start')
    ax4.legend()
    ax3.legend()
    plt.tight_layout()


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


def lickport_processing(bins, group_labels, lickport_trial_merged_df):
    grouped_lickport_by_trial_precentage = lickport_trial_merged_df.groupby(
        pd.cut(lickport_trial_merged_df['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("sum of lickport activations by reward:")
    for condition, group in grouped_lickport_by_trial_precentage:
        print(f"\t{condition}:")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['lickport_signal'].sum()}")
            reward_group.plot(kind='scatter', x='timestamp_x', y='lickport_signal',
                              title=f'lickport of trials {condition}, reward size {condition2}',
                              c=reward_group['reward_size'], cmap='viridis')
        print()
    print(f"\tall reward sizes:{grouped_lickport_by_trial_precentage['lickport_signal'].sum()}")
    print("\n\n")


def velocity_processing(bins, group_labels, velocity_trial_merged_df, config_json):
    grouped_velocity_by_trial_precentage = velocity_trial_merged_df.groupby(
        pd.cut(velocity_trial_merged_df['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("average velocity by reward:")
    for condition, group in grouped_velocity_by_trial_precentage:
        print(f"\t{condition}:")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(
                f"\t\tCondition- reward size {condition2}: {reward_group['Avg_velocity'].mean()}")  # considers all the data points, including all the 0's
            reward_group.plot(x='timestamp_x', y=['Avg_velocity', 'Rolling_Avg_Last_2', 'lickport_signal'],
                              title=f'velocity of trials {condition}, reward size {condition2}')
            plot_velocity_over_position(config_json, reward_group,
                                        f'velocity over position {condition}, reward size {condition2}')
        print()
    print(f"\tall reward sizes:{grouped_velocity_by_trial_precentage['Avg_velocity'].mean()}")
    print("\n\n")


def trial_length_processing(TrialTimeline_df, bins, group_labels):
    grouped_by_trial_precentage = TrialTimeline_df.groupby(
        pd.cut(TrialTimeline_df['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("average trial length by reward:")
    fig, ax = plt.subplots()
    for condition, group in grouped_by_trial_precentage:
        print(f"\t{condition}:")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['trial_length'].mean()}")
            print(f"\t\tnumber of trials: {reward_group['trial_num'].count()}")
            # Plot each reward group's data on the same axis
            reward_group.plot(x='trial_num', y='trial_length', ax=ax,
                              label=f'Length of Trials {condition}, Reward Size {condition2}')

        # Set title and labels for the plot
        ax.set_title(f'Length of Trials for all Reward Sizes')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Trial Length')
        ax.legend()
        print("\n\n")
    plt.figure()
    TrialTimeline_df['trial_length'].plot(kind='hist',
                                          bins=100,
                                          title="Trial Length Distribution",
                                          xlabel='Trial Length(sec)',
                                          ylabel='Frequency')


if __name__ == '__main__':
    post_processing("C:\\Users\\itama\\Desktop\\Virmen_Blue\\28-Dec-2023 103456 Blue_18_DavidParadigm", 0, 100)
