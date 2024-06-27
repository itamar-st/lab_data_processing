import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TONE_REWARD_DELAY = 0.5


##--------------------------------------------------     Function from main - do not change ---------------------------------
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


def plot_lick_around_time_event(stats_df, all_buffers, length_of_buff, title, x_axis):
    # Update to 3 rows, 1 column for subplots
    lick_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))  # 3 rows, 1 column

    # Plot each DataFrame in a loop, vertically spaced, on the first subplot
    for i, s in enumerate(all_buffers):
        s['location_in_scatter'] = i
        s.plot(kind='scatter', x=x_axis, y='location_in_scatter', ax=ax1, s=5)
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.axvline(x=-TONE_REWARD_DELAY, color='green', linestyle='--')
    ax1.set_title(title + ' -- Licks over time')
    ax1.set_xlabel('time')
    ax1.set_ylabel('start licking')

    # Second subplot for the calculated PDF histogram
    all_licks = pd.concat(all_buffers)
    hist_title = f"lickport {length_of_buff} sec around the start of the reward"
    bin_width = 0.05  # Fine granularity for more bars
    min_time, max_time = -length_of_buff, length_of_buff
    bins = np.arange(min_time, max_time + bin_width, bin_width)
    counts, bin_edges = np.histogram(all_licks[x_axis], bins=bins)
    probabilities = counts / counts.sum()  # Convert counts to probabilities
    ax2.bar(bin_edges[:-1], probabilities, width=bin_width, align='edge')
    ax2.axvline(x=0, color='red', linestyle='--', label='reward start')
    ax2.axvline(x=-TONE_REWARD_DELAY, color='green', linestyle='--', label='tone start')

    ax2.set_ylabel('Probability')
    ax2.legend()

    df = pd.DataFrame({title + " probabilities": probabilities})
    stats_df = pd.concat([stats_df, df], axis=1)

    # Third subplot for the additional histogram with 100 bins
    ax3.set_ylim([0, 120])
    ax3.hist(all_licks[x_axis], bins=100, label='licks', color='green', alpha=0.6)
    ax3.set_title(hist_title)
    ax3.axvline(x=0, color='red', linestyle='--', label='reward start')
    ax3.axvline(x=-TONE_REWARD_DELAY, color='green', linestyle='--', label='tone start')
    ax3.set_ylabel('Amount')
    ax3.legend()

    plt.tight_layout()
    return stats_df


def calc_licks_around_time_event(stats_df, lickport_trial_merged_df_with_zeros, reward_times, title, all_buffers):
    length_of_buff = 1  # time buffer around the start of the trial/reward
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


## ---------------------------------------------------------------------------------------------------------------------------


def convert_leonardo_csv(file_path):
    df = pd.read_csv(file_path)
    # fill the missing data between tone start and reward start
    fill_misssing_data(df, 'Dev1/port0/line7 S output', 'Tone S output', TONE_REWARD_DELAY)
    # fill the missing data between reward start and reward end
    fill_misssing_data(df, 'Dev1/port0/line7 F output', 'Dev1/port0/line7 S output', 0)
    # do it again incase we found new missing rows in the reward
    fill_misssing_data(df, 'Dev1/port0/line7 S output', 'Tone S output', TONE_REWARD_DELAY)
    if df['Dev1/port0/line7 S output'].count() != df['Dev1/port0/line7 F output'].count() or \
            df['Dev1/port0/line7 S output'].count() != df['Tone S output'].count():
        print("missing data - reward_start reward_end sound_start")
    # create the format for the preprocessing script
    TrialTimeline_df, reward_df = add_reward_and_trialnum(df)

    lickport_df = lickport_preprocessing(TrialTimeline_df, df)

    reward_time_range = reward_df['timestamp_reward_start'].values.tolist()
    bins = [float('-inf'), 0, 200, float('inf')]
    group_labels = ['below 0%', 'between 0%-100%', 'above 100%']
    results, calc_vectors = lickport_processing(pd.DataFrame(), bins, group_labels, lickport_df, pd.DataFrame(),
                                                TrialTimeline_df,
                                                reward_time_range, reward_df)
    plt.show()
    # fig = plt.gcf()
    # plt.close(fig)  # Close the figure to prevent it from displaying now
    # return fig


def lickport_preprocessing(TrialTimeline_df, df):
    lickport_df = pd.DataFrame({
        'timestamp': df['Dev1/ai13 G input'],
    })
    lickport_df = lickport_df.dropna()
    lickport_df = add_trial_num_to_raw_data(lickport_df, TrialTimeline_df)
    lickport_df = pd.merge(lickport_df, TrialTimeline_df, on='trial_num')
    lickport_df['lickport_signal'] = [1] * lickport_df['trial_num'].count()
    # Duplicate the DataFrame
    duplicated_df = lickport_df.copy()
    duplicated_df['lickport_signal'] = 0  # Set 'lickport_signal' to 0 for duplicated rows
    # We will interleave without using fractional indices
    # Add a new index column to help with interleaving
    lickport_df['new_index'] = range(1, len(lickport_df) * 2, 2)
    duplicated_df['new_index'] = range(0, len(duplicated_df) * 2, 2)
    # Concatenate the original and duplicated DataFrame
    combined_df = pd.concat([lickport_df, duplicated_df])
    # Sort by new index and reset the index
    lickport_df = combined_df.sort_values('new_index').reset_index(drop=True)
    lickport_df.drop('new_index', axis=1, inplace=True)  # Remove the auxiliary 'new_index' column
    return lickport_df


def add_reward_and_trialnum(df):
    # check if there are different reward values beside big and small
    reward_lengths = (df['Dev1/port0/line7 F output'] - df['Dev1/port0/line7 S output']).round(4)
    unique_reward_times = reward_lengths.unique().tolist()

    if len(unique_reward_times) > 2:
        print(f"unique_reward_times : {unique_reward_times}")
    if 0.0 in unique_reward_times:
        index = reward_lengths.loc[reward_lengths == 0.0].index
        print(f"missing reward value in lines : {index}")

    big_reward_val = 30
    small_reward_val = 8
    # recrate the TrialTimeline_df format
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
    return TrialTimeline_df, reward_df


# fill gaps in the original file. get 2 columns with known time difference and fill the missing rows accordingly
def fill_misssing_data(df, clo_A, col_B, shift):
    # shift one row and repeat
    while True:
        # find the time diff between the columns
        df['diff'] = df[clo_A] - df[col_B]
        # find the time diff between the difference. suppose to be small
        df['diff2'] = df['diff'] - df['diff'].shift(1)
        df['diff2'][0] = df['diff'][0] - shift
        diff = df['diff2']
        indices1 = diff[diff < -2].index
        indices2 = diff[diff > 2].index
        # sort the value to get the first missing value
        indices = sorted(indices1.tolist() + indices2.tolist())
        # no more missing values
        if not indices:
            break
        idx = indices[0]  # get the first missing value
        # shift the first col or the second one depending on the difference
        if idx in indices1:
            col_to_change = col_B
            correct_column = clo_A
            amount_of_change = - shift
        else:
            col_to_change = clo_A
            correct_column = col_B
            amount_of_change = shift
        df.loc[idx + 1:, col_to_change] = df.loc[idx:, col_to_change].shift(1)
        df.at[idx, col_to_change] = df.at[idx, correct_column] + amount_of_change  # Set the current index value to NaN


if __name__ == '__main__':
    path = "C:\\Users\\itama\\Downloads\\p_data\\David_P10_2Tone_PredictiveLicking_-13-06-2024.csv"

    convert_leonardo_csv(path)
