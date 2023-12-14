import pandas as pd
import matplotlib.pyplot as plt


def post_processing(path_of_directory, percentage_from_start, percentage_from_end):
    pd.set_option('display.max_columns', None)

    TrialTimeline_df = pd.read_csv(path_of_directory + "\\TrialTimeline.csv")
    Reward_df = pd.read_csv(path_of_directory + "\\Reward.csv")
    AB_lickport_record_df = pd.read_csv(path_of_directory + "\\A-B_leakport_record.csv")
    velocity_df = pd.read_csv(path_of_directory + "\\velocity.csv")

    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge
    trial_num_bottom_threshold = int(
        TrialTimeline_df.shape[0] * (percentage_from_start / 100))  # precentage to num of trials
    trial_num_top_threshold = int(TrialTimeline_df.shape[0] * (percentage_from_end / 100))
    bins = [float('-inf'), trial_num_bottom_threshold, trial_num_top_threshold, float('inf')]
    group_labels = [f'below {percentage_from_start}', f'between {percentage_from_start}-{percentage_from_end}',
                    f'above {percentage_from_end}']

    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']
    grouped_by_trial_precentage = TrialTimeline_df.groupby(
        pd.cut(TrialTimeline_df['trial_num'], bins=bins, labels=group_labels),
        observed=False)
    print("average trial length by reward:")
    for condition, group in grouped_by_trial_precentage:
        print(f"\t{condition}:")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['trial_length'].mean()}")
            print(f"\t\tnumber of trials: {reward_group['trial_num'].count()}")
            reward_group.plot(x='trial_num', y='trial_length',
                              title=f' length of trials {condition}, reward size {condition2} ')
        print("\n\n")

    AB_lickport_record_df['lickport_signal'] = AB_lickport_record_df['lickport_signal'].round(decimals=0)
    AB_lickport_record_df.loc[AB_lickport_record_df["lickport_signal"] >= 1, "lickport_signal"] = 1
    lickport_trial_merged_df = pd.merge(AB_lickport_record_df, TrialTimeline_df, on='trial_num')
    lickport_trial_merged_df = lickport_trial_merged_df[
        lickport_trial_merged_df['lickport_signal'] != 0]  # take all rows without 0
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

    velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')
    velocity_trial_merged_df['Rolling_Avg_Last_2'] = velocity_trial_merged_df['Avg_velocity'].rolling(2).mean().shift(
        -1)

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
        print()
    print(f"\tall reward sizes:{grouped_velocity_by_trial_precentage['Avg_velocity'].mean()}")
    print("\n\n")

    # plots
    velocity_trial_merged_df.plot(x='timestamp_x', y=['Avg_velocity', 'Rolling_Avg_Last_2'],
                                  title='velocity')
    TrialTimeline_df.plot(x='trial_num', y='trial_length',
                          title='trials length')
    lickport_trial_merged_df.plot(kind='scatter', x='timestamp_x', y='lickport_signal', title='lickport',
                                  c=lickport_trial_merged_df['reward_size'], cmap='viridis')
    # plt.show()


if __name__ == '__main__':
    post_processing("C:\\Users\\itama\\Desktop\\Virmen_Green\\05-Dec-2023 141012 Green_02_DavidParadigm", 20, 90)
