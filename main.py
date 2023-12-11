import csv
import pandas as pd
import scipy.io


def post_processing(percentage):
    pd.set_option('display.max_columns', None)

    TrialTimeline_df = pd.read_csv(
        "C:\\Users\\itama\\Desktop\\Virmen_Blue\\06-Dec-2023 124900 Blue_03_DavidParadigm\\TrialTimeline.csv")
    Reward_df = pd.read_csv(
        "C:\\Users\\itama\\Desktop\\Virmen_Blue\\06-Dec-2023 124900 Blue_03_DavidParadigm\\Reward.csv")
    AB_lickport_record_df = pd.read_csv(
        "C:\\Users\\itama\\Desktop\\Virmen_Blue\\06-Dec-2023 124900 Blue_03_DavidParadigm\\A-B_leakport_record.csv")
    velocity_df = pd.read_csv(
        "C:\\Users\\itama\\Desktop\\Virmen_Blue\\06-Dec-2023 124900 Blue_03_DavidParadigm\\velocity.csv")

    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge

    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']
    grouped_by_trial_precentage = TrialTimeline_df.groupby(TrialTimeline_df['trial_num'] <= percentage)
    print("average trial lenght by reward:")
    for condition, group in grouped_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['trial_length'].mean()}")
            print(f"\t\tnumber of trials: {reward_group['trial_num'].count()}")
        print()

    AB_lickport_record_df['lickport_signal'] = AB_lickport_record_df['lickport_signal'].round(decimals=0)
    lickport_trial_merged_df = pd.merge(AB_lickport_record_df, TrialTimeline_df, on='trial_num')
    grouped_lickport_by_trial_precentage = lickport_trial_merged_df.groupby(
        lickport_trial_merged_df['trial_num'] <= percentage)
    print("sum of lickport activations by reward:")
    for condition, group in grouped_lickport_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['lickport_signal'].sum()}")
        print()
    print(f"\tall reward sizes:{grouped_lickport_by_trial_precentage['lickport_signal'].sum()}")

    velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')
    grouped_velocity_by_trial_precentage = velocity_trial_merged_df.groupby(
        velocity_trial_merged_df['trial_num'] <= percentage)
    print("average velocity by reward:")
    for condition, group in grouped_velocity_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['Avg_velocity'].mean()}")  # considers all the data points, including all the 0's
        print()
    print(f"\tall reward sizes:{grouped_velocity_by_trial_precentage['Avg_velocity'].mean()}")


if __name__ == '__main__':
    post_processing(20)
