import pandas as pd
import matplotlib.pyplot as plt

def post_processing(path_of_directory, percentage):
    pd.set_option('display.max_columns', None)

    TrialTimeline_df = pd.read_csv(path_of_directory+"\\TrialTimeline.csv")
    Reward_df = pd.read_csv(path_of_directory+"\\Reward.csv")
    AB_lickport_record_df = pd.read_csv(path_of_directory+"\\A-B_leakport_record.csv")
    velocity_df = pd.read_csv(path_of_directory+"\\velocity.csv")

    TrialTimeline_df.rename(columns={"trialnum_start": "trial_num"}, inplace=True)  # change column name for merge
    trial_threshold = int(TrialTimeline_df.shape[0]*(percentage/100))
    # time passed from start of trial until reward was given
    TrialTimeline_df['trial_length'] = Reward_df['timestamp_reward_start'] - TrialTimeline_df['timestamp']
    grouped_by_trial_precentage = TrialTimeline_df.groupby(TrialTimeline_df['trial_num'] <= trial_threshold)
    print("average trial lenght by reward:")
    for condition, group in grouped_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['trial_length'].mean()}")
            print(f"\t\tnumber of trials: {reward_group['trial_num'].count()}")
            reward_group.plot(x='trial_num', y='trial_length',
                                  title=f'trials length of first {percentage}%: {condition}, reward size {condition2} ')
        print("\n\n")

    AB_lickport_record_df['lickport_signal'] = AB_lickport_record_df['lickport_signal'].round(decimals=0)
    AB_lickport_record_df.loc[AB_lickport_record_df["lickport_signal"] >= 1, "lickport_signal"] = 1
    lickport_trial_merged_df = pd.merge(AB_lickport_record_df, TrialTimeline_df, on='trial_num')
    grouped_lickport_by_trial_precentage = lickport_trial_merged_df.groupby(
        lickport_trial_merged_df['trial_num'] <= trial_threshold)
    print("sum of lickport activations by reward:")
    for condition, group in grouped_lickport_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['lickport_signal'].sum()}")
        print()
    print(f"\tall reward sizes:{grouped_lickport_by_trial_precentage['lickport_signal'].sum()}")
    print("\n\n")

    velocity_trial_merged_df = pd.merge(velocity_df, TrialTimeline_df, on='trial_num')
    velocity_trial_merged_df['Rolling_Avg_Last_2'] = velocity_trial_merged_df['Avg_velocity'].rolling(2).mean().shift(-1)

    grouped_velocity_by_trial_precentage = velocity_trial_merged_df.groupby(
        velocity_trial_merged_df['trial_num'] <= trial_threshold)
    print("average velocity by reward:")
    for condition, group in grouped_velocity_by_trial_precentage:
        print(f"\tCondition- first {percentage}%: {condition}")
        grouped_by_reward_type = group.groupby('reward_size')
        for condition2, reward_group in grouped_by_reward_type:
            print(f"\t\tCondition- reward size {condition2}: {reward_group['Avg_velocity'].mean()}")  # considers all the data points, including all the 0's
            reward_group.plot(x='timestamp_x', y=['Avg_velocity', 'Rolling_Avg_Last_2', 'lickport_signal'],
                                          title=f'velocity of first {percentage}%: {condition}, reward size {condition2}')
        print()
    print(f"\tall reward sizes:{grouped_velocity_by_trial_precentage['Avg_velocity'].mean()}")
    print("\n\n")

    velocity_trial_merged_df.plot(x='timestamp_x', y=['Avg_velocity', 'Rolling_Avg_Last_2'],
                                  title='velocity')
    TrialTimeline_df.plot(x='trial_num', y='trial_length',
                          title='trials length')
    filtered_lickport_trial_merged_df = lickport_trial_merged_df[lickport_trial_merged_df['lickport_signal'] != 0]  # take all rows without 0
    filtered_lickport_trial_merged_df.plot(kind='scatter', x='timestamp_x', y='lickport_signal', title='lickport',
                                           c=filtered_lickport_trial_merged_df['reward_size'], cmap='viridis')
    plt.show()


if __name__ == '__main__':
    post_processing("C:\\Users\\itama\\Desktop\\Virmen_Green\\05-Dec-2023 141012 Green_02_DavidParadigm", 0)
