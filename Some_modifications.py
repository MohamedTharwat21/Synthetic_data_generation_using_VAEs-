import pandas as pd
def modify2() :
    import pandas as pd
    df = pd.read_csv('new_synthetic_data_version_04_saturday.csv')
    df.loc[df['body_temperature'] >= 40, 'health_status'] = 'unhealthy'
    df.loc[(df['respiratory_rate'] >= 15) &( df['respiratory_rate'] <= 30) , 'health_status'] = 'healthy'
    # df.loc[df['heart_rate'] >= 55 & df['heart_rate'] <= 80 , 'health_status'] = 'healthy'
    df.loc[(df['heart_rate'] >= 55) & (df['heart_rate'] <= 80), 'health_status'] = 'healthy'

    # df.to_csv('new_data_v3.csv')

     # Remove the first column
    # df = df.drop(df.columns[0], axis=1)
    # Remove the first column using column indexing
    # df = df.iloc[:, 1:]
    df.to_csv('new_data_v5_2000.csv')


def modify() :
    import pandas as pd
    # Example DataFrame
    # data = {'A': [1, 2, 3, 4, 5],
    #         'B': [10, 20, 30, 40, 50]}
    # df = pd.DataFrame(data)

    df = pd.read_csv('new_synthetic_data_version_01.csv')

    df_modified = df['health_status']

    # Change the value of col2 to 'unhealthy' where col1 is greater than 40
    df.loc[df['body_temperature'] >= 40, 'health_status'] = 'unhealthy'

    # Count the number of modifications
    # num_modifications = (df['health_status'] != df_modified['health_status']).sum()

    # print("Number of modifications:", num_modifications)


    # dd = pd.read_csv('new_data_v1.csv')
    # print(dd.isna().sum())
    # print(df)
    # df.to_csv('new_data_v1.csv')

    # def count() :
    #     # Make a copy of the original DataFrame
    #     df_modified = df_original.copy()
    #
    #     # Change the value of col2 to 'unhealthy' where col1 is greater than 40
    #     df_modified.loc[df_modified['col1'] > 40, 'col2'] = 'unhealthy'
    #
    #     # Count the number of modifications
    #     num_modifications = (df_original['col2'] != df_modified['col2']).sum()
    #
    #     print("Number of modifications:", num_modifications)

if __name__ == '__main__' :
    modify2()