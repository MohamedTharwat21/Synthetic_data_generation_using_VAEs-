def preprocessing(data, option = 1):
    '''
    Args:

        data: numpy array examples x features
        option: 1 for MinMaxScaler and 2 for StandardScaler

    Returns: preprocessed data
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if option == 1:
        processor = MinMaxScaler()
    elif option == 2:
        processor = StandardScaler()
    else:
        return data, None # don't process

    return processor.fit_transform(data), processor





def load_data(folder_name, preprocessing_option = 1):
    # import pandas as pd
    from sdv.datasets.local import load_csvs
    from sdv.datasets.demo import download_demo


    # This is the default folder name that the GOogle Colab notebook uses.
    # Change this if you have your own folder with CSV files.
    # FOLDER_NAME = 'content/'


    try:
        data = load_csvs(folder_name= folder_name)
    except ValueError:
        # Generating syntetic data of the VAE itself
        print('You have not uploaded any csv files. Using some demo data instead.')
        data, _ = download_demo(
            modality='single_table',
            dataset_name='fake_hotels'
        )

    #     preprocessing on the training dataset (Optional)
    # x, _ = preprocessing(x, preprocessing_option)

    return data , preprocessing_option












