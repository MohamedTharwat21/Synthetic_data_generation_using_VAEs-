# sdv


# https://docs.sdv.dev/sdv


def load_data_as_dictionary():
    # Data Preparation and Preprocessing

    from data_helper import load_data

    folder_name = r'E:\Ahemd Tarek L4\TabularDataGeneration00\Dataset02'
    data, _ = load_data(folder_name , preprocessing_option= 3)  # any number if the data already processed

    # data in dictionary
    return data


# def dic_of_dataframe():
#     data = load_data_as_dictionary()
#     # print(f'Data keys are {data.keys()}')
#
#     respiratory_rate = data['respiratory_rate']
#     # print(respiratory_rate)
#
#     return data



def Multi_Table_Metadata():

    # use the head method to inspect the first few rows of the data
    from sdv.metadata import MultiTableMetadata


    data = load_data_as_dictionary()

    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(data)

    return metadata


def Single_Table_Metadata():
    # use the head method to inspect the first few rows of the data
    from sdv.metadata import SingleTableMetadata
    data = load_data_as_dictionary()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframes(data)

    return metadata




def Validation(option='multi'):

    if option == 'single':
        metadata = Single_Table_Metadata()

    elif option == 'multi':
        metadata = Multi_Table_Metadata()


    # here is the validation
    metadata.validate()




def Save_metadata():

    metadata = Multi_Table_Metadata()
    # Focus on metadata name , change it every time
    # you need to generate new data
    # before u save it :)

    metadata.save_to_json('metadata3.json')


def Visualize_meta_data():
    from sdv.metadata import MultiTableMetadata
    # in the future, you can reload the metadata object from the file
    metadata = MultiTableMetadata.load_from_json('metadata1.json')
    print('Auto detected data:\n')

    # there is a problem with visualize()
    # i'll fix it later

    metadata.visualize()


def print_metadata():
    metadata = Multi_Table_Metadata()

    # Pay attention it is a Json file

    print(metadata)




def Generate_synthetic_data():
    from sdv.multi_table import HMASynthesizer

    data = load_data_as_dictionary()
    metadata = Multi_Table_Metadata()


    # initialize the synthesizer
    synthesizer = HMASynthesizer(metadata)

    # Training
    synthesizer.fit(data)



    # sample(scale=1) :

    """Generate synthetic data for the entire dataset.
    Args:
        scale (float):
            A float representing how much to scale the data by. If scale is set to ``1.0``,
            this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
            create more rows than the original data by a factor of ``scale``.
            If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
            than the original tables. Defaults to ``1.0``.
    """



    # here our new synthetic data
    synthetic_data = synthesizer.sample(scale=2)



    """
       synthetic_data['smalldata89modeified'] 
       as we deal with synthetic_data like a schema of Database
       we want just on table of it which is "smalldata89modeified"
       that is why we in Multitable

    """

    def print_info_about_synthetic_data():
        # print(len(synthetic_data))
        print(len(synthetic_data['smalldata89modeified']))

        # synthetic_data shape
        print(synthetic_data['smalldata89modeified'].shape)

        # show the first 10 rows of the synthetic_data
        print(synthetic_data['smalldata89modeified'].head(10))

    return synthetic_data, print_info_about_synthetic_data


def DataFrameOurNewData():

    import pandas as pd
    synthetic_data, _ = Generate_synthetic_data()

    Dataframe_synthetic_data = pd.DataFrame(synthetic_data['new_data_v3'])
    # print(Dataframe_synthetic_data.head(10))

    return Dataframe_synthetic_data




def Save_our_new_data_set():
    Dataframe_synthetic_data = DataFrameOurNewData()
    # save it
    Dataframe_synthetic_data.to_csv("new_synthetic_data_version_04_saturday.csv", index=False)
    print('New Dataset Saved Successfully ....')




