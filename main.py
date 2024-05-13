
def Later() :

    """
     - using argParser to automate the project
     - using Deep learning solution for the training
     - using PYtourch
     - Modify the GANs network on the Graduation Team's Dataset
     - Cross Validation K folds = 5   which LR 94%
     - SVM
     -


    """


def generate_new_data() :
    from Data_Generation import DataFrameOurNewData, Save_our_new_data_set, Save_metadata
    new_synthetic_data = DataFrameOurNewData()
    # print(new_synthetic_data.shape)
    # print(new_synthetic_data.head(10))

    Save_metadata()
    Save_our_new_data_set()



if __name__ == '__main__' :
   generate_new_data()
   import pandas as pd

   # data = pd.read_csv('new_synthetic_data_version_02.csv')
   # print(data)

