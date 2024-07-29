import pandas as pd
from src.function import DatasetCleaner
import os
# r'D:\upgraded_git_repo\translation_model\dataset\original.xlsx')
class load_dataset:
    """
    A class to handle loading and cleaning a dataset.
    """
    def __init__(self):
        """
        Initializes the LoadDataset class.
        """
        pass

    def dataset(self): 
        """
        Loads and cleans the dataset.

        This method performs the following steps:
        1. Loads the dataset from an Excel file located in the 'dataset' directory.
        2. Initializes a DatasetCleaner instance with the loaded data.
        3. Cleans the dataset using the DatasetCleaner instance.
        4. Returns the cleaned dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        # Connecting Dataset
        language_dataset = pd.read_excel(os.path.join(os.getcwd(),'dataset','original.xlsx'))
                                        

        # Initialize 
        datacleaner = DatasetCleaner(language_dataset.iloc[:,:])

        # Initialize Sentence Cleaning
        clean_dataset = datacleaner.clean_dataset()

        return clean_dataset