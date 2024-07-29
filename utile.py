import pandas as pd
from src.function import DatasetCleaner

class load_dataset:
    def __init__(self):
          pass

    def dataset(self): 
        # Connecting Dataset
        language_dataset = pd.read_excel(r'D:\upgraded_git_repo\translation_model\dataset\original.xlsx')

        # Initialize 
        datacleaner = DatasetCleaner(language_dataset.iloc[:,:])

        # Initialize Sentence Cleaning
        clean_dataset = datacleaner.clean_dataset()

        return clean_dataset