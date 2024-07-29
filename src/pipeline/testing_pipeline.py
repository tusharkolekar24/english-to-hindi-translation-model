import os
import pandas as pd
import numpy as np
import sys
import pickle
import json
sys.path.append('D:/upgraded_git_repo/translation_model')

from src.function import DatasetCleaner, DatasetProcessor, AutoEncoder, ModelManager, model_performance
import warnings
warnings.filterwarnings('ignore')

# Configrating Root Path
root_path = r'D:\upgraded_git_repo\translation_model'

# Connecting Dataset
language_dataset = pd.read_excel(r'D:\upgraded_git_repo\translation_model\dataset\original.xlsx')

# Initialize 
datacleaner = DatasetCleaner(language_dataset.iloc[:,:])

# Initialize Sentence Cleaning
clean_dataset = datacleaner.clean_dataset()

# Initialize Pre-processing Process
processor = DatasetProcessor(dataset=clean_dataset)

# Create Tokens using clean dataset.
token_detail = processor.get_tokenizers()

# Seperate Hindi & English Sentences
english_sentences, hindi_sentences = processor.sentence_formations()

# Prepear Dataset for Model Training
encoder_input_data, decoder_input_data, decoder_target_data = processor.preparing_dataset()

# Initialize ModelManager object
model_config = ModelManager()
file_paths = os.path.join(root_path,'artifacts')

# Load Model to perform predictions
encoder_model_upg = model_config.load_model(file_paths,'encoder_model')
decoder_model_upg = model_config.load_model(file_paths,'decoder_model') 

result = model_performance(english_sentences = english_sentences,
                           hindi_sentences   = hindi_sentences,
                           token_detail      = token_detail,
                           encoder_model     = encoder_model_upg,
                           decoder_model     = decoder_model_upg)

result['token_id'] = np.arange(1,result.shape[0]+1)
result.to_excel(os.path.join(root_path,'dataset','outcome.xlsx'),index=False)