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

# Initialize AutoEncoder object
autoencoder = AutoEncoder(embedding_dim=50, 
                          latent_dim=256, 
                          token_detail=token_detail)

# Train the model
autoencoder.train(encoder_input_data = encoder_input_data, 
                  decoder_input_data = decoder_input_data, 
                  decoder_target_data = decoder_target_data, 
                  batch_size=1, 
                  epochs=10)

# Get inference models
encoder_model = autoencoder.inference_encoder_model()
decoder_model = autoencoder.inference_decoder_model()


# Initialize ModelManager object
model_config = ModelManager()
file_paths = os.path.join(root_path,'artifacts')

for model_, name_ in zip([encoder_model,decoder_model],['encoder_model','decoder_model']):
    # Save models
    model_config.save_model(model_, file_paths, name_)
