import pandas as pd
import numpy as np
import pickle
import re
import os
import json
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from fuzzywuzzy import fuzz
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model, model_from_json

class DatasetCleaner:
    """
    A class to clean and preprocess a dataset of bilingual text pairs.
    
    Attributes:
        dataset (pd.DataFrame): The dataset to be cleaned, expected to have columns for Hindi and English texts.
    """
    def __init__(self, dataset):
        """
        Initializes the DatasetCleaner with the provided dataset.
        
        Parameters:
            dataset (pd.DataFrame): The dataset to be cleaned, must include 'hindi' and 'english' columns.
        """
        self.dataset = dataset

    def remove_null_values(self):
        """
        Removes rows with null values in either the Hindi or English text columns.
        
        Also resets the index of the resulting DataFrame and adds a 'Duplicated' column
        to indicate if a row is duplicated.
        
        Returns:
            pd.DataFrame: The dataset with null values removed and index reset.
        """
        samples = self.dataset[(~self.dataset['hindi'].isnull()) & 
                               (~self.dataset['english'].isnull())]
        samples.index = np.arange(0, samples.shape[0])
        samples['Duplicated'] = samples.duplicated()
        return samples

    def clean_dataset(self):
        """
        Cleans and preprocesses the dataset by:
        1. Removing rows with '0, 0' in the Hindi text column.
        2. Stripping unwanted characters and whitespace from both Hindi and English text columns.
        3. Removing specific patterns from Hindi text.
        4. Collecting the cleaned data into a new DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned dataset with columns 'hindi', 'english', and 'token_id'.
        """
        sampleset   = self.remove_null_values()
        cleanset_list = []
        for text in sampleset.values:
            if text[0] != '0, 0':
                english_text = text[1].replace(" _ ", "").strip()
                hindi_text = text[0].replace(":", "").replace(".", "").replace('& lt; bgt;', '').replace('lt;/bgt;', '').strip()

                if hindi_text.endswith(")"):
                    for pattern in [r'\(_ [A-Za-z]\)', r'\([A-Za-z] _\)', r' _ \([A-Za-z]\)', r'\([A-Za-z]\)']:
                        hindi_text = re.sub(pattern, "", hindi_text)

                if english_text.startswith("_"):
                    english_text = english_text.replace("_", "").strip()

                english_text = english_text.replace(".", "").replace(':', '').strip()
                
                cleanset_list.append([hindi_text, english_text,text[3]])

        cleansets = pd.DataFrame(cleanset_list, columns=['hindi', 'english','token_id'])
        return cleansets
    
class DatasetProcessor:
    """
    A class to process a bilingual dataset for machine translation tasks.
    
    Attributes:
        dataset (pd.DataFrame): A DataFrame containing bilingual sentences with 'english' and 'hindi' columns.
        english_sentences (list): List of English sentences formatted for tokenization.
        hindi_sentences (list): List of Hindi sentences formatted for tokenization.
    """
    def __init__(self, dataset):
        """
        Initializes the DatasetProcessor with the provided dataset and prepares sentence formations.
        
        Parameters:
            dataset (pd.DataFrame): A DataFrame with columns 'english' and 'hindi'.
        """
        self.dataset = dataset
        self.english_sentences, self.hindi_sentences = self.sentence_formations()

    def sentence_formations(self):
        """
        Formats sentences by adding start and end tokens.
        
        Returns:
            tuple: Two lists containing formatted English and Hindi sentences.
        """
        english_sentences, hindi_sentences = [], []
        for vals in self.dataset.values:
            english_sentences.append('<start> '+vals[1]+' <end>')
            hindi_sentences.append('<start> '+vals[0]+' <end>')
        return english_sentences, hindi_sentences

    def get_tokenizers(self):
        """
        Tokenizes and pads the sentences for both English and Hindi languages.
        Tokenization and padding include:
        - Creating tokenizers for each language.
        - Fitting tokenizers on the respective sentence lists.
        - Converting sentences to sequences of integers.
        - Padding sequences to ensure uniform length.

        Returns:
            dict: A dictionary containing tokenizers and padded sequences for both languages, 
                  as well as the maximum sequence lengths.
        """
        # Tokenization and padding
        hindi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        hindi_tokenizer.fit_on_texts(self.hindi_sentences)
        hindi_sequences = hindi_tokenizer.texts_to_sequences(self.hindi_sentences)
        hindi_maxlen = max(len(seq) for seq in hindi_sequences)
        hindi_padded = tf.keras.preprocessing.sequence.pad_sequences(hindi_sequences, maxlen=hindi_maxlen, padding='post')

        english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        english_tokenizer.fit_on_texts(self.english_sentences)
        english_sequences = english_tokenizer.texts_to_sequences(self.english_sentences)
        english_maxlen = max(len(seq) for seq in english_sequences)
        english_padded = tf.keras.preprocessing.sequence.pad_sequences(english_sequences, maxlen=english_maxlen, padding='post')

        self.token_detail = {
            "hindi_tokenizer": hindi_tokenizer,
            "english_tokenizer": english_tokenizer,
            "hindi_padded": hindi_padded,
            "english_padded": english_padded,
            'hindi_maxlen': hindi_maxlen,
            'english_maxlen': english_maxlen
        }
        
        return self.token_detail

    def preparing_dataset(self):
        """
        Prepares data for training the translation model.
        
        This includes:
        - Encoding English sentences as input data.
        - Preparing decoder inputs by removing the last token from padded Hindi sequences (for teacher forcing).
        - Preparing decoder targets by shifting the padded Hindi sequences (for teacher forcing).

        Returns:
            tuple: Three numpy arrays representing encoder input data, decoder input data, and decoder target data.
        """
        token_detail = self.get_tokenizers()
        # Prepare data for model
        encoder_input_data  = np.array(token_detail.get('english_padded'))
        decoder_input_data  = np.array(token_detail.get('hindi_padded')[:, :-1]) # Remove last token for teacher forcing
        decoder_target_data = np.array(token_detail.get('hindi_padded')[:, 1:])  # Shifted by one for teacher forcing

        return encoder_input_data, decoder_input_data, decoder_target_data
    
class AutoEncoder:
    """
    A class to define and manage an AutoEncoder model for sequence-to-sequence tasks, such as machine translation.

    Attributes:
        embedding_dim (int): Dimension of the embedding layer.
        latent_dim (int): Dimension of the LSTM hidden state.
        token_detail (dict): Dictionary containing tokenizers and padding details for the datasets.
        vocab_size (int): Size of the vocabulary for the English language (input).
        hindi_tokenizer (Tokenizer): Tokenizer for the Hindi language (output).
        encoder_inputs (Input): Input layer for the encoder.
        encoder_embedding (Embedding): Embedding layer for the encoder.
        encoder_lstm (LSTM): LSTM layer for the encoder.
        encoder_states (list): List of encoder LSTM states.
        decoder_inputs (Input): Input layer for the decoder.
        decoder_embedding (Embedding): Embedding layer for the decoder.
        decoder_lstm (LSTM): LSTM layer for the decoder.
        decoder_dense (Dense): Dense layer for the decoder outputs.
        decoder_outputs (Tensor): Output tensor from the decoder.
        model (Model): The complete AutoEncoder model.
    """
    def __init__(self, embedding_dim, latent_dim, token_detail):
        """
        Initializes the AutoEncoder with specified dimensions and token details.

        Parameters:
            embedding_dim (int): Dimension of the embedding layer.
            latent_dim (int): Dimension of the LSTM hidden state.
            token_detail (dict): Dictionary containing tokenizers and padding details for the datasets.
        """
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.token_detail = token_detail
        self.vocab_size = len(token_detail.get("english_tokenizer").word_index) + 1
        self.hindi_tokenizer = token_detail.get('hindi_tokenizer')

        self.encoder_inputs = Input(shape=(None,))
        self.encoder_embedding = Embedding(self.vocab_size, self.embedding_dim)(self.encoder_inputs)
        self.encoder_lstm = LSTM(self.latent_dim, return_state=True)
        self.encoder_outputs, self.state_h, self.state_c = self.encoder_lstm(self.encoder_embedding)
        self.encoder_states = [self.state_h, self.state_c]

        self.decoder_inputs = Input(shape=(None,))
        self.decoder_embedding = Embedding(len(self.hindi_tokenizer.word_index) + 1, self.embedding_dim)(self.decoder_inputs)
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=self.encoder_states)
        self.decoder_dense = Dense(len(self.hindi_tokenizer.word_index) + 1, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles the AutoEncoder model.

        Returns:
            Model: The compiled Keras model.
        """
        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs):
        """
        Trains the AutoEncoder model.

        Parameters:
            encoder_input_data (numpy.ndarray): Input data for the encoder.
            decoder_input_data (numpy.ndarray): Input data for the decoder.
            decoder_target_data (numpy.ndarray): Target data for the decoder.
            batch_size (int): Size of each training batch.
            epochs (int): Number of epochs to train the model.
        """
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs)

    def inference_encoder_model(self):
        """
        Creates and returns the encoder model for inference.

        Returns:
            Model: The Keras model used for encoding inputs during inference.
        """
        encoder_model = Model(self.encoder_inputs, self.encoder_states)
        return encoder_model

    def inference_decoder_model(self):
        """
        Creates and returns the decoder model for inference.

        Returns:
            Model: The Keras model used for decoding sequences during inference.
        """
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return decoder_model
    
class ModelManager:
    """
    A class to manage saving and loading Keras models.

    Methods:
        save_model(models, file_path, model_name):
            Saves the Keras model architecture and weights to disk.
        load_model(file_path, model_name):
            Loads a Keras model from disk using saved architecture and weights.
    """
    def __init__(self):
        """
        Initializes the ModelManager class.
        """
        pass 
        
    def save_model(self,models,file_path,model_name):  
        """
        Saves the Keras model architecture and weights to disk.

        Parameters:
            models (tf.keras.Model): The Keras model to be saved.
            file_path (str): Directory path where the model files will be saved.
            model_name (str): The name to be used for the saved model files (without extension).
        
        This method saves:
            - The model architecture to a JSON file.
            - The model weights to an HDF5 file.
        """ 
        # serialize model to JSON
        model_json = models.to_json()
        with open(os.path.join(file_path,f"{model_name}.json"), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        models.save_weights(os.path.join(file_path,f"{model_name}.h5"))
        print("Saved model to disk")
            
    def load_model(self,file_path,model_name):
        """
        Loads a Keras model from disk.

        Parameters:
            file_path (str): Directory path where the model files are located.
            model_name (str): The name of the saved model files (without extension).
        
        Returns:
            tf.keras.Model: The loaded Keras model.
        
        This method:
            - Loads the model architecture from a JSON file.
            - Loads the model weights from an HDF5 file.
        """
        # load json and create model
        json_file = open(os.path.join(file_path,f"{model_name}.json"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(file_path,f"{model_name}.h5"))
        print("Loaded model from disk")

        return loaded_model
      
# Function to translate a Hindi sentence to English
def translate_sentence_to_hindi(input_sentence,token_detail,encoder_model,decoder_model):
    """
    Translates an input English sentence into Hindi using a trained sequence-to-sequence model.

    Parameters:
        input_sentence (str): The English sentence to be translated.
        token_detail (dict): A dictionary containing tokenizers and padding details.
        encoder_model (tf.keras.Model): The encoder model for the sequence-to-sequence architecture.
        decoder_model (tf.keras.Model): The decoder model for the sequence-to-sequence architecture.

    Returns:
        str: The translated Hindi sentence.
    """

    # Retrieve necessary tokenizers and padding details
    hindi_tokenizer   = token_detail.get('hindi_tokenizer')
    english_tokenizer = token_detail.get('english_tokenizer')
    hindi_maxlen      = token_detail.get('hindi_maxlen')
    english_maxlen    = token_detail.get('english_maxlen')
    
    # Preprocess the input sentence
    input_sentence = '<start> ' + input_sentence + ' <end>'
    input_seq = english_tokenizer.texts_to_sequences([input_sentence])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=english_maxlen, padding='post')
    
    # Encode the input sequence
    states_value = encoder_model.predict(input_seq)
    
    # Initialize the target sequence with the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hindi_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''

    # Decode the sequence one token at a time
    while not stop_condition:
        # Predict the next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Find the most probable token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hindi_tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word
        
        # Check for end token or max sentence length
        if sampled_word == '<end>' or len(decoded_sentence.split()) > hindi_maxlen:
            stop_condition = True
        
        # Update the target sequence and states for the next token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        states_value = [h, c]
    
    return decoded_sentence

def model_performance(english_sentences,hindi_sentences,token_detail,encoder_model,decoder_model):
    """
    Evaluates the performance of the translation model by comparing predicted translations with actual translations.

    Parameters:
        english_sentences (list of str): A list of English sentences to be translated.
        hindi_sentences (list of str): A list of corresponding actual Hindi translations.
        token_detail (dict): A dictionary containing tokenizers and padding details for the languages.
        encoder_model (tf.keras.Model): The encoder part of the sequence-to-sequence model.
        decoder_model (tf.keras.Model): The decoder part of the sequence-to-sequence model.

    Returns:
        pd.DataFrame: A DataFrame containing the input English sentences, actual Hindi translations, predicted Hindi translations, and their Fuzzy scores.
    """
    outcome = []
    for eng_sentence,hind_sentence in zip(english_sentences,hindi_sentences):
        # Remove special tokens from the input English sentence
        input_sentence = eng_sentence.replace("<start>","").replace("<end>","").strip()

        # Translate the English sentence to Hindi
        translated_sentence = translate_sentence_to_hindi(input_sentence,token_detail,encoder_model,decoder_model)
        
        # Remove special tokens from the actual and predicted Hindi sentences
        actual_translation   = hind_sentence.replace("<start>","").replace("<end>","").strip()
        predict_translation  = translated_sentence.replace("<start>","").replace("<end>","").strip()
        
        # Calculate the Fuzzy score for the translated sentence
        fuzz_score = fuzz.ratio(actual_translation,predict_translation)

        # Print detailed information for the current sentence
        print('------------------------------------------------------------')
        print('Input:', input_sentence)
        print('Translated (Pred):', predict_translation)
        print('Translated (Actual):', actual_translation)
        print("Fuzzy Score:",fuzz_score)
        print('------------------------------------------------------------')
        outcome.append([input_sentence,actual_translation,predict_translation,fuzz_score])  
        
    # Convert the outcome list to a DataFrame for easier analysis
    result = pd.DataFrame(outcome,columns=['English','Hindi (Actual)','Hindi (Pred)','Score'])
    
    # Calculate performance metrics
    true_positive  =  result[result['Score']>=100].shape[0]
    false_positive =  result[result['Score']<100].shape[0]
    
    # Calculate precision
    precision      = true_positive/(true_positive+false_positive)
    print(f"Pricision of the Model:{precision}")
    
    return result