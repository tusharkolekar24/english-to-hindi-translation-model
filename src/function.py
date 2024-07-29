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
    def __init__(self, dataset):
        self.dataset = dataset

    def remove_null_values(self):
        samples = self.dataset[(~self.dataset['hindi'].isnull()) & 
                               (~self.dataset['english'].isnull())]
        samples.index = np.arange(0, samples.shape[0])
        samples['Duplicated'] = samples.duplicated()
        return samples

    def clean_dataset(self):
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.english_sentences, self.hindi_sentences = self.sentence_formations()

    def sentence_formations(self):
        english_sentences, hindi_sentences = [], []
        for vals in self.dataset.values:
            english_sentences.append('<start> '+vals[1]+' <end>')
            hindi_sentences.append('<start> '+vals[0]+' <end>')
        return english_sentences, hindi_sentences

    def get_tokenizers(self):
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
        token_detail = self.get_tokenizers()
        # Prepare data for model
        encoder_input_data  = np.array(token_detail.get('english_padded'))
        decoder_input_data  = np.array(token_detail.get('hindi_padded')[:, :-1]) # Remove last token for teacher forcing
        decoder_target_data = np.array(token_detail.get('hindi_padded')[:, 1:])  # Shifted by one for teacher forcing

        return encoder_input_data, decoder_input_data, decoder_target_data
    
class AutoEncoder:
    def __init__(self, embedding_dim, latent_dim, token_detail):
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
        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs):
        self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs)

    def inference_encoder_model(self):
        encoder_model = Model(self.encoder_inputs, self.encoder_states)
        return encoder_model

    def inference_decoder_model(self):
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return decoder_model
    
class ModelManager:
      def __init__(self):
            pass 
        
      def save_model(self,models,file_path,model_name):   
            # serialize model to JSON
            model_json = models.to_json()
            with open(os.path.join(file_path,f"{model_name}.json"), "w") as json_file:
                json_file.write(model_json)

            # serialize weights to HDF5
            models.save_weights(os.path.join(file_path,f"{model_name}.h5"))
            print("Saved model to disk")
            
      def load_model(self,file_path,model_name):
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
    hindi_tokenizer   = token_detail.get('hindi_tokenizer')
    english_tokenizer = token_detail.get('english_tokenizer')
    hindi_maxlen      = token_detail.get('hindi_maxlen')
    english_maxlen    = token_detail.get('english_maxlen')
    
    input_sentence = '<start> ' + input_sentence + ' <end>'
    input_seq = english_tokenizer.texts_to_sequences([input_sentence])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=english_maxlen, padding='post')
    
    states_value = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hindi_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hindi_tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_word
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > hindi_maxlen:
            stop_condition = True
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        states_value = [h, c]
    
    return decoded_sentence

def model_performance(english_sentences,hindi_sentences,token_detail,encoder_model,decoder_model):
    outcome = []
    for eng_sentence,hind_sentence in zip(english_sentences,hindi_sentences):

        input_sentence = eng_sentence.replace("<start>","").replace("<end>","").strip()

        translated_sentence = translate_sentence_to_hindi(input_sentence,token_detail,encoder_model,decoder_model)
        
        actual_translation   = hind_sentence.replace("<start>","").replace("<end>","").strip()
        predict_translation  = translated_sentence.replace("<start>","").replace("<end>","").strip()
        
        fuzz_score = fuzz.ratio(actual_translation,predict_translation)
        print('------------------------------------------------------------')
        print('Input:', input_sentence)
        print('Translated (Pred):', predict_translation)
        print('Translated (Actual):', actual_translation)
        print("Fuzzy Score:",fuzz_score)
        print('------------------------------------------------------------')
        outcome.append([input_sentence,actual_translation,predict_translation,fuzz_score])  
        
    result = pd.DataFrame(outcome,columns=['English','Hindi (Actual)','Hindi (Pred)','Score'])
    
    true_positive  =  result[result['Score']>=100].shape[0]
    false_positive =  result[result['Score']<100].shape[0]
    
    precision      = true_positive/(true_positive+false_positive)
    
    print(f"Pricision of the Model:{precision}")
    
    return result