import os
from fuzzywuzzy import fuzz
from utile import load_dataset
from templates.layout import app_layout
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from src.function import DatasetProcessor, ModelManager, translate_sentence_to_hindi


# Configrating Root Path
root_path = os.path.join(os.getcwd())

# Load sample data
clean_dataset = load_dataset().dataset()

# Initialize Pre-processing Process
processor = DatasetProcessor(dataset=clean_dataset)

# Create Tokens using clean dataset.
token_detail = processor.get_tokenizers()


# Initialize ModelManager object
model_config = ModelManager()
file_paths = os.path.join(root_path,'artifacts')

# Load Model to perform predictions
encoder_model_upg = model_config.load_model(file_paths,'encoder_model')
decoder_model_upg = model_config.load_model(file_paths,'decoder_model')


# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define app layout
app.layout = app_layout

# Define callback to update dbc.Textarea based on dropdown selection
@app.callback(
    [Output('input-text', 'value'),Output('actual-text', 'value'),Output('pred-text', 'value'),
     Output('output-text', 'children')],  # Update the 'value' property of dbc.Textarea
    [Input('pandas-dropdown-2', 'value')]  # Trigger callback when dropdown value changes
)
def update_textarea(token_id):
    if token_id!= None:
        sample = clean_dataset[clean_dataset['token_id']==token_id]
        input_sentence = sample['english'].values[0]
        hind_sentence = sample['hindi'].values[0]
        
        translated_sentence = translate_sentence_to_hindi(input_sentence,token_detail,encoder_model_upg,decoder_model_upg)
        
        actual_translation   = hind_sentence.strip()
        predict_translation  = translated_sentence.replace("<start>","").replace("<end>","").strip()
        fuzz_score = fuzz.ratio(actual_translation,predict_translation)   
        
        return input_sentence, actual_translation,predict_translation,html.H5(f"Score of Text Traslated : {fuzz_score}")
    else:
        return "","","",html.H5("Score of Text Traslated : 0")

if __name__ == '__main__':
    app.run_server(debug=False)
    
