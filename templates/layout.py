from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from utile import load_dataset

language_dataset = load_dataset().dataset()

# Define form layout using dbc.Row and dbc.Col
app_layout = dbc.Container([
    html.H1("Translate English to Hindi",style={'margin-bottom':'25px'}),
    dbc.Row(
    [
        dbc.Col(
            [   html.H5("Select token_ID:"),
                dcc.Dropdown(
                    id='pandas-dropdown-2',
                    options=[{'label': tokenid, 'value': tokenid} for tokenid in language_dataset['token_id'].unique()],
                    value = '{}'.format(language_dataset['token_id'].unique()[0]),
                    placeholder="Select a nation...",
                    clearable=False
                ),
            ],
            width=2,
        ),
        dbc.Col(
            [   html.H5("Select English sentences by using token_id"),
                dbc.Textarea(
                    id='input-text',
                    placeholder='Enter text in English suggested by Token_id',
                    rows=5
                ),
            ],
            width=9,
        ),
    ],
    className="g-3 mb-3",
),
    
dbc.Row(
    [
        dbc.Col(
            [

            ],
            width=2,
        ),
        dbc.Col(
            [   html.H5("Traslated Text into Hindi (Actual):"),
                dbc.Textarea(
                    id='actual-text',
                    placeholder='Enter text in English suggested by Token_id',
                    rows=5
                ),
            ],
            width=9,
        ),
    ],
    className="g-3 mb-3",
),
    
dbc.Row(
    [
        dbc.Col(
            [

            ],
            width=2,
        ),
        dbc.Col(
            [   html.H5("Traslated Text into Hindi (Predicted):"),
                dbc.Textarea(
                    id='pred-text',
                    placeholder='Enter text in English suggested by Token_id',
                    rows=5
                ),
                
               html.Div(id='output-text',style={'margin-top':'25px'})
               # html.H5("Score of Text Traslated : 100",style={'margin-top':'25px'}),
            ],
            width=9,
        ),
    ],
    className="g-3",
)

],
className="mt-4")