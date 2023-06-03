            ### Importation of libraries ###

# Import flask.
from flask import Flask, request, jsonify

# Files' path.
import os.path

# Save and load files.
import csv
import pickle

# Data manipulations.
import numpy as np
import pandas as pd

# Import custom functions.
from shared_functions import interpretability_shap, obj_to_txt




            ### Initialization of Flask ###

app = Flask(__name__)




            ### Global files paths and names ###

IMPORTS_DIR_PATH = r'Exports/Preprocessed_data'
MODEL_DIR_PATH = r'Exports/Models/Selected'
SHAP_INTERPRETATIONS_DIR_PATH = r'Exports/Feature_interpretation/SHAP'

PKL_MODEL_FILE = 'selected_model.pkl'




            ### Global variables ###

# Load the optimized and trained model.
MODEL_PL = pickle.load(open(os.path.join(MODEL_DIR_PATH, PKL_MODEL_FILE), "rb"))

# Load the relevant datasets.
df_TRAIN = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_train.csv'))
df_TEST = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_new_customers.csv'))

# Set the customer IDs column's values as the dataframe indeces.
X_TRAIN = df_TRAIN.set_index('SK_ID_CURR')
X_TEST = df_TEST.set_index('SK_ID_CURR')




            ### Functions ###

# NB: Returning a dictionary seems to auto jisonify it.
#       => The use of jsonify(<object_to_return>) seems unnecessary. 

@app.route('/api/predictions/<int:customer_id>') #{customer_id}
def predictions (customer_id): #customer_id=100001.0

    """ Get and send the model prediction result corresponding to the customer's profile selected or
        the global explanations made over the dataset received as input. """
       
    # Get the arguments of the request received.
    #customer_id = request.args.get('customer_id')
    
    # Prediction for the selected customers.
    yhat = MODEL_PL.predict_proba(X_TEST.loc[[customer_id]])   
       
    # NB: It seems that numpy.array are not jsonable (TypeError: Object of type ndarray is not JSON serializable).
    #     => yhat is converted to a list to be jsonable and sent to the requester.
    return yhat.tolist()[0]

@app.route('/api/interpretations/<int:customer_id>/<int:cat_class>')
def shap_interpretations (customer_id, cat_class = 0):

    """ Get, serialize and send the SHAP explanations corresponding to the customer's profile selected."""

    # Get the model and the scaler separately from the pipeline.
    scaler = MODEL_PL['scaler']
    model = MODEL_PL['model']

    # Drop the target column in X_TRAIN.
    X_train = X_TRAIN.drop('TARGET', axis=1)

    # Shap explanations.
    if customer_id == 0: # Global.
    
        # Method 1 (longer): Generate SHAP explications at real time.
        #explanations, _ = interpretability_shap(model, scaler, X_train, X_TEST, cat_class)
        
        # Method 2 (Faster): Load an already generated global SHAP explication file.
        with open(os.path.join(SHAP_INTERPRETATIONS_DIR_PATH, 'global_shap_explanations.pkl'), "rb") as explanations_global_file:
            explanations = pickle.load(explanations_global_file)

    else: # Local.
        explanations, _ = interpretability_shap(model, scaler, X_train, X_TEST.loc[[customer_id]], cat_class)
       
    # Serialization of the shap explanations object as a string to allow its transfer across APIs.
    # NB: Step required because impossible to jsonify otherwise.
    explanations_serialized = obj_to_txt(explanations)
    
    print(type(explanations))
    
    return explanations_serialized #{'status': 'ok', 'explanations': explanations} #[explanations] #json.dumps(explanations, cls=to_json)


# NB: Display the root message at the end of the script when everything else happened well until here.
@app.route('/')
def api_running ():

    """ Message to display at the root of the flask server when it is running."""

    return 'The flask API server about model predictions and features interpretations is running...'




            ### Launch the flask API service (a server) ###

# Launch the flask API.
if __name__ == '__main__':
    app.run(debug=True)