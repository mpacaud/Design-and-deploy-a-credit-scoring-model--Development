            ### Importation of libraries ###

# Import streamlit.
import streamlit as st

import requests
import json

# Files' path.
import os.path

# Save and load files.
import csv
import pickle

# Data manipulations.
import numpy as np
import pandas as pd

# SHAP.
import shap
#import streamlit.components.v1 as components
from streamlit_shap import st_shap # NB: SHAP wrapper in order to replace shap.initjs() which was not working because of javascript library not loading.
#shap.initjs()
#shap.getjs()


import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go


# Import custom functions.
from shared_functions import txt_to_obj


# Global files paths and names.
IMPORT_DATA_DIR_PATH = r'Exports\Preprocessed_data'
IMPORT_MODELS_SUM_DIR_PATH = r'Exports\Models\Tried'
API_SERVER_PATH = 'http://127.0.0.1:5000/'
PKL_MODELS_SUM_FILE = 'models_info.pkl'


# Tell the program the "app" should be considered as a flask app.
#app = Flask(__name__)


            ### Functions ###

@st.cache_data
def load_data ():

    df = pd.read_csv(os.path.join(IMPORT_DATA_DIR_PATH, 'preprocessed_data_new_customers.csv'))
        
    return df

def get_prediction (customer_id):

    # Set the url and the parameters to get the predictions. 
    url_api_predictions = API_SERVER_PATH + 'api/predictions/%i' % int(customer_id)
    #var_dict = {'customer_id': selected_customer_id}

    # Make the request and get the corresponding response in a json format.
    response_pred = requests.get(url_api_predictions)

    # Extract the result of the prediction and convert it back from list to a np.array format.
    yhat_customer = np.array(response_pred.json()['yhat'][0])
    
    return yhat_customer
   
def get_shap_explanations (customer_id, cat_class = 0):

    # Set the url and the required parameters to get the shap explanations. 
    url_api_shap_expl = API_SERVER_PATH + 'api/interpretations/%i/%i' % (int(customer_id), int(cat_class))

    # Make the request and get the corresponding response in a json format.
    response_expl = requests.get(url_api_shap_expl)

    # Extract the results.
    shap_expl = txt_to_obj(response_expl.text)
    
    return shap_expl


#def st_shap(shap_plot, height=10):
#    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
#    components.html(shap_html, height=height)






            ### Data loading ###

# Load customers' data and store it in cache for quicker use.
df = load_data()


            ### Dashboard building ###

# Set grid.
col1, col2, _ = st.columns([1,3,1])

CAT_CLASS = 0

# Create the a selection button between customers.
with col1:
    selected_customer_id = st.selectbox("Customer ID selection", df['SK_ID_CURR'])
    

            ### Get customer information ###

# Get the prediction recommendation for the selected customer.
yhat_customer = get_prediction(selected_customer_id)

# Get the SHAP explanations of the prediction for the selected customer.
shap_expl = get_shap_explanations(selected_customer_id, cat_class=CAT_CLASS)

df_models_sum = pd.read_pickle(os.path.join(IMPORT_MODELS_SUM_DIR_PATH, PKL_MODELS_SUM_FILE))
proba_thr = df_models_sum.iloc[-1, 3]

with col2:   
    if yhat_customer[CAT_CLASS] > proba_thr:
        st.markdown("<h2 style='text-align: center; color: lightgreen'> Application accepted </h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: salmon'> Application denied </h2>", unsafe_allow_html=True)
    



# Set grid.
col1, col2, col3 = st.columns([1,3,1])
    
with col2:
    fig = go.Figure(go.Indicator(mode = "gauge+number+delta",
                                 value = yhat_customer[CAT_CLASS],
                                 domain = {'x': [0, 1], 'y': [0, 1]},
                                 title = {'text': "Customer's repay probability"},
                                 delta = {'reference': proba_thr, 'increasing': {'color': "lightgreen"}, 'decreasing': {'color': "salmon"}},
                                 gauge = {'axis': {'range': [0, 1]},
                                          'bar': {'color': "royalblue"},
                                          'borderwidth': 2,
                                          'bordercolor': "black",
                                          'steps': [{'range': [0, proba_thr], 'color': "salmon"}, {'range': [proba_thr, 1], 'color': "lightgreen"}],
                                          'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 1, 'value': shap_expl.base_values[0]}},
                                ))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig, use_container_width=False)

with col3:
#    m = st.markdown("""<style>div.stButton > button:first-child {background-color: salmon;}</style>""", unsafe_allow_html=True)
#    b = st.button("Trustless range")

    # Help button for the gauge figure.
    description = "1. Show the recommendation of the model.\
                   2. The gauge quickly highlight the predicted probability that the customers repays its credit in blue.\
                      Red: Trustless range.\
                      Green: Trustful range.\
                      White bar: Average trust among all customers.\
                      Colored triangle: How far the customer's repay probability is from the probability threshold."
    st.button("?", help=description, use_container_width=False)
    



# Set grid.
_, col2, _ = st.columns([1, 3, 1])

# Title of the section.
with col2:

    description = "This figure quickly highlights the main features which influenced the most the model prediction.\
                   NB: The 2nd figure down below show the same information at a more detailed format."
    st.markdown("<h4 style='text-align: center; color: black;'> Force plot in line </h4>", unsafe_allow_html=True, help=description)

# Draw the interactive SHAP line force plot.
st_shap(shap.plots.force(shap_expl[0]))


# NB: st_shap has not been used in this section because it was impossible to manage
#     the size and shape winthin the dashboard of those figures drawn this way.
description = "Select the number of features to show in both figures down below."
top_ft = st.slider("Number of features to show", min_value=1, max_value=len(shap_expl[0].values), value=10, step=1,
                   help=description, disabled=False, label_visibility="visible")



col1, col2 = st.columns(2, gap="small")

# Draw the absolute SHAP values as probabilities.
fig, ax = plt.subplots() # Required to set the shap.plots as a matplotlib figure (It won't display properly otherwise with st.pyplot).
with col1:

    # Title of the section.
    description = "This graphic shows the importance of each feature toward the model prediction in term of absolute values."
    st.markdown("<h4 style='text-align: center; color: black;'> Absolute SHAP values </h4>", unsafe_allow_html=True, help=description)
    
    # Turn the SHAP graphical object to a displayable matplotlib object.
    ax = shap.plots.bar(shap_expl, max_display=top_ft+1, show=False)
    
    # Draw the figure.
    st.pyplot(fig)

# Draw the detailed force plot of SHAP values as probabilities.
fig, ax = plt.subplots() # Reinitialize the figure environment in order to avoid any superimpose with the previous graphic.
with col2:

    # Title of the section.
    description = "This graphic shows in which measure each feature contributed to lead to the model prediction result\
                   from the average customers' probability to see their application accepted \
                   (NB: The red line shows the limit above which the model will recommend the acceptation of the customer's application)."
    st.markdown("<h4 style='text-align: center; color: black;'> Detailed force plot </h4>", unsafe_allow_html=True, help=description)
    
    # Add the probability threshold on the figure which will decide if the customer's application should be accepted or denied.
    ax.axvline(x=proba_thr, color='r', linestyle='--') # Add 
    
    # Turn the SHAP graphical object to a displayable matplotlib object.
    ax = shap.plots.waterfall(shap_expl[0], max_display=top_ft+1, show=False)
    
    # Draw the final figure
    st.pyplot(fig)



# Title of the section.
_, col2, _ = st.columns([1,3,1])
with col2:
    description = "This table shows all customer's data used for the model prediction."
    st.markdown("<h4 style='text-align: center; color: black;'> Customer's data </h4>", unsafe_allow_html=True, help=description)

    # Display the raw data corresponding to the selected customer.
    st.dataframe(df[df['SK_ID_CURR'] == selected_customer_id].set_index('SK_ID_CURR').T, use_container_width=True)

