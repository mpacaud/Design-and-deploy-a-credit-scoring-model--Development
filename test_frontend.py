# Import base modules for testing.
import unittest
import json

# Import the application and its functions.
#import dashboard_streamlit as my_app
from dashboard_streamlit import load_data as load_data_fct
from dashboard_streamlit import get_prediction as get_pred_fct
from dashboard_streamlit import get_shap_explanations as get_shap_expls

# Import additional required modules.
import shap
from shared_functions import txt_to_obj

# Import global paths.
IMPORT_DATA_DIR_PATH = r'Exports/Preprocessed_data'
#API_SERVER_PATH = json.load(open('urls.json', 'r'))['on_lan']['backend_url'] # NB: on_lan / on_line


#import warnings
#warnings.filterwarnings('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)
   
       
            ### Test the function loading data in cache ###
            
def test_load_data_fct_returned_object_type (self):
    
    """ Load data and check if it has the expected shape. """
        
    # Arrange.
    value = os.path.join(IMPORT_DATA_DIR_PATH, 'preprocessed_data_new_customers.csv')
        
    # Act.
    df = pd.read_csv(value)
        
    # Assert.
    assert df.shape == (48744, 366)
    



            ### Test the request function for predictions ###

def test_get_pred_fct_returned_object_type (self):
    
    """ Check response object type. """
        
    value = 100001
    result = get_prediction(value)
    assert type(float) == type(value)
        
        
def test_get_pred_fct_returned_object_type (self):
    
    """ Check response object value. """
    
    value = 100001
    result = get_prediction(value)
    yhat_customer = result[0]
    assert yhat_customer > 0
    assert yhat_customer < 1
        
        
        
        
            ### Test the request function for model interpretations ###
    
#    def get_shap_expls (self):
#    
#        """ Check response object type. """
#    
#        val1 = 100001
#        val2 = 0    
#        result = get_shap_explanations(val1, val2)
#        assert str(type(shap_expl)) == "<class 'shap._explanation.Explanation'>"
#        
