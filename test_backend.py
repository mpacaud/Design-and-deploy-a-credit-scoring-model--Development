# Import base modules for testing.
import unittest
import json

# Import the application and its functions.
import app as my_app
from app import predictions as pred_fct
from app import shap_interpretations as shap_fct

# Import additional required modules.
import shap
from shared_functions import txt_to_obj

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# Create the testing class.
class unittests_backend (unittest.TestCase):
    
            ### Create a client to simulate the requester ###
    
    def setUp (self):
    
        """ Create a client to simulate requests. """
        
        my_app.app.testing = True
        self.app = my_app.app.test_client()
        


            ### Test the home page of the API ###

    def test_home_response_code (self):
    
        """ Check responsivness. """
    
        response = self.app.get('/')
        assert response.status_code == 200
        
    def test_home_message (self):
    
        """ Check response. """
    
        response = self.app.get('/')
        self.assertEqual(response.text, 'The flask API server about model predictions and features interpretations is running...')
           


            ### Test the prediction function ###

    def test_pred_fct_response_code (self):
    
        """ Check responsivness. """
    
        value = '/api/predictions/100001'
        response = self.app.get(value)
        assert response.status_code == 200
    
    def test_pred_fct_returned_object_type (self):
    
        """ Check response object type. """
    
        value = '/api/predictions/100001'
        response = self.app.get(value)
        assert type(response.json) == type(list())
        
    def test_pred_fct_returned_value (self):
    
        """ Check response object value. """
    
        value = '/api/predictions/100001'
        response = self.app.get(value)
        yhat_customer = response.json[0]
        assert yhat_customer > 0
        assert yhat_customer < 1



            ### Test the interpretation function ###

    def test_interpret_fct_response_code (self):
    
        """ Check responsivness. """
        
        #Arrange.
        value = '/api/interpretations/100001/0'
        
        # Act.
        response = self.app.get(value)
        
        # Assert.
        assert response.status_code == 200

    def test_interpret_fct_returned_object_type (self):
    
        """ Check response serialized object type. """
        
        value = '/api/interpretations/100001/0'
        response = self.app.get(value)
        assert type(response.text) == type(str())
        
    def test_pred_fct_returned_value (self):
    
        """ Check response object type. """
    
        value = '/api/interpretations/100001/0'
        response = self.app.get(value)
        shap_expl = txt_to_obj(response.text)
        assert str(type(shap_expl)) == "<class 'shap._explanation.Explanation'>"
    

# Run the tests.
if __name__ == '__main__':
    unittest.main()
