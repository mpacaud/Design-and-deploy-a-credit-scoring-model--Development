from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api/predictions/')
def predictions (customer_id):

    # Load the dataset 
    

    # Load the optimized and trained model.
    model_pl = pickle.load(open(os.path.join(SELECTED_MODEL_DIR_PATH, PKL_MODEL_FILE), "rb"))
    
    y_pred = model_fit_predict(model, X, y, cv):

    return 'This is my first API call!'



@app.route('/post', methods=["POST"])
def testpost():
     input_json = request.get_json(force=True) 
     dictToReturn = {'text':input_json['text']}
     return jsonify(dictToReturn)