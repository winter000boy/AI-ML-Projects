from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the pickle file
model_path = os.path.join(current_dir, 'cancer.pkl')

# Load the model
model = pickle.load(open(model_path, 'rb'))  # Load the model in rb = "Read Binary mode"

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    features = features.split(',')
    np_features = np.asarray(features, dtype=np.float32)

    # prediction
    pred = model.predict(np_features.reshape(1, -1))
    message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
    return render_template('Index.html', message=message)

# Python main
if __name__ == '__main__':
    app.run(debug=True)