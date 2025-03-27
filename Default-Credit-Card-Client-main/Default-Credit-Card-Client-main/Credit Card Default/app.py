import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
with open('random_forest_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('log_regression_model.pkl', 'rb') as log_reg_file:
    log_reg_model = pickle.load(log_reg_file)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form input
    features = [int(request.form['limit_balance']),
                int(request.form['sex']),
                int(request.form['education']),
                int(request.form['marriage']),
                int(request.form['age']),
                int(request.form['pay_0']),
                int(request.form['pay_2']),
                int(request.form['pay_3']),
                int(request.form['pay_4']),
                int(request.form['pay_5']),
                int(request.form['pay_6']),
                int(request.form['bill_amt1']),
                int(request.form['bill_amt2']),
                int(request.form['bill_amt3']),
                int(request.form['bill_amt4']),
                int(request.form['bill_amt5']),
                int(request.form['bill_amt6']),
                int(request.form['pay_amt1']),
                int(request.form['pay_amt2']),
                int(request.form['pay_amt3']),
                int(request.form['pay_amt4']),
                int(request.form['pay_amt5']),
                int(request.form['pay_amt6'])]
    
    # Convert features into DataFrame
    features_df = pd.DataFrame([features], columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 
                                                   'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 
                                                   'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                                                   'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    
    # Prediction using Random Forest Model
    rf_prediction = rf_model.predict(features_df)
    result = "Default" if rf_prediction[0] == 1 else "No Default"
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
