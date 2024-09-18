import sys
import os
import pickle
import joblib
import pandas as pd
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi import FastAPI, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from models import LoanRequest

templates = Jinja2Templates(directory="templates")

# Instantiate PredictPipeline 
pipeline = PredictPipeline()

app = FastAPI()

# Load the model
try:
    model = joblib.load("artifacts/models/xgboost_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the categorical columns and their encoders
categorical_columns = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial',
    'Neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
    'credit_type', 'co_applicant_credit_type', 'age',
    'submission_of_application', 'Region', 'Security_Type']





def generate_html(input_data, prediction):
    input_items = list(input_data.items())
    num_columns = 4  # Number of columns in the table
    num_rows = (len(input_items) + num_columns - 1) // num_columns  # Calculate the required number of rows

    results_html = """
    <html>
        <head></head>
        <body style="background-color: #f5f5f5;"> <!-- Light grey background for the whole page -->
            <h1 style='text-align: center;'>Client Data</h1>
            <table style="border-collapse: collapse; width: 75%; margin: 0 auto;"> <!-- Light blue/green background color for table header -->
                <tr style="background-color: #e0f7fa;">
    """
    
    # Create table headers
    for col in range(num_columns):
        results_html += "<th style='border: 1px solid black; padding: 8px;'>Field</th><th style='border: 1px solid black; padding: 8px;'>Value</th>"
    results_html += "</tr>"
    
    # Fill the table with input data
    for row in range(num_rows):
        results_html += "<tr>"
        for col in range(num_columns):
            index = row * num_columns + col
            if index < len(input_items):
                field, value = input_items[index]
                results_html += f"<td style='border: 1px solid black; padding: 8px; background-color: #e0f7fa;'>{field}</td><td style='border: 1px solid black; padding: 8px; background-color: #e0f7fa;'>{value}</td>"
            else:
                results_html += "<td style='border: 1px solid black; padding: 8px;'></td><td style='border: 1px solid black; padding: 8px;'></td>"  # Empty cells for remaining spaces
        results_html += "</tr>"

    # Determine the prediction statement based on the prediction value
    if prediction == 1:
        prediction_heading = "<h2 style='text-align: center;'>Prediction Result</h2>"
        prediction_statement = "<div style='border: 1px solid black; background-color: #d4edda; padding: 10px; margin: 20px auto; width: 65%; text-align: center;'>Client qualifies for the loan.</div>"
    else:
        prediction_heading = "<h2 style='text-align: center;'>Prediction Result</h2>"
        prediction_statement = "<div style='border: 1px solid black; background-color: #f8d7da; padding: 10px; margin: 20px auto; width: 65%; text-align: center;'>Unfortunately, the client does not qualify for the loan.</div>"
    
    # Add the prediction statement and form link
    form_link = "<p style='text-align: center;'><a href=\"/\" style='color: blue;'>Click here to go back to the client form</a></p>"
    # Add the prediction statement
    results_html += f"""
            </table>
            {prediction_heading}
            {prediction_statement}
            {form_link}
        </body>
    </html>
    """
    
    return results_html

@app.get("/", response_class=HTMLResponse)
def show_form():
    return """
    <html>
        <head>
            <style>
                body {
                    background-color: lightgrey;
                    font-family: Arial, sans-serif;
                }
                .form-section {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 10px;
                    max-width: 80%;
                    margin: 20px auto;
                    background-color: lightgrey;
                    padding: 20px;
                    border-radius: 10px;
                }
                .form-column {
                    background-color: lightblue;
                    padding: 10px;
                    border-radius: 5px;
                    box-sizing: border-box;
                }
                .form-column input, .form-column select {
                    width: 100%;
                    margin-bottom: 10px;
                    padding: 5px;
                    box-sizing: border-box;
                }
                .submit-button {
                    background-color: orange;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    display: block;
                    margin: 20px auto;
                }
                h2 {
                    text-align: center;
                }
                .form-column:last-child {
                    grid-row: span 2;
                }
            </style>
        </head>
        <body>
            <h2 style="color: navy; text-decoration: underline;">Welcome! Please fill in the form below:</h2>
            <form action="/predict" method="post">
                <div class="form-section">
                    <div class="form-column">
                        <label for="loan_limit">Loan Limit:</label>
                        <select name="loan_limit">
                            <option value="cf">cf</option>
                            <option value="ncf">ncf</option>
                        </select>
                        
                        <label for="gender">Gender:</label>
                        <select name="gender">
                            <option value="Sex Not Available">Sex Not Available</option>
                            <option value="Male">Male</option>
                            <option value="Joint">Joint</option>
                            <option value="Female">Female</option>
                        </select>
                        
                        <label for="approv_in_adv">Approval in Advance:</label>
                        <select name="approv_in_adv">
                            <option value="nopre">nopre</option>
                            <option value="pre">pre</option>
                        </select>
                        
                        <label for="loan_type">Loan Type:</label>
                        <select name="loan_type">
                            <option value="type1">type1</option>
                            <option value="type2">type2</option>
                            <option value="type3">type3</option>
                        </select>
                        
                        <label for="loan_purpose">Loan Purpose:</label>
                        <select name="loan_purpose">
                            <option value="p1">p1</option>
                            <option value="p4">p4</option>
                            <option value="p3">p3</option>
                            <option value="p2">p2</option>
                        </select>
                        
                        <label for="credit_worthiness">Credit Worthiness:</label>
                        <select name="credit_worthiness">
                            <option value="l1">l1</option>
                            <option value="l2">l2</option>
                        </select>
                        
                        <label for="open_credit">Open Credit:</label>
                        <select name="open_credit">
                            <option value="nopc">nopc</option>
                            <option value="opc">opc</option>
                        </select>
                        
                        <label for="business_or_commercial">Business or Commercial:</label>
                        <select name="business_or_commercial">
                            <option value="nob/c">nob/c</option>
                            <option value="b/c">b/c</option>
                        </select>
                    </div>

                    <div class="form-column">
                        <label for="loan_amount">Loan Amount:</label>
                        <input type="number" name="loan_amount" step="0.01">
                        
                        <label for="rate_of_interest">Rate of Interest:</label>
                        <input type="number" name="rate_of_interest" step="0.01">
                        
                        <label for="interest_rate_spread">Interest Rate Spread:</label>
                        <input type="number" name="interest_rate_spread" step="0.01">
                        
                        <label for="upfront_charges">Upfront Charges:</label>
                        <input type="number" name="upfront_charges" step="0.01">
                        
                        <label for="term">Term:</label>
                        <input type="number" name="term" step="0.01">
                        
                        <label for="neg_ammortization">Negative Amortization:</label>
                        <select name="neg_ammortization">
                            <option value="not_neg">not_neg</option>
                            <option value="neg_amm">neg_amm</option>
                        </select>
                        
                        <label for="interest_only">Interest Only:</label>
                        <select name="interest_only">
                            <option value="not_int">not_int</option>
                            <option value="int_only">int_only</option>
                        </select>
                        
                        <label for="lump_sum_payment">Lump Sum Payment:</label>
                        <select name="lump_sum_payment">
                            <option value="not_lpsm">not_lpsm</option>
                            <option value="lpsm">lpsm</option>
                        </select>
                    </div>

                    <div class="form-column">
                        <label for="property_value">Property Value:</label>
                        <input type="number" name="property_value" step="0.01">
                        
                        <label for="construction_type">Construction Type:</label>
                        <select name="construction_type">
                            <option value="sb">sb</option>
                            <option value="mh">mh</option>
                        </select>
                        
                        <label for="occupancy_type">Occupancy Type:</label>
                        <select name="occupancy_type">
                            <option value="pr">pr</option>
                            <option value="sr">sr</option>
                            <option value="ir">ir</option>
                        </select>
                        
                        <label for="secured_by">Secured By:</label>
                        <select name="secured_by">
                            <option value="home">home</option>
                            <option value="land">land</option>
                        </select>
                        
                        <label for="total_units">Total Units:</label>
                        <select name="total_units">
                            <option value="1U">1U</option>
                            <option value="2U">2U</option>
                            <option value="3U">3U</option>
                            <option value="4U">4U</option>
                        </select>
                        
                        <label for="income">Income:</label>
                        <input type="number" name="income" step="0.01">
                        
                        <label for="credit_type">Credit Type:</label>
                        <select name="credit_type">
                            <option value="EXP">EXP</option>
                            <option value="EQUI">EQUI</option>
                            <option value="CRIF">CRIF</option>
                            <option value="CIB">CIB</option>
                        </select>

                        <label for="credit_score">Credit Score:</label>
                        <input type="number" name="credit_score" step="1">

                    </div>

                    <div class="form-column">
                        
                        <label for="co_applicant_credit_type">Co-applicant Credit Type:</label>
                        <select name="co_applicant_credit_type">
                            <option value="CIB">CIB</option>
                            <option value="EXP">EXP</option>
                        </select>
                        
                        <label for="age">Age:</label>
                        <select name="age">
                            <option value="25-34">25-34</option>
                            <option value="55-64">55-64</option>
                            <option value="35-44">35-44</option>
                            <option value="45-54">45-54</option>
                            <option value="65+">65+</option>
                            <option value="18-24">18-24</option>
                        </select>
                        
                        <label for="submission_of_application">Submission of Application:</label>
                        <select name="submission_of_application">
                            <option value="to_inst">nsub</option>
                            <option value="not_inst">sub</option>
                        </select>
                        
                        <label for="ltv">LTV:</label>
                        <input type="number" name="ltv" step="0.01">
                        
                        <label for="region">Region:</label>
                        <select name="region">
                            <option value="south">south</option>
                            <option value="North">North</option>
                            <option value="central">central</option>
                            <option value="North-East">North-East</option>
                        </select>
                        
                        <label for="security_type">Security Type:</label>
                        <select name="security_type">
                           <option value="direct">direct</option>
                            <option value="indirect">indirect</option>
                        </select>
                        
                        <label for="dtir1">DTIR1:</label>
                        <input type="number" name="dtir1" step="0.01">
                    </div>
                </div>
                <button type="submit" class="submit-button">Submit</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(
    loan_limit: str = Form(...),
    gender: str = Form(...),
    approv_in_adv: str = Form(...),
    loan_type: str = Form(...),
    loan_purpose: str = Form(...),
    credit_worthiness: str = Form(...),  # Note the capital 'W'
    open_credit: str = Form(...),
    business_or_commercial: str = Form(...),
    loan_amount: int = Form(...),  # int for loan_amount
    rate_of_interest: float = Form(...),  # float for rate_of_interest
    interest_rate_spread: float = Form(...),  # float for interest_rate_spread
    upfront_charges: float = Form(...),  # float for upfront_charges
    term: int = Form(...),  # int for term
    neg_ammortization: str = Form(...),
    interest_only: str = Form(...),
    lump_sum_payment: str = Form(...),
    property_value: float = Form(...),  # float for property_value
    construction_type: str = Form(...),
    occupancy_type: str = Form(...),
    secured_by: str = Form(...),
    total_units: str = Form(...),  # str for total_units
    income: float = Form(...),  # float for income
    credit_type: str = Form(...),
    credit_score: int = Form(...),  # int for credit_Score
    co_applicant_credit_type: str = Form(...),
    age: str = Form(...),  # str for age
    submission_of_application: str = Form(...),
    ltv: float = Form(...),  # float for ltv
    region: str = Form(...),
    security_type: str = Form(...),
    dtir1: int = Form(...)  # int for dtir1
):
    
    try:
        # Prepare input data
        # Prepare input data without using CustomData
        input_data = {
        'loan_limit': loan_limit, 
        'gender': gender, 
        'approv_in_adv': approv_in_adv,
        'loan_type': loan_type, 
        'loan_purpose': loan_purpose, 
        'credit_worthiness': credit_worthiness,  # Notice the capital 'W'
        'open_credit': open_credit, 
        'business_or_commercial': business_or_commercial,
        'loan_amount': loan_amount,  # int
        'rate_of_interest': rate_of_interest,  # float
        'interest_rate_spread': interest_rate_spread,  # float
        'upfront_charges': upfront_charges,  # float
        'term': term,  # int
        'neg_ammortization': neg_ammortization, 
        'interest_only': interest_only,
        'lump_sum_payment': lump_sum_payment, 
        'property_value': property_value,  # float
        'construction_type': construction_type,
        'occupancy_type': occupancy_type, 
        'secured_by': secured_by, 
        'total_units': total_units,  # str (Not int as before)
        'income': income,  # float
        'credit_type': credit_type, 
        'credit_score': credit_score,  # int
        'co_applicant_credit_type': co_applicant_credit_type,
        'age': age,  # str (Not int as before)
        'submission_of_application': submission_of_application, 
        'ltv': ltv,  # float
        'region': region,
        'security_type': security_type, 
        'dtir1': dtir1
        }


        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        print("Before Prediction and input data converted to df")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        predictions=predict_pipeline.predict(df)
        print("after Prediction")
        prediction = predictions[0]
        
    
        # Generate HTML response

        if prediction == 1:
            prediction_statement = "<p>Client qualifies for the loan.</p>"
        else:
            prediction_statement = "<p>Unfortunately, the client does not qualify for the loan.</p>"
        
        html_response = generate_html(input_data, prediction_statement)
        return HTMLResponse(content=html_response)
    
    


    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        error_message = f"An error occurred: {''.join(tb_str)}"
        print(error_message)
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
