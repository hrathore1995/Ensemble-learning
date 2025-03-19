import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved models and preprocessors
best_model_rf = joblib.load("C:/Users/harsh/ensemble-learning-project/models/random_forest_model.pkl")
best_gb_model = joblib.load("C:/Users/harsh/ensemble-learning-project/models/gradient_boosting_model.pkl")
best_xgb_model = joblib.load("C:/Users/harsh/ensemble-learning-project/models/xgboost_model.pkl")
scaler = joblib.load("C:/Users/harsh/ensemble-learning-project/models/scaler.pkl")  # StandardScaler
le = joblib.load("C:/Users/harsh/ensemble-learning-project/models/label_encoder.pkl")  # LabelEncoder
dummies = joblib.load("C:/Users/harsh/ensemble-learning-project/models/one_hot_columns.pkl")  # Dummies for 'Education_Level'

# Streamlit UI
st.title("Driver Attrition Prediction")
st.write("This app predicts whether a driver will leave Ola or not based on various factors.")

# Input form
age = st.slider("Age", min_value=18, max_value=100)
income = st.number_input("Income", min_value=0)
total_business_value = st.number_input("Total Business Value", min_value=0)
quarterly_rating = st.slider("Quarterly Rating", min_value=0, max_value=5)
education_level_1 = st.selectbox("Education Level_1", [0,1])
education_level_2 = st.selectbox("Education Level_2", [0,1])
city = st.selectbox("City", ['C23', 'C7', 'C13', 'C9', 'C11', 'C2', 'C19', 'C26', 'C20', 'C17',
                            'C29', 'C10', 'C24', 'C14', 'C6', 'C28', 'C5', 'C18', 'C27', 'C15',
                            'C8', 'C25', 'C21', 'C1', 'C4', 'C3', 'C16', 'C22', 'C12'])
joining_month = st.selectbox("Joining Month", list(range(1, 13)))
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2020)
quaterly_rating_raise = st.slider("Quarterly Rating Raise", min_value=0, max_value=5)
income_raised = st.number_input("Income Raised", min_value=0)
no_of_reportings = st.number_input("Number of Reportings", min_value=0)
gender = st.selectbox("Gender", [0,1])  # Add 'Male' and 'Female' options for gender
grade = st.number_input("Grade", min_value=0)

# Button to trigger prediction
if st.button("Predict"):
    # Data Preprocessing
    data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Total Business Value': [total_business_value],
        'Quarterly Rating': [quarterly_rating],
        'Education_Level_1': [education_level_1],
        'Education_Level_2': [education_level_2],
        'City': [city],
        'joining_month': [joining_month],
        'joining_year': [joining_year],
        'quaterly_rating_raise': [quaterly_rating_raise],
        'Income_raised': [income_raised],
        'no_of_reportings': [no_of_reportings],
        'Gender': [gender],
        'Grade': [grade]
    })

    # Encode categorical features (only for 'Gender' and 'Education_Level' if needed)
    # 'Gender' is label encoded but 'Education_Level_1' and 'Education_Level_2' are already one-hot encoded during training, so skip label encoding for those
    data['Gender'] = le.transform(data['Gender'])

    # One-hot encode 'City' only, as 'Education_Level' is already one-hot encoded
    data = pd.get_dummies(data, columns=['City'], drop_first=True)

    # Standardize numerical columns
    data[['Age', 'Income', 'Total Business Value', 'Quarterly Rating']] = scaler.transform(data[['Age', 'Income', 'Total Business Value', 'Quarterly Rating']])

    # Ensure the input data has all the columns as in the training set
    expected_columns = ['no_of_reportings', 'Age', 'Gender', 'Grade', 'Total Business Value',
                        'Income', 'Joining Designation', 'Quarterly Rating', 'joining_month',
                        'joining_year', 'quaterly_rating_raise', 'Income_raised',
                        'Education_Level_1', 'Education_Level_2', 'City_C10', 'City_C11',
                        'City_C12', 'City_C13', 'City_C14', 'City_C15', 'City_C16', 'City_C17',
                        'City_C18', 'City_C19', 'City_C2', 'City_C20', 'City_C21', 'City_C22',
                        'City_C23', 'City_C24', 'City_C25', 'City_C26', 'City_C27', 'City_C28',
                        'City_C29', 'City_C3', 'City_C4', 'City_C5', 'City_C6', 'City_C7',
                        'City_C8', 'City_C9']

    # Add missing columns with 0s for those that might not be in the input (like missing City dummies)
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns to match the expected order of the model
    data = data[expected_columns]

    # Make predictions using the models
    rf_pred = best_model_rf.predict(data)
    gb_pred = best_gb_model.predict(data)
    xgb_pred = best_xgb_model.predict(data)

    # Use voting: Majority Voting
    final_prediction = np.round((rf_pred + gb_pred + xgb_pred) / 3)

    # Display prediction result
    if final_prediction[0] == 0:
        st.write("The driver has **churned** (left the company).")
    else:
        st.write("The driver has **not churned** (still with the company).")
