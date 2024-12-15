import streamlit as st
import pandas as pd
import pickle
import math
import random

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def main():
    dummy1, title_col, dummy2 = st.columns([2, 3, 2])
    with title_col:
        st.header("LOAN ELIGIBILITY PREDICTION")
        st.caption("By: Daniel Karori")

    st.subheader("Model Prediction")
    st.write("Enter the form below to check loan eligibility ")

    # Loading the trained model and scaler
    with open("models/logistic_regression.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("models/standard_scaler.pkl", "rb") as scaler_file:
        scale = pickle.load(scaler_file)

    # scaler_mean = getattr(scale, "mean_", None)
    # scaler_scale = getattr(scale, "scale_", None)
    #
    # scaler_mean, scaler_scale

        # form to collect input from the user
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.number_input("Number of Dependents", min_value=0, step=1)
            if dependents > 2:
                dependents = 3
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income (Kshs)", min_value=0)
            if applicant_income:
                applicant_income = math.sqrt(applicant_income)
        with col2:
            coapplicant_income = st.number_input("Coapplicant Income (Kshs)", min_value=0)
            if coapplicant_income:
                coapplicant_income = math.sqrt(coapplicant_income)
            loan_amount = st.number_input("Loan Amount (Kshs)", min_value=0)
            if loan_amount:
                loan_amount = math.sqrt(loan_amount)
            loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
            credit_history = st.selectbox("Credit History", [1, 0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit_button = st.form_submit_button("Submit")

    # Processing
    if submit_button:
        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 0 if education == "Graduate" else 1
        self_employed = 1 if self_employed == "Yes" else 0
        property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

        applicant_income = (applicant_income * 12) / 132
        coapplicant_income = (coapplicant_income * 12) / 132
        loan_amount = (loan_amount * 12) / 132

        data = {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area],
        }

        testing_df = pd.DataFrame(data)

        # scaler = RobustScaler()
        # scaled_data = scaler.fit_transform(testing_df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]])
        # st.write("Scaled Data (ApplicantIncome, CoapplicantIncome, LoanAmount):")
        # st.write(scaled_data)

        scaled_data = scale.fit_transform(testing_df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]])

        # st.write("Scaled Data (ApplicantIncome, CoapplicantIncome, LoanAmount):")
        # st.write(scaled_data)

        # expected_features = scale.feature_names_in_
        # st.write("Expected Features for Scaling:", expected_features)


        if len(testing_df.shape) == 1:
            testing_df = testing_df.reshape(1, -1)


        # # Prediction
        prediction = model.predict_proba(testing_df)

        prediction_class = model.predict(testing_df)

        # Determine prediction label and accuracy
        labels = {1: "Approved", 0: "Not Approved"}
        predicted_label = labels[prediction_class[0]]
        prediction_accuracy = round(prediction[0][prediction_class[0]] * 100, 2)

        # Display Results
        st.subheader("Prediction Result")
        st.write(f"Loan Status: **{predicted_label}**")
        st.write(f"Prediction Accuracy: **{prediction_accuracy}%**")

        # st.write(prediction*100)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Karori Project",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()



