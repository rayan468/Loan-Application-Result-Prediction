import streamlit as st
import joblib 
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
le=LabelEncoder()
df.drop("Loan_ID",axis=1,inplace=True)
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Gender']= le.fit_transform(df['Gender'])
df['Married']= le.fit_transform(df['Married'])
df['Education']= le.fit_transform(df['Education'])
df['Self_Employed']= le.fit_transform(df['Self_Employed'])
df['Property_Area']= le.fit_transform(df['Property_Area'])
df['Loan_Status']= le.fit_transform(df['Loan_Status'])
df['Dependents']= le.fit_transform(df['Dependents'])
x=df.drop("Loan_Status",axis=1)
y=df["Loan_Status"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
lr=LogisticRegression(max_iter=10000)
lr.fit(xtrain,ytrain)
st.title("Loan Approval Prediction")
st.header("Enter the details to predict loan approval status") 
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self_Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit_History", ["0", "1"])
property_area = st.selectbox("Property_Area", ["Urban", "Semiurban", "Rural"])
applicant_income = st.number_input("ApplicantIncome", min_value=0)
coapplicant_income = st.number_input("CoapplicantIncome", min_value=0)
loan_amount = st.number_input("LoanAmount", min_value=0)
loan_amount_term = st.number_input("Loan_Amount_Term", min_value=0)
if gender == "Male":
    gender_new = 1
else:
    gender_new = 0
if married == "Yes":
    married_new = 1  
else:
    married_new = 0
if dependents == "3+":
    dependents_new = 3
else:
    dependents_new = int(dependents)
if education == "Graduate":
    education_new = 1
else:    education_new = 0
if self_employed == "Yes":
    self_employed_new = 1
else:    self_employed_new = 0
credit_history_new = int(credit_history)
if property_area == "Urban":
    property_area_new = 2
elif property_area == "Semiurban":
    property_area_new = 1
else:    property_area_new = 0
input_data = [[gender_new, married_new, dependents_new, education_new, self_employed_new, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history_new, property_area_new]]
if st.button("Predict Loan Status"):
    prediction = lr.predict(input_data)
    if prediction[0] == 1:
        st.write("Loan Approved")
    else:
        st.write("Loan Rejected")
