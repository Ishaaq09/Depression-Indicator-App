import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle

st.title("Mental Health Checker using AI")
st.divider()
st.write("Enter patient details to check depression risk.")

name = st.text_input("Name")
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 15, 100, 25)
city = st.selectbox("City", [
    'Ludhiana', 'Varanasi', 'Visakhapatnam', 'Mumbai', 'Kanpur',
    'Ahmedabad', 'Thane', 'Nashik', 'Bangalore', 'Patna', 'Rajkot',
    'Jaipur', 'Pune', 'Lucknow', 'Meerut', 'Agra', 'Surat',
    'Faridabad', 'Hyderabad', 'Srinagar', 'Ghaziabad', 'Kolkata',
    'Chennai', 'Kalyan', 'Nagpur', 'Vadodara', 'Vasai-Virar', 'Delhi',
    'Bhopal', 'Indore', 'Gurgaon'
])
status = st.selectbox("Are you a Working Professional or Student?", ["Working Professional", "Student"])
profession = st.selectbox("Profession", [
    'Chef', 'Teacher', 'Business Analyst', 'Software Engineer',
    'Data Scientist', 'Accountant', 'Designer', 'Pharmacist',
    'Architect', 'Consultant', 'Lawyer', 'Doctor'
])
work_study_hours = st.slider("Work/Study Hours", 0, 20, 8)
work_pressure_var = st.slider("Work Pressure (0-5)", 0, 5, 2)
work_pressure = float(work_pressure_var)
job_satisfaction_var = st.slider("Job Satisfaction (0-5)", 0, 5, 3)
job_satisfaction = float(job_satisfaction_var)
financial_stress_var = st.slider("Financial Stress (0-5)", 0, 5, 2)
financial_stress = float(financial_stress_var)
sleep_duration = st.selectbox("Sleep Duration", ['Less than 5 hours','5-6 hours','6-7 hours','7-8 hours','More than 8 hours'])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
degree = st.selectbox("Degree", [
    "B.Tech", "M.Tech", "MBA", "PhD", "B.Sc", "M.Sc", "B.Com", "M.Com",
    "MCA", "BBA", "MBBS", "LLB", "LLM", "M.Ed", "B.Ed", "M.Pharm", "B.Pharm"
])
suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])

with open("preprocessor_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# defining the mutlilayer perceptron architecture here
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# preprocessing the user input by the preprocessor_pipeline.pkl that we stored.
user_input = pd.DataFrame([{
    "Age": age,
    "Work/Study Hours": work_study_hours,
    "Work Pressure": work_pressure,
    "Job Satisfaction": job_satisfaction,
    "Financial Stress": financial_stress,
    "Gender": gender,
    "City": city,
    "Working Professional or Student": status,
    "Profession": profession,
    "Sleep Duration": sleep_duration,
    "Dietary Habits": dietary_habits,
    "Degree": degree,
    "Have you ever had suicidal thoughts ?": suicidal_thoughts,
    "Family History of Mental Illness": family_history
}])

X_processed = preprocessor.transform(user_input)
X_tensor = torch.tensor(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, dtype=torch.float32)

input_dim = X_tensor.shape[1]
model = MLP(input_dim)
model.load_state_dict(torch.load("depression_model.pth")) # loading the saved model parameters with which we trained the MLP
model.eval()

if st.button("Predict Depression Risk"):
    with torch.no_grad():
        y_pred = model(X_tensor)
        result = "High Risk of Depression" if y_pred.item() > 0.5 else "Low Risk of Depression"
    st.subheader(f"Prediction: {result}")
