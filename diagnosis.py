import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import streamlit as st

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_column):
    print("Loading file from:", file_path)
    data = pd.read_csv(file_path)
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X = pd.get_dummies(X, drop_first=True)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, y, scaler, X.columns

# Load and preprocess datasets
parkinsons_X, parkinsons_y, parkinsons_scaler, parkinsons_columns = load_and_preprocess_data(
    'c:/Users/rahul kumar/Downloads/Python_Project/Python_Project/uploads/parkinsons_data.csv', 'status')
heart_X, heart_y, heart_scaler, heart_columns = load_and_preprocess_data(
    'c:/Users/rahul kumar/Downloads/Python_Project/Python_Project/uploads/heart_data.csv', 'target')
cancer_X, cancer_y, cancer_scaler, cancer_columns = load_and_preprocess_data(
    'c:/Users/rahul kumar/Downloads/Python_Project/Python_Project/uploads/cancer_data.csv', 'diagnosis')

# Train models
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return model, accuracy, precision, recall, f1

parkinsons_model, parkinsons_accuracy, parkinsons_precision, parkinsons_recall, parkinsons_f1 = train_model(parkinsons_X, parkinsons_y)
heart_model, heart_accuracy, heart_precision, heart_recall, heart_f1 = train_model(heart_X, heart_y)
cancer_model, cancer_accuracy, cancer_precision, cancer_recall, cancer_f1 = train_model(cancer_X, cancer_y)

# Save models and scalers
joblib.dump(parkinsons_model, 'parkinsons_model.pkl')
joblib.dump(parkinsons_scaler, 'parkinsons_scaler.pkl')
joblib.dump(parkinsons_columns, 'parkinsons_columns.pkl')

joblib.dump(heart_model, 'heart_model.pkl')
joblib.dump(heart_scaler, 'heart_scaler.pkl')
joblib.dump(heart_columns, 'heart_columns.pkl')

joblib.dump(cancer_model, 'cancer_model.pkl')
joblib.dump(cancer_scaler, 'cancer_scaler.pkl')
joblib.dump(cancer_columns, 'cancer_columns.pkl')

# Load models and scalers
parkinsons_model = joblib.load('parkinsons_model.pkl')
parkinsons_scaler = joblib.load('parkinsons_scaler.pkl')
parkinsons_columns = joblib.load('parkinsons_columns.pkl')

heart_model = joblib.load('heart_model.pkl')
heart_scaler = joblib.load('heart_scaler.pkl')
heart_columns = joblib.load('heart_columns.pkl')

cancer_model = joblib.load('cancer_model.pkl')
cancer_scaler = joblib.load('cancer_scaler.pkl')
cancer_columns = joblib.load('cancer_columns.pkl')

# Streamlit app
st.title('Medical Diagnosis Using AI')
st.write('Enter the medical data to get a diagnosis.')

# Example input fields (customize based on your dataset)
age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=300, value=150)
# Add more input fields as needed

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'blood_pressure': [blood_pressure],
    'cholesterol': [cholesterol],
    # Add more columns as needed
})

# Preprocess the input data for each disease
def preprocess_input_data(input_data, scaler, columns):
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

parkinsons_input_data = preprocess_input_data(input_data, parkinsons_scaler, parkinsons_columns)
heart_input_data = preprocess_input_data(input_data, heart_scaler, heart_columns)
cancer_input_data = preprocess_input_data(input_data, cancer_scaler, cancer_columns)

# Make predictions
if st.button('Diagnose'):
    parkinsons_prediction = parkinsons_model.predict(parkinsons_input_data)
    parkinsons_prediction_proba = parkinsons_model.predict_proba(parkinsons_input_data)
    st.write(f'Parkinson\'s Diagnosis: {parkinsons_prediction[0]}')
    st.write(f'Parkinson\'s Probability: {parkinsons_prediction_proba[0]}')

    heart_prediction = heart_model.predict(heart_input_data)
    heart_prediction_proba = heart_model.predict_proba(heart_input_data)
    st.write(f'Heart Disease Diagnosis: {heart_prediction[0]}')
    st.write(f'Heart Disease Probability: {heart_prediction_proba[0]}')

    cancer_prediction = cancer_model.predict(cancer_input_data)
    cancer_prediction_proba = cancer_model.predict_proba(cancer_input_data)
    st.write(f'Cancer Diagnosis: {cancer_prediction[0]}')
    st.write(f'Cancer Probability: {cancer_prediction_proba[0]}')

# Display model evaluation metrics
st.write('Model Evaluation Metrics:')
st.write(f'Parkinson\'s - Accuracy: {parkinsons_accuracy}, Precision: {parkinsons_precision}, Recall: {parkinsons_recall}, F1 Score: {parkinsons_f1}')
st.write(f'Heart Disease - Accuracy: {heart_accuracy}, Precision: {heart_precision}, Recall: {heart_recall}, F1 Score: {heart_f1}')
st.write(f'Cancer - Accuracy: {cancer_accuracy}, Precision: {cancer_precision}, Recall: {cancer_recall}, F1 Score: {cancer_f1}')