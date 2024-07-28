import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the model and preprocessing tools
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')
encoders = joblib.load('models/encoders.joblib')

# Function to preprocess input data
def preprocess_input(data, scaler, encoders, numeric_columns, categorical_columns):
    # Convert boolean columns to strings
    for col in categorical_columns:
        data[col] = data[col].astype(str)

    # Apply the same preprocessing as used during training
    data[numeric_columns] = scaler.transform(data[numeric_columns])

    for col in categorical_columns:
        encoder = encoders[col]
        # Handle unseen labels by mapping them to 'unknown'
        data[col] = data[col].apply(lambda x: x if x in encoder.classes_ else 'unknown')
        encoder.classes_ = np.append(encoder.classes_, 'unknown')
        data[col] = encoder.transform(data[col])

    return data

# Load the training data to get column names
data = pd.read_excel('data/heart_disease.xlsx', sheet_name='Heart_disease')
data.dropna(inplace=True)

X = data.drop(['fbs'], axis=1)
y = data['fbs']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Streamlit App
st.title("Heart Disease Prediction")

# Sidebar for input
st.sidebar.header("Patient Details")
st.sidebar.markdown("### Please enter the patient's details:")
# Collect user input in sidebar
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox("Sex", options=['male', 'female'])
cp = st.sidebar.selectbox("Chest Pain Type", options=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.sidebar.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=['normal', 'having ST-T wave abnormality', 'showing probable or definite left ventricular hypertrophy'])
thalch = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[True, False])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=['upsloping', 'flat', 'downsloping'])
thal = st.sidebar.selectbox("Thalassemia", options=['normal', 'fixed defect', 'reversible defect'])
num = st.sidebar.number_input("Num", min_value=0, max_value=4, value=0)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'thal': [thal],
    'num': [num]
})

# Ensure input data has the same columns as the training data
#input_data = input_data[numeric_columns + categorical_columns]

# Add a button to show results
if st.button("Show Results"):
    try:
        # Preprocess the input data
        input_data = preprocess_input(input_data, scaler, encoders, numeric_columns, categorical_columns)
        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 0:
            st.success("The model predicts that the patient does NOT have heart disease.")
        else:
            st.error("The model predicts that the patient has heart disease.")
    except ValueError as e:
        st.error(f"Input error: {e}")

# Add a footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            text-align: center;
        }
    </style>
    <div class="footer">
        <p>Heart Disease Prediction App &copy; 2024</p>
    </div>
""", unsafe_allow_html=True)
