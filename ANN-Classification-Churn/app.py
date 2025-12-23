import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
import os


# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    try:
        # Try different possible paths
        model_paths = [
            'churn_model.h5',
            './churn_model.h5',
            os.path.join(os.path.dirname(__file__), 'churn_model.h5')
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return tf.keras.models.load_model(path)
        
        # If none found, raise error
        raise FileNotFoundError("churn_model.h5 not found in any expected location")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_resource
def load_preprocessors():
    try:
        # Try different possible paths for pickle files
        base_paths = ['.', os.path.dirname(__file__)]
        
        onehot_encoder = None
        label_encoder_gender = None
        scaler = None
        
        for base_path in base_paths:
            try:
                with open(os.path.join(base_path, 'onehot_encoder_geo.pkl'), 'rb') as f:
                    onehot_encoder = pickle.load(f)
                
                with open(os.path.join(base_path, 'label_encoder_gender.pkl'), 'rb') as f:
                    label_encoder_gender = pickle.load(f)
                
                with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
                    scaler = pickle.load(f)
                
                break  # If successful, break out of loop
            except FileNotFoundError:
                continue
        
        if onehot_encoder is None or label_encoder_gender is None or scaler is None:
            raise FileNotFoundError("One or more preprocessor files not found")
        
        return onehot_encoder, label_encoder_gender, scaler
    
    except Exception as e:
        st.error(f"Error loading preprocessors: {str(e)}")
        st.stop()

# Load model and preprocessors
model = load_model()
onehot_encoder, label_encoder_gender, scaler = load_preprocessors()


# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'churn_model.h5'))

# Load the preprocessors
with open(os.path.join(os.path.dirname(__file__), 'onehot_encoder_geo.pkl'), 'rb') as f:
    onehot_encoder = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'label_encoder_gender.pkl'), 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)


## streamlit app
st.title("Customer Churn Prediction")

# Input fields
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=92)
tenure = st.slider("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", min_value=350, max_value=850)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")


# Prepare input data
# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# one hot Encoding 'Geography'
geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))


# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# since geography not added previously no need to drop it

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')