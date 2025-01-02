## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import requests
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Display Images
from PIL import Image
img = Image.open("crops.png")

st.set_page_config(
    page_title="Advanced Crop Prediction Solutions",
    page_icon="🌾",
    layout="wide",
)

# Display the project name at the top left side
st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text;'>KrishiGyaan: Empowering Farmers with Smart Solutions 🌿</h1>
""", unsafe_allow_html=True)



# Display the main image
st.image(img, width=700)

df = pd.read_csv('Crop_recommendation.csv')

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
labels = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)

# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    # Assuming you are using placeholder images
    image_path = os.path.join('crop_images', crop_name.lower() + '.jpeg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_container_width=True)
    else:
        # Placeholder image in case the crop-specific image is not found
        placeholder_image_path = 'crop_images/placeholder.jpeg'
        st.image(placeholder_image_path, caption=f"Recommended crop: {crop_name}", use_container_width=True)

import pickle
RF_pkl_filename = 'RF.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

RF_Model_pkl = pickle.load(open('RF.pkl', 'rb'))

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

# Function to submit feedback
def submit_feedback(name, email, feedback):
    url = "https://api.web3forms.com/submit"
    payload = {
        "access_key": "4ffcbd0a-8334-41a7-af0a-d8552c02dd27",
        "name": name,
        "email": email,
        "message": feedback
    }
    response = requests.post(url, data=payload)
    return response

# About page
def about():
    st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>About KrishiGyaan</h1>
""", unsafe_allow_html=True)

    st.markdown("""
        KrishiGyaan is an advanced crop prediction solution designed to assist farmers in selecting the most suitable crops based on environmental factors. 
        Utilizing machine learning algorithms, our platform analyzes soil nutrients, weather conditions, and other relevant data to recommend the best crops for maximum yield.

        ### How It Works:
        1. **Input Environmental Factors**: Enter the values for nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.
        2. **Prediction**: Our Random Forest model processes the input data to predict the most suitable crop.
        3. **Result**: The recommended crop is displayed along with an image for easy identification.
    """)


# Feedback form page
def feedback():
    st.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: center;'>Feedback / Suggestions</h1>
""", unsafe_allow_html=True)
    st.markdown("We value your feedback! Please provide your suggestions or feedback below:")

    with st.form(key='feedback_form', clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback_text = st.text_area("Your Feedback/Suggestions", height=150)
        
        submit_feedback_btn = st.form_submit_button(label='Submit Feedback')
        
        if submit_feedback_btn:
            if name and email and feedback_text:
                response = submit_feedback(name, email, feedback_text)
                if response.status_code == 200:
                    st.success("Your feedback has been submitted successfully! Thank you.")
                else:
                    st.error("Failed to submit feedback. Please try again later.")
            else:
                st.warning("Please fill out all fields.")

# Main Crop Prediction page
def crop_prediction():
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(270deg, #3C8D40, #6A9B4D); color: white; 
    text-shadow: 2px 2px 5px rgba(0,0,0,0.7);'>ADVANCED CROP PREDICTION SOLUTIONS</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Input Environmental Factors")
        nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    
    with col2:
        st.header("Predict Crop")
        # Predict button below the last input field
        if st.button("Predict"):
            inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
                st.error("Please fill in all input fields with valid values before predicting.")
            else:
                prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                st.success(f"The recommended crop is: {prediction[0]}")
                show_crop_image(prediction[0])

                st.markdown("<h3 style='text-align: center; font-weight: bold; color: #3C8D40;'>Your crops are ready to thrive. Start planting for success! 🌱</h3>", unsafe_allow_html=True)


# Sidebar navigation
st.sidebar.markdown("""
    <h1 style='font-weight: bold; color: transparent; background: linear-gradient(270deg, #CDDC39, #4FC3F7); 
    -webkit-background-clip: text; text-align: left;'>WELCOME</h1>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Crop Prediction", "About", "Feedback"])

# Navigate based on sidebar selection
if page == "Crop Prediction":
    crop_prediction()
elif page == "About":
    about()
elif page == "Feedback":
    feedback()
