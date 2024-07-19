import streamlit as st
import numpy as np
import pickle

# Load the trained Random Forest Classifier model
model_file = 'rfc.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Mobile Phone Price Predictor", page_icon="📱", layout="wide")

st.title('📱 Mobile Phone Price Range Predictor')
st.write("Predict the price range of a mobile phone based on its specifications.")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Phone Specifications")
    
    # Create three columns for inputs
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        battery_power = st.slider('Battery (mAh)', 500, 2000, 1000)
        fc = st.slider('Front Camera (MP)', 0, 20, 10)
        int_memory = st.slider('Internal Memory (GB)', 2, 128, 64)
        mobile_wt = st.slider('Mobile Weight (g)', 80, 200, 150)
        four_g = st.selectbox("4G", ["No", "Yes"])

    with input_col2:
        pc = st.slider('Primary Camera (MP)', 0, 20, 10)
        ram = st.slider("RAM (MB)", 256, 8192, 2048)
        px_height = st.slider('Pixel Resolution Height', 0, 2000, 1000)
        px_width = st.slider('Pixel Resolution Width', 0, 2000, 1000)
        talk_time = st.slider('Talk Time (hours)', 2, 20, 10)

        

with col2:
    st.subheader("Prediction")
    
    # Convert categorical values to numerical
    
    four_g = 1 if four_g == "Yes" else 0
    

    # Define price range dictionary
    price_range = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

    # Predict price range
    if st.button('Predict Price Range'):
        # Collect feature values
        feature_values = [battery_power, fc, four_g, int_memory, mobile_wt, pc,
       px_height, px_width, ram, talk_time]
        
        # Convert the list of feature values into numpy array
        feature_values_array = np.array(feature_values).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_values_array)[0]
        
        # Display predicted price range
        st.success(f'The predicted price range is: {price_range[prediction]}')
