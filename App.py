import streamlit as st
import numpy as np
import pickle

# Load the trained Random Forest Classifier model
model_file = 'rfc.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Mobile Phone Price Predictor", page_icon="📱", layout="wide")

st.title('📱 Machine Learning-Driven Price Analysis')

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Phone attributes")
    # Create three columns for inputs
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        battery_power = st.slider('Battery (mAh)', 500, 2000, 1000)
        fc = st.slider('Front Camera (MP)', 0, 20, 10)
        int_memory = st.slider('ROM (GB)', 2, 128, 64)
        mobile_wt = st.slider('Mobile Weight (g)', 80, 200, 150)
        four_g = st.selectbox("5G", ["No", "Yes"])

    with input_col2:
        pc = st.slider('Primary Camera (MP)', 0, 20, 10)
        ram = st.slider("RAM (MB)", 256, 8192, 2048)
        px_height = st.slider('Mobile Height', 0, 2000, 1000)
        px_width = st.slider('Mobile Width', 0, 2000, 1000)
        talk_time = st.slider('Battery Standby (hours)', 2, 20, 10)

        

with col2:
    st.subheader("Expected range for this device")
    
    # Convert categorical values to numerical
    
    four_g = 1 if four_g == "Yes" else 0
    

    # Define price range dictionary
    price_range = {0: "Economy", 1: "Moderate", 2: "Premiumt", 3: "Ultra-Premium"}


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
        st.success(f'The price range for your perfect mobile: {price_range[prediction]}')


     # Additional developer details
st.write('**Built by: Venkatesh Sahadevan**')
st.write('**Linked in** [Venkatesh S](https://www.linkedin.com/in/venkatesh-s-78554a29a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app )')
st.write('**GitHub:** [Venkatesh S](https://github.com/vkvenkat18/)')
