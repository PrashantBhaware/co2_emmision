import streamlit as st
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('co2_emissions (1).csv',sep=';')

filename = "standard_scaler.pkl"
with open(filename, 'rb') as file:
    scaler = pickle.load(file)


filename1 ="final_model.pkl"
with open(filename1, 'rb') as file:
    model_ = pickle.load(file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # title of project
    st.title("Car Co2 Emission Predictor")

    # Taking input features from user
    car_brand = st.selectbox("Select Car Brand:", data.make.unique())

    model = st.selectbox('Select Car model:', data.loc[data.make == car_brand]['model'].unique())

    vehicle_class = st.selectbox("Vehicle Class Type:", data.loc[data.model==model]['vehicle_class'].unique())

    engine_size = st.selectbox("Engine Size:", data.engine_size.sort_values(ascending=True).unique())

    cylinders = st.selectbox("Cylinders:", data['cylinders'].sort_values().unique())

    transmission = st.selectbox("Transmision Type:",('Automatic with Select Shift', 'Manual', 'Continuously Variable', 'Automated Manual', 'Automatic'))

    fuel_type = st.selectbox("Fuel Type", ('Premium Gasoline','Diesel','Regular Gasoline','Ethanol(E85)','Natural Gas'))

    fuel_consp_city = st.slider("Fuel Consumption (City): ", 0.0, 25.0, 10.0)

    fuel_consp_hwy = st.slider("Fuel Consumption (Highway): ", 0.0, 25.0, 10.0)

    fuel_consp_comb = st.slider("Fuel Consumption(l/100km)", 0.0,20.0,10.0)

    fuel_consp_comb_mpg = st.slider("Fuel consumption(mpg)", 15,50,30)

    user_input = {
                    'Car Brand': [car_brand],
                    'Car Model': [model],
                    'Vehicle Class': [vehicle_class],
                    'engine_size': [engine_size],
                    'Cylinders': [cylinders],
                    'Transmission Type': [transmission],
                    'Fuel Type': [fuel_type],
                    'fuel_consumption_city': [fuel_consp_city],
                    'fuel_consumption_hwy': [fuel_consp_hwy],
                    'fuel_consumption_comb(l/100km)': [fuel_consp_comb],
                    'fuel_consumption_comb(mpg)': [fuel_consp_comb_mpg]
                  }

    user_input_df = pd.DataFrame(user_input)

    x_test=user_input_df[['fuel_consumption_hwy','fuel_consumption_comb(mpg)','fuel_consumption_city','fuel_consumption_comb(l/100km)','engine_size']]

    x_test_scaled = scaler.transform(x_test)

    # y_pred = model_.predict(x_test_scaled)

    # st.button('Predict Co2 Emission')

    if st.button('Predict Co2 Emission'):
        # When the button is clicked, perform the prediction/action
        prediction_result = model_.predict(x_test_scaled)
        # Display the result or output
        st.write('Co2 emission for your car is:', str(int(prediction_result[0])) + ' g/km')

    # st.text_input('Co2 emission for your car is:', str(int(y_pred[0])) + ' g/km')

    # Display the header with the custom style
    st.markdown('<div class="header">Get in Touch With me!</div>', unsafe_allow_html=True)

    contact_form = """
    <form action = "https://formsubmit.co/prashantbhaware2018@gmail.com" method = "POST" >
       <input type="hidden" name="_captcha" value="false">
       <input type = "text" name = "name" placeholder="Enter your name" required>
       <input type = "email" name = "email" placeholder="Enter your e-mail" required>
       <textarea name="message"  placeholder="Type your message here..." required></textarea>
       <button type = "submit" > Submit your response.. </button>
   </form>
   """
    st.markdown(contact_form, unsafe_allow_html=True)

    #use local css file
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")

