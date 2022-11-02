import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Loading model weights
model = pickle.load(open("catboost_model_tuned.sav", "rb"))

columns = ['year','manufacturer','model','condition','cylinders','fuel','odometer','title_status','transmission','drive','type','paint_color','state']

# App components
st.title("What is the Fair Market Value (FMV) of a used car? :car:")
st.markdown("This is a machine learning web application, which utilizes a CatBoost Regression algorithm trained on the [US Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) for the purpose of predicting a car's price given its features. The predicted price is in USD.")

car_year = st.slider("Select Car's Year", 1960, 2022) 

col1, col2 = st.columns(2)

car_make = col1.text_input("Enter Car's Make", 'bmw')
car_model = col1.text_input("Enter Car's Model", 'm3')
car_condition = col1.selectbox("Select Car's Condition", ['salvage','fair','good','like new','excellent'])
car_cylinders = col1.selectbox("Select Number of Engine Cylinders", 
                             ['3 cylinders','4 cylinedrs','5 cylinders','6 cylinders','8 cylinders','10 cylinders','12 cylinders','other'])
car_fuel = col1.selectbox("Select Cars' Fuel Type",['gas','diesel','electric','hybrid','other'])
car_odometer = col1.number_input("Enter Car's Mileage in Miles",0,1000000)
car_title_status = col2.selectbox("Select Car's Title Status", ['rebuilt','salvage','missing','lien','parts only','clean']) 
car_transmission = col2.selectbox("Select Car's Transmission", ['automatic','manual','other'])
car_drive = col2.selectbox("Select Car's Drivetrain", ['fwd','rwd','4wd']) 
car_type = col2.selectbox("Select Car's Bodytype", ['sedan','SUV','pickup','truck','coupe','hatchback','wagon','van','convertible','mini-van','offroad','bus','other'])
car_color = col2.selectbox("Select Car's Color in lower case", ['white','black','silver','blue','red','grey','green','brown','custom','orange','yellow','purple'])
car_state = col2.text_input("Enter the Two Letter State Code. Ex: Enter ny for New York", 'ny')

# Making a prediction
def predict(): 
    row = np.array([car_year,car_make,car_model,car_condition,car_cylinders,car_fuel,car_odometer,
                    car_title_status,car_transmission,car_drive,car_type,car_color,car_state]) 
    X = pd.DataFrame([row], columns = columns)
    
    prediction = model.predict(X)
    prediction = int(prediction)
    result = "The FMV of this car is " + str(prediction) + " " + "USD"
    st.success(result, icon="✅")

trigger = st.button('Predict', on_click=predict)

st.write(" ")
st.markdown("This application was created by [Mohammed Adi](https://www.linkedin.com/in/mohammed-adi/)")