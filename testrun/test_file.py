import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from model import model_train


st.write(""" 
# Product Average Monthly Sales Predictor API
This app is used to show the predicted **Average Monthly Sales** of Product at a Particular Depot for a given **Month**.
""")

#create variable for locaions from train df
path = 'utils\data\data-1671661749260.csv'


df = pd.read_csv(path)
df.drop(columns = ['region','tms','year'], inplace=True)
locations = df.depot.unique().tolist()
locations[0] = 'LOCATION'

st.subheader('Select The Depot Location from the Dropdown')
location = st.selectbox(label='Depot Locations', options = locations)

st.subheader('Input the Product Identifier(item_no)')
product = st.text_input("")

st.subheader('Input The Target Month')
month = st.number_input('Select Month', min_value=1, max_value=12)

json_obj = {'depot':location,'item_no':product.upper(),
                'month':month}

st.subheader('Hit Button to Make Prediction')
if st.button(label='Predict', on_click = model_train, args = (df,json_obj, month)):

    metrics, pred = model_train(df,json_obj, month)

    st.subheader('Predicted Result')
    st.write(pred)
    st.write(metrics)