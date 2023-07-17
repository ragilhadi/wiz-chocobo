import streamlit as st
import pandas as pd
import datetime
import pickle

st.set_page_config(
    page_title="Wiz Chocobo Prediction",
    page_icon=":rocket:"
)

if 'model' not in st.session_state:
    model = pickle.load(open('xgboost.sav', 'rb'))
    st.session_state['model'] = model

st.title(':magic_wand: Total Visitor Wiz Chocobo Post Prediction')
st.subheader('Please input the features below : ')

date= st.date_input("What\'s the date ? start from March 2012",datetime.date(2012, 3, 1))
season = st.selectbox(
    'How\'s the season ?',
    ('SPRING', 'SUMMER', 'FALL', 'WINTER'))
weather = st.selectbox(
    'How\'s the weather ?',
    ('clear/cloudy', 'mist', 'light rain/thunderstorm', 'heavy rain/fog'))
hour = st.number_input("What\'s the hour ?")
temperature = st.number_input("What\'s the temperature ?")
wind_speed = st.number_input("What\'s the wind speed ?")
remark = st.selectbox(
    'What\'s the remark today ?',
    ("New Year's Day", 'Amazingly nothing out of the ordinary',
       'Business as usual', 'Nothing happened',
       'Imperial Army sighted near post',
       'Nothing of particular interest',
       'A small batch of gysahl greens went bad today',
       'Surprisingly normal', '-',
       'Sahagins are out on a hunt near the lake', 'Boring',
       'No special remarks', 'nan', 'Snake? SNAKE?! SNAAAAAAAAAAAAKE!',
       'Boko wreaks havoc on the post. Nothing too serious',
       "It's quiet... Too quiet.", 'na', 'Day of the Oracle',
       'Situation normal',
       'Someone heard Deadeye in the distance, but no real threats',
       'Somnus Day', 'Starscourge Memorial Day', 'Great War Memorial Day',
       'Founding Day', 'Moogle Chocobo Carnival',
       'Lucis-Accordo Treaty Day', 'Crownsguards Day',
       'Festival of the Hunt', 'Feast of the Astrals'))

if st.button('Model Predict'):
    year = date.year
    month = date.month
    dayname = date.strftime('%A')
    data = pd.DataFrame({
        'season': [season],
        'year': [year],
        'month': [month],
        'hour': [hour],
        'weekday': [dayname],
        'weather': [weather],
        'temperature': [temperature],
        'wind_speed': [wind_speed],
        'remark': [remark]
    })
    result = st.session_state['model'].predict(data)
    st.write(f'Prediction model : {round(result[0])} person')
else:
    st.write('Please input the feature above to start modelling')
