import numpy as np
import pandas as pd
import streamlit as st
import sklearn


from xgboost import XGBRegressor
from mapie.regression import MapieRegressor

import pickle
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_df(file):
    x = pd.read_csv(file)
    x['date_time'] = pd.to_datetime(x['date_time'])
    x['month'] = x['date_time'].dt.month_name()
    x['day'] = x['date_time'].dt.day_name()
    x['hour'] = x['date_time'].dt.hour
    x = x.drop(columns = ['date_time', 'traffic_volume'])
    return x

traffic_df = load_df('Traffic_Volume.csv')

@st.cache_resource
def load_pickle(model):
    xg_pickle = open(model, 'rb')
    x = pickle.load(xg_pickle)
    xg_pickle.close()
    return x

xg_ml = load_pickle('xg_ml.pickle')

st.title('Traffic Volume Predictor')
st.text('Use this advanced Machine Learning application to predict traffic volume')
st.image('traffic_image.gif')

st.sidebar.image('traffic_sidebar.jpg')
st.sidebar.caption('Traffic volume predictor')
st.sidebar.subheader('Input Features')
st.sidebar.text('You can either upload a CSV file or manually enter input features')
with st.sidebar.expander('Option 1: Upload CSV file'):
    uploaded_df = st.file_uploader('Upload a CSV file containing traffic details')
    st.subheader('Sample Data Format for Upload')
    st.dataframe(traffic_df.head(), width = 'stretch')
    st.warning('Ensure your uploaded file has the same column names and data types as shown above')

with st.sidebar.expander('Option 2: Fill Out Form'):
    st.text('Enter the traffic details manually using the form below')
    holiday = st.selectbox('Choose if today is a holiday or not', options = [None, 'Christmas Day', 'Columbus Day', 'Independence Day', 'Labor Day',
                                                                                     'MLK Day, Memorial Day', 'New Years Day', 'State Fair', 'Thanksgiving Day',
                                                                                     'Veterans Day', 'Washingtons Birthday'])
    temperature = st.number_input('Average temperature in Kelvin', value = float(290))
    rainfall = st.number_input('Amount in mm of rainfall that occurred in the hour', value = float(0))
    snowfall = st.number_input('Amount in mmy of snowfall that occurred in the hour', value = float(0))
    cloud_cov = st.number_input('Percentage of cloud coverage', value = 50, step = 1)
    curr_weather = st.selectbox('Choose the current weather', options = ['Clouds', 'Clear', 'Mist', 'Rain',
                                                                                 'Snow', 'Drizzle', 'Haze', 'Thunderstorm',
                                                                                 'Fog', 'Smoke', 'Squall'])
    month = st.selectbox('Choose month', options = ['January', 'February', 'March', 'April',
                                                            'May', 'June', 'July', 'August', 'September',
                                                            'October', 'November', 'December'])
    day = st.selectbox('Choose day', options = ['Monday', 'Tuesday', 'Wednesday',
                                                        'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour = st.selectbox('Choose hour', options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                          11, 12, 13, 14, 15, 16, 17, 18,
                                                          19, 20, 21, 22, 23, 24], help = 'Military time (e.g. 22 = 10pm [22-12])')
    button = st.button('Submit Form Data')

if uploaded_df is None:
    if button:
        st.success('Form data submitted successfully')
    else:
        st.info('Please choose a data input method to proceed')

elif uploaded_df is not None:
    st.success('CSV file uploaded successfully')

alpha = st.slider('Select alpha value for prediction intervals', min_value = 0.01, max_value = 0.5, value = 0.1)

if 'user_row' not in st.session_state:
    st.session_state['user_row'] = None

st.header('Predicting Traffic Volume...')

if uploaded_df is None:
    if button:
        user_inputs = [holiday, temperature, rainfall, snowfall, cloud_cov, curr_weather, month, day, hour]
        user_df = traffic_df.copy()
        user_df.loc[len(user_df)] = user_inputs

        dummy_user_df = pd.get_dummies(user_df)

        user_df2 = dummy_user_df.tail(1)
        st.session_state['user_row'] = user_df2

    if st.session_state['user_row'] is not None:
        user_df2 = st.session_state['user_row']
        y_pred, y_pred_int = xg_ml.predict(user_df2, alpha = alpha)

        st.metric('Predicted Traffic Volume', int(y_pred))

        low = y_pred_int[0][0]
        high = y_pred_int[0][1]
        st.markdown(f"**Prediction Interval** ({int((1-alpha)*100)}%): [{int(low)}, {int(high)}]")

elif uploaded_df is not None:
    user_uploaded_df = pd.read_csv(uploaded_df)
    encoded = traffic_df.copy()

    user_uploaded_df.columns = encoded.columns

    concat_df = pd.concat([encoded, user_uploaded_df], axis=0)

    rows = encoded.shape[0]

    concat_dummy_df = pd.get_dummies(concat_df)

    user_dummy_df = concat_dummy_df[rows:]

    y_pred, y_pred_int = xg_ml.predict(user_dummy_df, alpha = alpha)

    lower = []
    upper = []
    for i in y_pred_int:
        lower.append(int(i[0]))
    
    for j in y_pred_int:
        upper.append(int(j[1]))
    
    user_uploaded_df['Predicted Traffic Volume'] = y_pred
    user_uploaded_df['Lower Limit'] = lower
    user_uploaded_df['Upper Limit'] = upper

    st.subheader(f'Prediction Results with {int((1-alpha)*100)}% Prediction Interval')
    st.dataframe(user_uploaded_df)

st.subheader('Model Performance and Insights')

tab1, tab2, tab3, tab4 = st.tabs(['Feature Importance', 'Histogram of Residuals', 'Predicted vs. Actual', 'Coverage Plot'])

with tab1:
    st.subheader('Feature Importance')
    st.image('xg_feature_importance.svg')
    st.caption('Relative importance of features in prediction')

with tab2:
    st.subheader('Histogram of Residuals')
    st.image('xg_res_hist.svg')
    st.caption('Distribution of residuals to evaluate prediction quality')

with tab3:
    st.subheader('Plot of Predicted vs. Actual')
    st.image('xg_scatter.svg')
    st.caption('Visual comparison of predicted and actual values')

with tab4:
    st.subheader('Coverage Plot')
    st.image('xg_coverage.svg')
    st.caption('Range of predictions with confidence intervals')

# For this app, I utilized CoPilot(AI) to debug code whenever I ran into an error.  Generally,
# the errors were simple typos that the AI found quickly.  I also used it to assist me in designing
# the session_state code to allow the alpha slider to run independently from the button.  Finally,
# I used AI to help me understand how the XGboost machine learning model works.
