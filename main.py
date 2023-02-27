import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
import pandas as pd
from PIL import Image
import pickle


st.sidebar.title("About")
st.sidebar.info("Forecasting Rainfall at Waduk Selorejo using 'NeuralProphet' Machine Learning model.")

def get_input():
	st.sidebar.header("Input From user")
	st.sidebar.subheader("Select range of Date for visualize data for particular date range.")
	st.sidebar.write("(From 01-01-1983 to 31-12-2020)")
	start_date = st.sidebar.text_input("Start Date", "01-01-1983")
	end_date = st.sidebar.text_input("End Date", "31-12-2020")
	st.write("")
	st.sidebar.subheader("Enter Period for Forecasting of Rainfall")
	period = st.sidebar.text_input("Period (In Days)", "30")
	return start_date, end_date, period

image = Image.open('rainfall-forcasting.jpeg')
st.image(image, use_column_width=True)

START = "01-01-1983"
TODAY = date.today().strftime("%d-%M-%Y")

st.title('Rainfall Forecasting')

def get_data(start, end):
	df = pd.read_csv('CDR_198301_202003.csv')

	start = pd.to_datetime(start)
	end = pd.to_datetime(end)

	start_row = 0
	end_row = 0

	for i in range(0, len(df)):
		if start <=	pd.to_datetime(df['Date'][i]):
			start_row = i
			break

	for j in range(0, len(df)):
		if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
			end_row = len(df) - 1 - j
			break

	df = df.set_index(pd.DatetimeIndex(df['Date'].values))

	return df.iloc[start_row:end_row+1, :]

start, end, period = get_input()
data = get_data(start, end)

st.subheader("Data")

st.write("First 5 Columns")
st.write(data.head())
st.write("Last 5 Columns")
st.write(data.tail())

st.subheader('Rain')
st.write("Zoom In/Zoom Out for better visualization.")
st.line_chart(data[['Rain']])

st.subheader("Data Statistics")
st.write(data.describe())

st.header("Prediction")

def model_np():
	model = pickle.load(open('save-model.pkl', 'rb'))

	st.subheader("Using NeuralProphet")
	df = data.copy()
	df.reset_index(inplace=True)
	df_train = df[['Date','Rain']]
	df_train = df_train.rename(columns={"Date": "ds", "Rain": "y"})


	future = model.make_future_dataframe(df_train, periods=int(period))
	forecast = model.predict(future)
	forecast = forecast.rename(columns={"ds": "Date", "yhat1": "Rain"})
	st.write("Forecasting of Rainfall from 02-12-2019 to 31-12-2020")
	st.write(forecast[['Date', 'Rain']])
	st.write(f"Forecasting of Rainfall of {period} days")
	st.line_chart(forecast['Rain'])

model_np()