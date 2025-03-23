import numpy as np   
import pandas as pd 
import yfinance as yf 
from keras.models import load_model #type:ignore
import streamlit as st #type:ignore 
import matplotlib.pyplot as plt 

model = load_model(r'C:\Users\Vijay kumar\OneDrive\Desktop\ML project for stock market\stock pridiction model.keras')

st.header('stock Market pridiction')

stock = st.text_input('Enter stock Symbol','GOOG')
start = '2012-01-01'
end = '2022-01-31' 

data = yf.download(stock,start,end) 

st.subheader('stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)]) 

from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100) 
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('price  vs MA50')
ma_50_days = data.Close.rolling(50).mean()

# Create a figure with a specific size
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('price  vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2) 

st.subheader('price  vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)

x = [] 
y = []

for i in range(100,data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0]) 

x,y = np.array(x), np.array(y) 

predict = model.predict(x) 

scale = 1/scaler.scale_ 

predict = predict*scale 
y = y*scale 

st.subheader('Original Price  vs Pridicted Price')
fig4= plt.figure(figsize=(8, 6))
plt.plot(predict,'r',label='Original Price')
plt.plot(y,'g',label='Pridicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

