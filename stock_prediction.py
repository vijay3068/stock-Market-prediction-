import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to get stock data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period)
    return stock_data

# Function to prepare the data for machine learning model
def prepare_data(stock_data):
    stock_data = stock_data[['Close']].copy()  # Create a copy to avoid warnings
    stock_data['Prediction'] = stock_data['Close'].shift(-30)
    stock_data.dropna(inplace=True)
    X = stock_data[['Close']].values
    y = stock_data['Prediction'].values
    return X, y, stock_data

# Function to train the model and make predictions
def train_predict_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, y_test

# Function to plot the predictions vs actual data
def plot_results(actual_prices, predicted_prices, ticker):
    plt.figure(figsize=(10,6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.legend(loc='best')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()  # This will display the plot

# Main execution
if __name__ == "__main__":
    ticker = 'AAPL'  # You can change the ticker here
    stock_data = get_stock_data(ticker)
    
    # Prepare data for prediction
    X, y, stock_data = prepare_data(stock_data)
    
    # Train the model and make predictions
    predictions, y_test = train_predict_model(X, y)
    
    # Plot the results
    plot_results(y_test, predictions, ticker)
