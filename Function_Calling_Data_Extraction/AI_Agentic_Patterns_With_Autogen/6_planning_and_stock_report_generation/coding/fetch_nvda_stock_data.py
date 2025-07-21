# filename: fetch_nvda_stock_data.py
import yfinance as yf
import pandas as pd

# Setting the ticker and time period
ticker_symbol = 'NVDA'
start_date = '2024-03-23'
end_date = '2024-04-23'

# Fetching the historical data from Yahoo Finance
nvda_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Displaying the data
print(nvda_data)

# Calculating the percentage change in closing price from the start to the end of the period
percentage_change = ((nvda_data['Close'][-1] - nvda_data['Close'][0]) / nvda_data['Close'][0]) * 100
print(f"The percentage change in stock price from {start_date} to {end_date} is: {percentage_change:.2f}%")