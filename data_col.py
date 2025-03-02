import yfinance as yf
import pandas as pd


ticker = yf.Ticker('XLE')
aapl_historical = ticker.history(period="max")
aapl_historical.to_csv('XLE.csv')