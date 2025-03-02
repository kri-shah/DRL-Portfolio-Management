import yfinance as yf
import pandas as pd

def col_data(ticker):
    t = yf.Ticker(ticker)
    t_hist = t.history(period="max")
    t_hist.to_csv(f'{ticker}.csv')


def main():
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    for sector in sectors:
        col_data(sector)

if __name__ == '__main__':
    main()