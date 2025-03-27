import yfinance as yf
import pandas as pd
import pytz
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def collect_data(ticker: str, save_path: str = "sector_data"):
    """Fetches historical data for a given ticker and saves it as a CSV file."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    t = yf.Ticker(ticker)
    t_hist = t.history(period="max")
    t_hist.to_csv(save_dir / f"{ticker}.csv")

def fix_vix_date(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes the VIX dataset's date format by converting it to EST."""
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    
    eastern = pytz.timezone("America/New_York")
    df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert(eastern)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    
    return df

def load_price_data(filenames: list, data_path: str = "sector_data") -> pd.DataFrame:
    """Loads price data from CSV files, processes dates, and merges into a single DataFrame."""
    data_frames = []
    data_dir = Path(data_path)

    for file in filenames:
        file_path = data_dir / f"{file}.csv"
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

        if file == "VIX":
            df = fix_vix_date(df)
            df = df.set_index("Date")

        df = df.rename(columns={"Close": file})
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
        df = df.set_index("Date")

        data_frames.append(df[[file]])

    return pd.concat(data_frames, axis=1).dropna()

def add_vol(df):
    """Calculating and adding volitility indicators"""
    df['Return'] = df['SPY'].pct_change()
    df['vol20'] = df['Return'].rolling(window=20).std()
    df['vol60'] = df['Return'].rolling(window=60).std()
    df['volRatio'] = df['vol20'] / df['vol60']

def main():
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU", "SPY", "VIX"]
    for sector in sectors:
        collect_data(sector)
    price_data = load_price_data(sectors)
    add_vol(price_data)
    
    price_data.to_csv("all_sector_data.csv", index=True)
    print(price_data.head())

if __name__ == "__main__":
    main()
