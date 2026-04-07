# data.py — Fetches Bank Nifty data and computes indicators

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# All features used by Transformer and GA/PSO
FEATURE_COLS = [
    'Close', 'RSI_14', 'RSI_7',
    'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
    'MACD', 'MACD_Sig', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Pos',
    'ATR_Norm', 'Ret_5d', 'Ret_10d', 'Ret_20d', 'Vol_20'
]


def fetch_data(symbol="^NSEBANK", start="2007-01-01", end="2025-06-01"):
    """Download OHLC data from Yahoo Finance"""
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    df = raw[['Open', 'High', 'Low', 'Close']].reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().reset_index(drop=True)
    return df


def add_indicators(df):
    """Compute all 20 technical indicators"""
    df    = df.copy()
    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # RSI
    def rsi(series, p):
        d  = series.diff()
        ag = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
        al = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
        return 100 - 100 / (1 + ag / al)

    df['RSI_14'] = rsi(close, 14)
    df['RSI_7']  = rsi(close, 7)

    # Moving Averages
    for p in [10, 20, 50, 200]:
        df[f'SMA_{p}'] = close.rolling(p).mean()
    for p in [10, 20, 50]:
        df[f'EMA_{p}'] = close.ewm(span=p, adjust=False).mean()

    # MACD
    e12             = close.ewm(span=12, adjust=False).mean()
    e26             = close.ewm(span=26, adjust=False).mean()
    df['MACD']      = e12 - e26
    df['MACD_Sig']  = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Sig']

    # Bollinger Bands
    ma20           = close.rolling(20).mean()
    sd20           = close.rolling(20).std()
    df['BB_Upper'] = ma20 + 2 * sd20
    df['BB_Lower'] = ma20 - 2 * sd20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / ma20
    df['BB_Pos']   = (close - df['BB_Lower']) / (
                      df['BB_Upper'] - df['BB_Lower'])

    # ATR
    pc             = close.shift(1)
    tr             = pd.concat([high - low,
                                (high - pc).abs(),
                                (low  - pc).abs()], axis=1).max(axis=1)
    df['ATR_14']   = tr.rolling(14).mean()
    df['ATR_Norm'] = df['ATR_14'] / close

    # Return features
    for p in [5, 10, 20]:
        df[f'Ret_{p}d'] = close.pct_change(p)

    df['Daily_Return'] = close.pct_change()
    df['Vol_20']       = df['Daily_Return'].rolling(20).std()

    return df


def prepare_data(df):
    """Split into train/val/test and scale features"""
    df = add_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    n         = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Scale — fit ONLY on train data
    scaler       = MinMaxScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[FEATURE_COLS] = scaler.fit_transform(
        train_df[FEATURE_COLS])
    val_scaled[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_scaled[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    return (train_df, val_df, test_df,
            train_scaled, val_scaled, test_scaled, scaler)