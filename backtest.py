# backtest.py — Trading simulation engine

import numpy as np


def decode(chrom):
    """Convert raw chromosome values to valid trading parameters"""
    rsi_buy, rsi_sell, ma_s, ma_l, sl, tp = chrom
    ma_short = min([10, 20, 50],   key=lambda x: abs(x - ma_s))
    ma_long  = min([50, 100, 200], key=lambda x: abs(x - ma_l))
    if ma_short == ma_long:
        ma_long = 200
    return rsi_buy, rsi_sell, ma_short, ma_long, sl, tp


def decode_t(chrom):
    """Decode chromosome with transformer weight"""
    rsi_buy, rsi_sell, ma_s, ma_l, sl, tp, tw = chrom
    ma_short = min([10, 20, 50],   key=lambda x: abs(x - ma_s))
    ma_long  = min([50, 100, 200], key=lambda x: abs(x - ma_l))
    if ma_short == ma_long:
        ma_long = 200
    return rsi_buy, rsi_sell, ma_short, ma_long, sl, tp, tw


def _compute_metrics(portfolio, capital, df, trades):
    """Compute Sharpe, return, drawdown from portfolio values"""
    total_return = (capital - 100000) / 100000
    port         = np.array(portfolio)
    daily_ret    = np.diff(port) / (port[:-1] + 1e-9)

    if len(daily_ret) > 1 and daily_ret.std() > 1e-6:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    sharpe = float(np.clip(sharpe, -10, 10))
    peak   = np.maximum.accumulate(port)
    dd     = (port - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    return {
        'sharpe'      : round(sharpe, 4),
        'total_return': round(total_return * 100, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'final_value' : round(capital, 2),
        'num_trades'  : len([t for t in trades if t['type'] == 'BUY']),
        'trades'      : trades,
        'portfolio'   : [round(v, 2) for v in portfolio],
        'dates'       : [str(df.iloc[i]['Date'].date())
                         for i in range(len(df))]
    }


def backtest(df, rsi_buy, rsi_sell, ma_short, ma_long,
             stop_loss, take_profit):
    """Run backtest without transformer signal"""
    capital   = 100000.0
    shares    = 0.0
    buy_price = 0.0
    portfolio = []
    trades    = []
    df        = df.reset_index(drop=True)

    for i in range(len(df)):
        row   = df.iloc[i]
        price = float(row['Close'])
        rsi   = float(row['RSI_14'])
        date  = str(row['Date'].date())
        sma_s = f'SMA_{int(ma_short)}'
        sma_l = f'SMA_{int(ma_long)}'

        if sma_s not in df.columns or sma_l not in df.columns:
            portfolio.append(capital + shares * price)
            continue

        # BUY signal
        if (shares == 0
                and rsi < rsi_buy
                and float(row[sma_s]) > float(row[sma_l])):
            shares    = capital // price
            capital  -= shares * price
            buy_price = price
            trades.append({'type': 'BUY', 'date': date,
                           'price': round(price, 2)})

        # SELL signal
        elif shares > 0:
            pnl = (price - buy_price) / buy_price
            if (rsi > rsi_sell
                    or pnl <= -stop_loss
                    or pnl >= take_profit):
                capital += shares * price
                trades.append({'type': 'SELL', 'date': date,
                               'price': round(price, 2),
                               'pnl': round(pnl * 100, 2)})
                shares    = 0.0
                buy_price = 0.0

        portfolio.append(capital + shares * price)

    # Close any open position at end
    if shares > 0:
        final_price   = float(df.iloc[-1]['Close'])
        capital      += shares * final_price
        portfolio[-1] = capital

    return _compute_metrics(portfolio, capital, df, trades)


def backtest_with_signal(df, rsi_buy, rsi_sell, ma_short, ma_long,
                          stop_loss, take_profit, t_weight):
    """Run backtest WITH transformer signal adjusting RSI threshold"""
    capital   = 100000.0
    shares    = 0.0
    buy_price = 0.0
    portfolio = []
    trades    = []
    df        = df.reset_index(drop=True)

    for i in range(len(df)):
        row   = df.iloc[i]
        price = float(row['Close'])
        rsi   = float(row['RSI_14'])
        date  = str(row['Date'].date())
        sma_s = f'SMA_{int(ma_short)}'
        sma_l = f'SMA_{int(ma_long)}'

        if sma_s not in df.columns or sma_l not in df.columns:
            portfolio.append(capital + shares * price)
            continue

        # Transformer adjusts effective RSI buy threshold
        t_signal = float(row.get('Transformer_Signal', 0.5))
        eff_rsi  = rsi_buy - t_weight * (t_signal - 0.5) * 20

        # BUY signal
        if (shares == 0
                and rsi < eff_rsi
                and float(row[sma_s]) > float(row[sma_l])):
            shares    = capital // price
            capital  -= shares * price
            buy_price = price
            trades.append({'type': 'BUY', 'date': date,
                           'price': round(price, 2)})

        # SELL signal
        elif shares > 0:
            pnl = (price - buy_price) / buy_price
            if (rsi > rsi_sell
                    or pnl <= -stop_loss
                    or pnl >= take_profit):
                capital += shares * price
                trades.append({'type': 'SELL', 'date': date,
                               'price': round(price, 2),
                               'pnl': round(pnl * 100, 2)})
                shares    = 0.0
                buy_price = 0.0

        portfolio.append(capital + shares * price)

    if shares > 0:
        final_price   = float(df.iloc[-1]['Close'])
        capital      += shares * final_price
        portfolio[-1] = capital

    return _compute_metrics(portfolio, capital, df, trades)