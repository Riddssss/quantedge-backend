# main.py — FastAPI backend (all endpoints)

import time
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data import fetch_data, add_indicators, prepare_data, FEATURE_COLS
from backtest import decode, decode_t, backtest, backtest_with_signal
from ga import run_ga
from pso import run_pso
from transformer import generate_transformer_signals

app = FastAPI(title="Trading Strategy API")

# Allow all origins — fixes CORS error from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Sector symbol mapping
SECTOR_NAMES = {
    "^NSEBANK"   : "Bank Nifty",
    "^CNXIT"     : "Nifty IT",
    "^NSEI"      : "Nifty 50",
    "^CNXPHARMA" : "Nifty Pharma",
    "^CNXAUTO"   : "Nifty Auto",
}

# In-memory cache
_cache      = {}
_live_cache = {}


def get_data(start_date="2007-01-01", end_date="2025-06-01",
             sector="^NSEBANK"):
    cache_key = f"data_{sector}_{start_date}_{end_date}"
    if cache_key not in _cache:
        sector_name = SECTOR_NAMES.get(sector, sector)
        print(f"Fetching {sector_name} data {start_date} to {end_date}...")
        df = fetch_data(symbol=sector, start=start_date, end=end_date)
        (train_df, val_df, test_df,
         train_scaled, val_scaled,
         test_scaled, scaler) = prepare_data(df)
        _cache[cache_key] = {
            'train_df'     : train_df,
            'val_df'       : val_df,
            'test_df'      : test_df,
            'train_scaled' : train_scaled,
            'test_scaled'  : test_scaled,
        }
        print(f"Data ready — {len(df)} rows loaded")
    return _cache[cache_key]


# ── Request body models ───────────────────────────────────────

class OptimizeRequest(BaseModel):
    pop_size        : Optional[int]  = 50
    generations     : Optional[int]  = 40
    use_transformer : Optional[bool] = True
    seq_len         : Optional[int]  = 60
    start_date      : Optional[str]  = "2007-01-01"
    end_date        : Optional[str]  = "2025-06-01"
    sector          : Optional[str]  = "^NSEBANK"


class PaperTradeRequest(BaseModel):
    model           : str            = "PSO"
    capital         : float          = 100000
    use_transformer : Optional[bool] = True


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Trading Strategy API is running ✅"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sectors")
def get_sectors():
    """Return available sectors"""
    return {
        "sectors": [
            {"symbol": k, "name": v}
            for k, v in SECTOR_NAMES.items()
        ]
    }


@app.options("/api/optimize")
def options_optimize():
    return {}


@app.options("/api/paper-trade")
def options_paper_trade():
    return {}


@app.options("/api/transformer-window")
def options_transformer_window():
    return {}


@app.options("/api/live-signal")
def options_live_signal():
    return {}


@app.get("/api/live-signal")
def live_signal(sector: str = "^NSEBANK"):
    """Get today's live market signal for selected sector"""

    # Return cached if less than 1 hour old
    cache_key = f"live_{sector}"
    if (cache_key in _live_cache and
            _live_cache[cache_key].get("timestamp") is not None and
            time.time() - _live_cache[cache_key]["timestamp"] < 3600):
        print(f"Returning cached live signal for {sector}...")
        return _live_cache[cache_key]["data"]

    # Fetch latest data up to today
    today       = str(date.today())
    sector_name = SECTOR_NAMES.get(sector, sector)
    print(f"Fetching live {sector_name} data up to {today}...")
    df = fetch_data(symbol=sector, start="2020-01-01", end=today)
    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)

    # Get last two rows
    row      = df.iloc[-1]
    prev_row = df.iloc[-2]

    # Extract values
    close      = float(row['Close'])
    prev_close = float(prev_row['Close'])
    change_pct = (close - prev_close) / prev_close * 100
    rsi_14     = float(row['RSI_14'])
    sma_50     = float(row['SMA_50'])
    sma_200    = float(row['SMA_200'])
    macd       = float(row['MACD'])
    macd_sig   = float(row['MACD_Sig'])
    bb_pos     = float(row['BB_Pos'])

    # Compute signals
    ma_cross    = "Bullish" if sma_50 > sma_200 else "Bearish"
    rsi_signal  = ("Oversold"   if rsi_14 < 30
                   else "Overbought" if rsi_14 > 70
                   else "Neutral")
    macd_status = "Bullish" if macd > macd_sig else "Bearish"
    bb_signal   = ("Oversold"   if bb_pos < 0.2
                   else "Overbought" if bb_pos > 0.8
                   else "Neutral")

    # Count bullish signals
    bullish = 0
    if ma_cross    == "Bullish":  bullish += 1
    if rsi_signal  == "Oversold": bullish += 1
    if macd_status == "Bullish":  bullish += 1
    if bb_signal   == "Oversold": bullish += 1

    recommendation = ("BUY"  if bullish >= 3
                      else "SELL" if bullish <= 1
                      else "HOLD")

    result = {
        "sector"        : sector,
        "sector_name"   : sector_name,
        "current_date"  : str(row['Date'].date()),
        "close_price"   : round(close, 2),
        "change_pct"    : round(change_pct, 2),
        "rsi_14"        : round(rsi_14, 2),
        "rsi_signal"    : rsi_signal,
        "ma_cross"      : ma_cross,
        "macd_status"   : macd_status,
        "bb_signal"     : bb_signal,
        "bullish_count" : bullish,
        "recommendation": recommendation,
        "signals": [
            {
                "name"  : "MA Cross (50/200)",
                "status": ma_cross,
                "value" : f"SMA50={round(sma_50,0)} SMA200={round(sma_200,0)}"
            },
            {
                "name"  : "RSI 14",
                "status": rsi_signal,
                "value" : str(round(rsi_14, 1))
            },
            {
                "name"  : "MACD",
                "status": macd_status,
                "value" : f"{round(macd,1)} vs {round(macd_sig,1)}"
            },
            {
                "name"  : "Bollinger Band",
                "status": bb_signal,
                "value" : f"Position: {round(bb_pos,2)}"
            }
        ]
    }

    # Cache result
    _live_cache[cache_key] = {
        "data"     : result,
        "timestamp": time.time()
    }

    print(f"Live signal for {sector_name}: "
          f"{recommendation} ({bullish}/4 bullish)")
    return result


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """
    Main endpoint — runs GA + PSO and returns results
    Takes ~5 minutes for pop_size=50, generations=40
    """
    cache    = get_data(
        start_date=req.start_date,
        end_date=req.end_date,
        sector=req.sector
    )
    train_df = cache['train_df'].copy()
    test_df  = cache['test_df'].copy()

    sector_name = SECTOR_NAMES.get(req.sector, req.sector)

    # Generate transformer signals if enabled
    if req.use_transformer:
        print(f"Generating transformer signals (window={req.seq_len})...")
        train_df, test_df = generate_transformer_signals(
            train_df, test_df,
            cache['train_scaled'], cache['test_scaled'],
            FEATURE_COLS, seq_len=req.seq_len
        )

    # Run GA
    print(f"Running GA on {sector_name}...")
    ga_chrom, ga_history = run_ga(
        train_df,
        pop_size=req.pop_size,
        generations=req.generations,
        use_transformer=req.use_transformer
    )

    # Run PSO
    print(f"Running PSO on {sector_name}...")
    pso_chrom, pso_history = run_pso(
        train_df,
        n_particles=req.pop_size,
        n_iters=req.generations,
        use_transformer=req.use_transformer
    )

    # Evaluate on test data
    if req.use_transformer:
        ga_result  = backtest_with_signal(test_df, *decode_t(ga_chrom))
        pso_result = backtest_with_signal(test_df, *decode_t(pso_chrom))
    else:
        ga_result  = backtest(test_df, *decode(ga_chrom))
        pso_result = backtest(test_df, *decode(pso_chrom))

    # Buy and Hold baseline
    bh_return = ((test_df['Close'].iloc[-1]
                  - test_df['Close'].iloc[0])
                 / test_df['Close'].iloc[0] * 100)
    bh_port   = [round(100000 * test_df['Close'].iloc[i]
                       / test_df['Close'].iloc[0], 2)
                 for i in range(len(test_df))]

    # Cache for paper trading
    _cache['ga_chrom']        = ga_chrom
    _cache['pso_chrom']       = pso_chrom
    _cache['use_transformer'] = req.use_transformer
    _cache['test_df']         = test_df
    _cache['ready']           = True

    return {
        "ga"          : {**ga_result,  "history": ga_history},
        "pso"         : {**pso_result, "history": pso_history},
        "buy_and_hold": {
            "total_return": round(bh_return, 2),
            "portfolio"   : bh_port
        },
        "dates"      : ga_result['dates'],
        "start_date" : req.start_date,
        "end_date"   : req.end_date,
        "sector"     : req.sector,
        "sector_name": sector_name
    }


@app.post("/api/paper-trade")
def paper_trade(req: PaperTradeRequest):
    """Simulate paper trading with user-defined capital"""
    if 'ga_chrom' not in _cache:
        return {"error": "Please run /api/optimize first"}

    test_df         = _cache['test_df']
    use_transformer = _cache.get('use_transformer', False)
    scale           = req.capital / 100000

    # Select model
    if use_transformer:
        if req.model in ["GA", "GA+Transformer"]:
            result = backtest_with_signal(
                test_df, *decode_t(_cache['ga_chrom']))
        else:
            result = backtest_with_signal(
                test_df, *decode_t(_cache['pso_chrom']))
    else:
        if req.model == "GA":
            result = backtest(test_df, *decode(_cache['ga_chrom']))
        else:
            result = backtest(test_df, *decode(_cache['pso_chrom']))

    return {
        "starting_capital": req.capital,
        "final_value"     : round(result['final_value'] * scale, 2),
        "profit"          : round(
            (result['final_value'] - 100000) * scale, 2),
        "total_return"    : result['total_return'],
        "sharpe"          : result['sharpe'],
        "max_drawdown"    : result['max_drawdown'],
        "num_trades"      : result['num_trades'],
        "trades"          : result['trades'],
        "portfolio"       : [round(v * scale, 2)
                             for v in result['portfolio']],
        "dates"           : result['dates']
    }


@app.post("/api/transformer-window")
def transformer_window(seq_len: int = 60):
    """Regenerate transformer signals with different window size"""
    cache    = get_data()
    train_df = cache['train_df'].copy()
    test_df  = cache['test_df'].copy()

    _, test_df = generate_transformer_signals(
        train_df, test_df,
        cache['train_scaled'], cache['test_scaled'],
        FEATURE_COLS, seq_len=seq_len
    )

    signals = test_df['Transformer_Signal'].tolist()
    dates   = [str(test_df.iloc[i]['Date'].date())
               for i in range(len(test_df))]

    return {
        "seq_len" : seq_len,
        "signals" : [round(s, 4) for s in signals],
        "dates"   : dates,
        "bullish" : sum(1 for s in signals if s > 0.5),
        "bearish" : sum(1 for s in signals if s <= 0.5)
    }