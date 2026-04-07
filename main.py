# main.py — FastAPI backend (all endpoints)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data import fetch_data, prepare_data, FEATURE_COLS
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

# In-memory cache — avoids re-downloading data every request
_cache = {}


def get_data():
    """Load and cache Bank Nifty data"""
    if 'ready' not in _cache:
        print("Fetching Bank Nifty data from Yahoo Finance...")
        df = fetch_data()
        (train_df, val_df, test_df,
         train_scaled, val_scaled,
         test_scaled, scaler) = prepare_data(df)
        _cache['train_df']     = train_df
        _cache['val_df']       = val_df
        _cache['test_df']      = test_df
        _cache['train_scaled'] = train_scaled
        _cache['test_scaled']  = test_scaled
        _cache['ready']        = True
        print(f"Data ready — {len(df)} rows loaded")
    return _cache


# ── Request body models ───────────────────────────────────────

class OptimizeRequest(BaseModel):
    pop_size        : Optional[int]  = 50
    generations     : Optional[int]  = 40
    use_transformer : Optional[bool] = True
    seq_len         : Optional[int]  = 60


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


@app.options("/api/optimize")
def options_optimize():
    return {}


@app.options("/api/paper-trade")
def options_paper_trade():
    return {}


@app.options("/api/transformer-window")
def options_transformer_window():
    return {}


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """
    Main endpoint — runs GA + PSO and returns results
    Takes ~5 minutes for pop_size=50, generations=40
    """
    cache    = get_data()
    train_df = cache['train_df'].copy()
    test_df  = cache['test_df'].copy()

    # Generate transformer signals if enabled
    if req.use_transformer:
        print(f"Generating transformer signals (window={req.seq_len})...")
        train_df, test_df = generate_transformer_signals(
            train_df, test_df,
            cache['train_scaled'], cache['test_scaled'],
            FEATURE_COLS, seq_len=req.seq_len
        )

    # Run GA
    print("Running GA...")
    ga_chrom, ga_history = run_ga(
        train_df,
        pop_size=req.pop_size,
        generations=req.generations,
        use_transformer=req.use_transformer
    )

    # Run PSO
    print("Running PSO...")
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

    return {
        "ga"          : {**ga_result,  "history": ga_history},
        "pso"         : {**pso_result, "history": pso_history},
        "buy_and_hold": {
            "total_return": round(bh_return, 2),
            "portfolio"   : bh_port
        },
        "dates": ga_result['dates']
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