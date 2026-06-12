"""
data_pipeline.py
================
Confirmed design decisions (from project discussion):
------------------------------------------------------
1. TARGETS & TIME ALIGNMENT
   - rv_{t-20:t-1} : 20-day rolling vol EXCLUDING t, known at t → Naive baseline + ML features
   - σ_t           : 20-day rolling vol INCLUDING t → ML prediction target + BSM benchmark σ
   - These two differ by exactly one day's information (t-day close)

2. REGIME
   - Each ticker's own rv_{t-20:t-1} vs its median → High_Vol / Low_Vol
   - VIX is NOT used for regime classification

3. VIX / vix_proxy
   - Role: ML feature capturing systemic fear; NOT substituted into BSM as σ
   - Reason: (a) index-level ≠ individual stock, (b) fixed 30-day tenor ≠ dynamic T,
             (c) implied vol ≠ historical vol in BSM context

4. UNIVERSE & MARKET CAP STRATIFICATION
   - S&P 500: 11 GICS sectors × 50 tickers = 7 large-cap + 3 mid-cap per 10
     (35 large-cap + 15 mid-cap per sector, ranked by market-cap descending)
   - Hang Seng Tech: 30 core constituents
   - CSI 300: 30 core constituents
   - Large-cap: lower vol, tests ML marginal contribution where BSM holds better
   - Mid-cap: higher vol, tests ML pricing optimisation in sentiment-sensitive regime

5. SENTIMENT
   - AV API: only top-10 per sector (rate-limit constraint: 500 calls/day free tier)
   - Non-top-10 US tickers: sentiment = NaN, flag = 1
   - CN tickers: AV structurally unsupported for .SS/.SZ → neutral 0.0 + flag = 1
   - All sentiment features are LAGGED (sent_lag1, sent_ma5) → no look-ahead bias

6. FEATURE ENGINEERING (final schema, keep original col name, no rename to md alias)
   date | ticker | region | sector | cap_group |
   close | log_return |
   rv_lag  (rv_{t-20:t-1}, Naive baseline & ML input) |
   sigma_t (rv including t-day, ML target & benchmark σ) |
   rv_5 | rv_60 | har_daily | har_weekly | har_monthly |
   beta | vix_proxy | vix_beta_adj |
   rf_rate | r_lag1 | S_lag1 |
   sentiment_raw | sentiment_flag |
   sent_lag1 | sent_ma5 |
   regime |
   bsm_naive   (BSM using rv_lag  → Naive pricing) |
   bsm_oracle  (BSM using sigma_t → Benchmark pricing, "ex-post informed price") |

7. HAR BASELINE: har three sub-feature saved only, rv_har OLS weight fit in machine_learning.py

8. EVALUATION FRAMEWORK: implement later in machine_learning.py
Fix: ANSS→TYL, remove invalid "shturl." in HSTECH list + auto create dir globally
"""
import argparse
import os
import time
import warnings
import requests
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  GLOBAL CONFIG
# ─────────────────────────────────────────────
START_DATE     = "2018-01-01"
END_DATE       = "2019-12-31"
# KEY: read env first then fallback fixed key (Scheme B)
FRED_API_KEY   = os.getenv("FRED_API_KEY", "926c9e047076f239237763f6a03961da")
AV_API_KEY     = os.getenv("AV_API_KEY", "RXP2APE979NRUVVT")

TRADING_DAYS   = 252
ROLLING_SHORT  = 5
ROLLING_MAIN   = 20    # primary window for rv_lag and sigma_t
ROLLING_LONG   = 60
BETA_WINDOW    = 252   # 1-year rolling beta
HAR_DAILY      = 1
HAR_WEEKLY     = 5
HAR_MONTHLY    = 22

STRIKE         = 110
T_MATURITY     = 1 / 12   # fixed 1-month option

OUTPUT_DIR     = Path("data/extended")
CACHE_DIR      = Path("data/cache")
RAW_DIR        = Path("data/raw")

YF_SLEEP       = 0.4
AV_SLEEP       = 12        # AV free: 5 calls/min, 500/day

# ─────────────────────────────────────────────
#  CACHE UTILITIES
# ─────────────────────────────────────────────
def make_dirs():
    for d in [OUTPUT_DIR, CACHE_DIR, RAW_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# 【关键修复：全局启动立刻创建文件夹，避免目录缺失报错】
make_dirs()

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"

def load_cache(name: str):
    p = cache_path(name)
    if p.exists():
        print(f"  [cache] {name}")
        return pd.read_parquet(p)
    return None

def save_cache(df: pd.DataFrame, name: str):
    df.to_parquet(cache_path(name))

# ─────────────────────────────────────────────
#  UNIVERSE DEFINITIONS (fixed ticker list, no online update | FIXED INVALID SYMBOL)
# ─────────────────────────────────────────────
SP500_UNIVERSE = {
    "Information_Technology": [
        # ── Large-cap (35) ──
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","QCOM","TXN","INTC",
        "IBM","NOW","AMAT","LRCX","MU","ADI","KLAC","MCHP","CDNS","SNPS",
        "FTNT","PANW","TYL","TER","KEYS","EPAM","CTSH","IT","GLW","STX",
        "FFIV","JNPR","NTAP","AKAM","VRSN",
        # ── Mid-cap (15) ──
        "LDOS","SAIC","CACI","DXC","HPE",
        "ZBRA","TRMB","ENPH","SEDG","IPGP",
        "COHU","ONTO","FORM","ACLS","AMBA",
    ],
    "Health_Care": [
        "LLY","JNJ","UNH","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "SYK","ISRG","MDT","ELV","CI","HUM","CVS","MCK","CAH","COR",
        "BSX","EW","BDX","ZBH","HOLX","BAX","RMD","DXCM","ALGN","MTD",
        "HSIC","PDCO","MMSI","NVCR","INSP",
        "OMCL","ACLX","ITGR","HCAT","PRVA",
        "ACAD","RARE","KURA","FOLD","RCKT",
    ],
    "Financials": [
        "JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","MCO",
        "ICE","CME","CB","PGR","TRV","MET","PRU","AFL","ALL","AIG",
        "USB","PNC","TFC","FITB","KEY","CFG","HBAN","RF","MTB","ZION",
        "FHN","SNV","ONB","WTFC","BOKF",
        "IBOC","SFNC","HTLF","CATY","EWBC",
        "GBCI","WAL","BANR","CVBF","FFIN",
    ],
    "Consumer_Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
        "ORLY","AZO","ROST","DHI","LEN","PHM","NVR","TOL","MAR","HLT",
        "GM","F","APTV","BWA","LKQ","AAP","AN","KMX","LAD","PAG",
        "SIG","BOOT","GRMN","POOL","LESL",
        "GPC","CVNA","DRVN","THRM","FOXF",
        "LCII","PATK","SHYF","WGO","THO",
    ],
    "Consumer_Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KHC",
        "GIS","K","CPB","SJM","HRL","MKC","CAG","CHD","CLX","EL",
        "KR","SYY","PFGC","USFD","BJ","GO","CASY","WINN","VLGEA","IMKTA",
        "CENT","LANC","JJSF","HAIN","SMPL",
        "FRPT","MGPI","WDFC","COTT","PRMW",
        "FIZZ","COKE","CELH","MNST","KDP",
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","PXD","OXY",
        "HAL","BKR","DVN","HES","FANG","MRO","APA","CTRA","EQT","RRC",
        "AR","CNX","SM","CHK","NOG","VTLE","MNRL","PHX","FLNG","TELL",
        "CIVI","CPE","REI","SWN","MTDR",
        "ESTE","CRGY","KOS","BORR","NINE",
        "KLXE","RES","PTEN","PUMP","LBRT",
    ],
    "Industrials": [
        "GE","CAT","UNP","HON","UPS","RTX","BA","LMT","NOC","GD",
        "DE","EMR","ETN","ITW","PH","ROK","AME","XYL","IEX","FAST",
        "SWK","IR","TT","CARR","OTIS","WAB","NSC","CSX","FDX","DAL",
        "GNRC","MIDD","BWXT","HXL","TDG",
        "KTOS","AXON","CW","MOOG","ESE",
        "KAMN","HEICO","SPR","WWD","FLIR",
    ],
    "Materials": [
        "LIN","APD","SHW","ECL","FCX","NEM","NUE","STLD","RS","VMC",
        "MLM","ALB","CF","MOS","FMC","PPG","RPM","SON","SEE","PKG",
        "IP","WRK","SLGN","BERY","ATR","GEF","TREX","UFPI","LPX","OSB",
        "IOSP","HWKN","BCPC","MTRN","VNTR",
        "TPC","CSTM","CENX","KALU","HAYN",
        "CRS","ATI","ARNC","CMC","ZEUS",
    ],
    "Real_Estate": [
        "PLD","AMT","EQIX","CCI","SPG","WELL","DLR","PSA","EQR","AVB",
        "O","VTR","VICI","WPC","NNN","STAG","COLD","TRNO","EXR","LSI",
        "MAA","UDR","ESS","CPT","AIV","BRT","NXRT","IRT","ELME","VRE",
        "GMRE","GOOD","LAND","PINE","SAFE",
        "EPRT","ADC","NTST","PSTL","GIPR",
        "FCPT","PLYM","IIPR","HASI","NYMT",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","XEL","SRE","PEG","ED","EXC",
        "ES","AWK","WEC","ETR","FE","EIX","PPL","DTE","CMS","NI",
        "LNT","EVRG","OGE","AEE","CNP","PNW","AVA","NWE","MGEE","SJW",
        "ARTNA","MSEX","YORW","CWCO","GWRS",
        "NOVA","SHLS","ARRY","FSLR","SPWR",
        "RUN","BE","PLUG","AMRC","CLNE",
    ],
    "Communication_Services": [
        "META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS",
        "ATVI","EA","TTWO","RBLX","U","SNAP","PINS","MTCH","ZM","DOCU",
        "FOXA","FOX","NWSA","NWS","IPG","OMC","WBD","PARA","LYV","IACI",
        "CABO","LUMN","ATUS","WOW","CNSL",
        "SHEN","OOMA","BAND","LPSN","AVYA",
        "RNG","MSGN","AMC","CNK","IMAX",
    ],
}

# Fix: remove invalid "shturl." dummy text
HSTECH_UNIVERSE = [
    "0700.HK","9988.HK","3690.HK","9999.HK","1024.HK",
    "9618.HK","2382.HK","0992.HK","6098.HK","0241.HK",
    "1810.HK","2269.HK","9868.HK","9866.HK","2015.HK",
    "9698.HK","3888.HK","6060.HK","0268.HK","1347.HK",
    "2518.HK","9626.HK","1357.HK","9961.HK","6862.HK",
    "6690.HK","0020.HK","2359.HK","9961.HK","9987.HK",
]

CSI300_UNIVERSE = [
    "601318.SS","600519.SS","601166.SS","600036.SS","000858.SZ",
    "601288.SS","601398.SS","600900.SS","000002.SZ","600276.SS",
    "002594.SZ","000651.SZ","000333.SZ","600887.SS","002415.SZ",
    "601628.SS","601601.SS","000568.SZ","600030.SS","601688.SS",
    "600009.SS","601919.SS","600048.SS","600016.SS","601186.SS",
    "600050.SS","601088.SS","600104.SS","600028.SS","601857.SS",
]

# ─────────────────────────────────────────────
#  BSM PRICER (vectorised, NaN-safe)
# ─────────────────────────────────────────────
def black_scholes_vec(S, K, T, r, sigma):
    from scipy.stats import norm
    S     = np.asarray(S,     dtype=float)
    r     = np.asarray(r,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    invalid = (sigma <= 1e-6) | (T <= 0) | np.isnan(S) | np.isnan(r) | np.isnan(sigma)
    ss = np.where(invalid, 1.0, sigma)
    d1 = (np.log(S / K) + (r + 0.5 * ss**2) * T) / (ss * np.sqrt(T))
    d2 = d1 - ss * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return np.where(invalid, np.nan, price)

# ─────────────────────────────────────────────
#  MARKET CAP STRATIFICATION: 35 large /15 mid, missing cap use list order
# ─────────────────────────────────────────────
def get_cap_group(tickers: list) -> dict:
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap", None)
            caps[t] = mc if mc else 0
            time.sleep(0.2)
        except Exception:
            caps[t] = 0
    ranked = sorted(tickers, key=lambda x: caps.get(x, 0), reverse=True)
    groups = {}
    for i, t in enumerate(ranked):
        groups[t] = "large" if i < 35 else "mid"
    return groups

# ─────────────────────────────────────────────
#  PRICE DOWNLOADER
# ─────────────────────────────────────────────
def download_prices(tickers: list, label: str) -> pd.DataFrame:
    cached = load_cache(f"px_{label}")
    if cached is not None:
        return cached
    frames = []
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = yf.download(
                ticker, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                print(f"  [{i}/{len(tickers)}] {ticker} — no data")
                continue
            df = raw[["Close"]].copy()
            df.columns = ["close"]
            df.index.name = "date"
            df["ticker"] = ticker
            df.reset_index(inplace=True)
            frames.append(df)
            print(f"  [{i}/{len(tickers)}] {ticker} ✓ ({len(df)} rows)")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {ticker} — ERROR: {e}")
        time.sleep(YF_SLEEP)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    save_cache(out, f"px_{label}")
    return out

# ─────────────────────────────────────────────
#  SPY REFERENCE PRICES  (for Beta calculation)
# ─────────────────────────────────────────────
def download_spy() -> pd.Series:
    cached = load_cache("spy_close")
    if cached is not None:
        return cached.set_index("date")["close"]
    raw = yf.download("SPY", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    df = raw[["Close"]].copy()
    df.columns = ["close"]
    df.index.name = "date"
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    save_cache(df, "spy_close")
    return df.set_index("date")["close"]

# ─────────────────────────────────────────────
#  VIX
# ─────────────────────────────────────────────
def download_vix() -> pd.DataFrame:
    cached = load_cache("vix")
    if cached is not None:
        return cached
    raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    df = raw[["Close"]].copy()
    df.columns = ["vix_proxy"]
    df.index.name = "date"
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    save_cache(df, "vix")
    return df

# ─────────────────────────────────────────────
#  RISK-FREE RATES
# ─────────────────────────────────────────────
def download_rf_us() -> pd.DataFrame:
    cached = load_cache("rf_us")
    if cached is not None:
        return cached
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series("DGS10", observation_start=START_DATE, observation_end=END_DATE)
    df = s.to_frame(name="rf_rate")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.reset_index(inplace=True)
    df["rf_rate"] = df["rf_rate"] / 100
    df.dropna(subset=["rf_rate"], inplace=True)
    save_cache(df, "rf_us")
    return df

def download_rf_hk() -> pd.DataFrame:
    cached = load_cache("rf_hk")
    if cached is not None:
        return cached
    try:
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series("HKIR3TED", observation_start=START_DATE, observation_end=END_DATE)
        df = s.to_frame(name="rf_rate")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.reset_index(inplace=True)
        df["rf_rate"] = df["rf_rate"] / 100
    except Exception:
        print("  HIBOR unavailable — using 2% constant proxy for HK rf_rate")
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
        df = pd.DataFrame({"date": dates, "rf_rate": 0.02})
    save_cache(df, "rf_hk")
    return df

def download_rf_cn() -> pd.DataFrame:
    cached = load_cache("rf_cn")
    if cached is not None:
        return cached
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    df = pd.DataFrame({"date": dates, "rf_rate": 0.0435})  # LPR 1Y 2018-2019
    save_cache(df, "rf_cn")
    return df

# ─────────────────────────────────────────────
#  SENTIMENT  (Alpha Vantage)
# ─────────────────────────────────────────────
def _av_fetch_ticker(ticker: str) -> pd.DataFrame:
    all_articles = []
    start   = datetime.strptime(START_DATE, "%Y-%m-%d")
    end     = datetime.strptime(END_DATE,   "%Y-%m-%d")
    current = start
    while current <= end:
        next_m  = datetime(current.year + (current.month == 12),
                           (current.month % 12) + 1, 1)
        period_end = next_m - timedelta(days=1)
        url = (
            "https://www.alphavantage.co/query?"
            "function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&time_from={current.strftime('%Y%m%dT%H%M')}"
            f"&time_to={period_end.strftime('%Y%m%dT%H%M')}"
            f"&apikey={AV_API_KEY}&limit=1000"
        )
        try:
            resp = requests.get(url, timeout=15)
            all_articles.extend(resp.json().get("feed", []))
        except Exception as e:
            print(f"    AV error {ticker} {current:%Y-%m}: {e}")
        time.sleep(AV_SLEEP)
        current = next_m
    rows = []
    for art in all_articles:
        try:
            rows.append({
                "date":      pd.to_datetime(art["time_published"][:8], format="%Y%m%d"),
                "sentiment": float(art["overall_sentiment_score"]),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "sentiment"])
    df = pd.DataFrame(rows).groupby("date")["sentiment"].mean().reset_index()
    df["ticker"] = ticker
    return df

def download_sentiment_tickers(tickers: list, label: str) -> pd.DataFrame:
    cached = load_cache(f"sent_{label}")
    if cached is not None:
        return cached
    frames = []
    for i, t in enumerate(tickers, 1):
        key = f"sent_single_{t.replace('.', '_')}"
        c = load_cache(key)
        if c is not None:
            frames.append(c)
            continue
        print(f"  [{i}/{len(tickers)}] AV sentiment: {t}")
        df = _av_fetch_ticker(t)
        if not df.empty:
            save_cache(df, key)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "sentiment"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    save_cache(out, f"sent_{label}")
    return out

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING  (per-ticker)
# ─────────────────────────────────────────────
def build_features_single(
    price_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    spy_ret: pd.Series,
    sentiment_df: pd.DataFrame,
    ticker: str,
    sector: str,
    region: str,
    cap_group: str,
    sentiment_flag_default: int = 0,
) -> pd.DataFrame:
    df = price_df[price_df["ticker"] == ticker].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    # log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # rv_lag: shift first then rolling, exclude t
    df["rv_lag"] = df["log_return"].shift(1).rolling(ROLLING_MAIN).std() * np.sqrt(TRADING_DAYS)
    # sigma_t no shift, include t
    df["sigma_t"] = df["log_return"].rolling(ROLLING_MAIN).std() * np.sqrt(TRADING_DAYS)
    # short/long lagged vol
    df["rv_5"] = df["log_return"].shift(1).rolling(ROLLING_SHORT).std() * np.sqrt(TRADING_DAYS)
    df["rv_60"] = df["log_return"].shift(1).rolling(ROLLING_LONG).std() * np.sqrt(TRADING_DAYS)
    # HAR three sub features only, no rv_har combine here
    df["har_daily"]   = df["log_return"].shift(1).abs() * np.sqrt(TRADING_DAYS)
    df["har_weekly"]  = df["log_return"].shift(1).rolling(HAR_WEEKLY).std() * np.sqrt(TRADING_DAYS)
    df["har_monthly"] = df["log_return"].shift(1).rolling(HAR_MONTHLY).std() * np.sqrt(TRADING_DAYS)
    # rolling beta vs SPY
    spy_aligned = spy_ret.reindex(df["date"].values)
    spy_rets_arr = spy_aligned.values
    betas = np.full(len(df), np.nan)
    for i in range(BETA_WINDOW, len(df)):
        y = df["log_return"].values[i - BETA_WINDOW: i]
        x = spy_rets_arr[i - BETA_WINDOW: i]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 30:
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            betas[i] = slope
    df["beta"] = betas
    # vix merge
    vix_df_c = vix_df.copy()
    vix_df_c["date"] = pd.to_datetime(vix_df_c["date"])
    df = df.merge(vix_df_c[["date", "vix_proxy"]], on="date", how="left")
    df["vix_proxy"] = df["vix_proxy"].ffill()
    df["vix_beta_adj"] = df["beta"] * df["vix_proxy"] / 100
    # rf merge
    rf_df_c = rf_df.copy()
    rf_df_c["date"] = pd.to_datetime(rf_df_c["date"])
    df = df.merge(rf_df_c[["date", "rf_rate"]], on="date", how="left")
    df["rf_rate"] = df["rf_rate"].ffill()
    # lag S & r
    df["S_lag1"] = df["close"].shift(1)
    df["r_lag1"] = df["rf_rate"].shift(1)
    # sentiment fill rule
    if not sentiment_df.empty and ticker in sentiment_df["ticker"].values:
        sent_tick = sentiment_df[sentiment_df["ticker"] == ticker][["date", "sentiment"]].copy()
        sent_tick["date"] = pd.to_datetime(sent_tick["date"])
        df = df.merge(sent_tick, on="date", how="left")
        df["sentiment_flag"] = sentiment_flag_default
    else:
        df["sentiment"] = 0.0 if region == "CN" else np.nan
        df["sentiment_flag"] = 1
    df["sent_lag1"] = df["sentiment"].shift(1)
    df["sent_ma5"]  = df["sentiment"].shift(1).rolling(5).mean()
    # meta info
    df["sector"]    = sector
    df["region"]    = region
    df["cap_group"] = cap_group
    # regime: only rv_lag median, no VIX
    df.dropna(subset=["rv_lag", "sigma_t", "log_return"], inplace=True)
    if df.empty:
        return pd.DataFrame()
    median_rv = df["rv_lag"].median()
    df["regime"] = np.where(df["rv_lag"] >= median_rv, "High_Vol", "Low_Vol")
    # BSM pricing
    S  = df["S_lag1"].values
    r  = df["r_lag1"].values
    df["bsm_naive"]  = black_scholes_vec(S, STRIKE, T_MATURITY, r, df["rv_lag"].values)
    df["bsm_oracle"] = black_scholes_vec(S, STRIKE, T_MATURITY, r, df["sigma_t"].values)
    # final column order
    col_order = [
        "date", "ticker", "region", "sector", "cap_group",
        "close", "log_return",
        "rv_lag", "sigma_t",
        "rv_5", "rv_60",
        "har_daily", "har_weekly", "har_monthly",
        "beta", "vix_proxy", "vix_beta_adj",
        "rf_rate", "r_lag1", "S_lag1",
        "sentiment", "sentiment_flag",
        "sent_lag1", "sent_ma5",
        "regime",
        "bsm_naive", "bsm_oracle",
    ]
    existing = [c for c in col_order if c in df.columns]
    return df[existing].reset_index(drop=True)

# ─────────────────────────────────────────────
#  REGION PIPELINES
# ─────────────────────────────────────────────
def run_jpm_pipeline() -> pd.DataFrame:
    print("\n" + "="*60)
    print("  JPM SINGLE-TICKER PIPELINE")
    print("="*60)
    prices    = download_prices(["JPM"], "jpm")
    vix       = download_vix()
    rf        = download_rf_us()
    spy_close = download_spy()
    spy_ret   = np.log(spy_close / spy_close.shift(1))
    top10_sent = download_sentiment_tickers(["JPM"], "jpm")
    df = build_features_single(
        price_df=prices, rf_df=rf, vix_df=vix,
        spy_ret=spy_ret, sentiment_df=top10_sent,
        ticker="JPM", sector="Financials",
        region="US", cap_group="large",
    )
    out_parq = OUTPUT_DIR / "jpm_processed.parquet"
    out_csv  = OUTPUT_DIR / "jpm_processed.csv"
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)
    print(f"JPM saved: {len(df)} rows → {out_parq}, {out_csv}")
    return df

def run_us_pipeline() -> pd.DataFrame:
    print("\n" + "="*60)
    print("  US PIPELINE — S&P 500 (11 GICS sectors)")
    print("="*60)
    vix       = download_vix()
    rf        = download_rf_us()
    spy_close = download_spy()
    spy_ret   = np.log(spy_close / spy_close.shift(1))
    all_panels = []
    for sector, tickers in SP500_UNIVERSE.items():
        print(f"\n── Sector: {sector}")
        cap_groups = get_cap_group(tickers)
        prices = download_prices(tickers, f"us_{sector}")
        top10    = tickers[:10]
        sent_top = download_sentiment_tickers(top10, f"us_{sector}_top10")
        sector_frames = []
        for ticker in tickers:
            cap = cap_groups.get(ticker, "large")
            in_top10 = ticker in top10
            flag_default = 0 if in_top10 else 1
            df_t = build_features_single(
                price_df=prices, rf_df=rf, vix_df=vix,
                spy_ret=spy_ret, sentiment_df=sent_top,
                ticker=ticker, sector=sector,
                region="US", cap_group=cap,
                sentiment_flag_default=flag_default,
            )
            if not df_t.empty:
                sector_frames.append(df_t)
        if sector_frames:
            sector_panel = pd.concat(sector_frames, ignore_index=True)
            parq_p = OUTPUT_DIR / f"us_{sector}.parquet"
            csv_p  = OUTPUT_DIR / f"us_{sector}.csv"
            sector_panel.to_parquet(parq_p, index=False)
            sector_panel.to_csv(csv_p, index=False)
            print(f"  Saved → {parq_p}, {csv_p} ({len(sector_panel):,} rows)")
            all_panels.append(sector_panel)
    us_panel = pd.concat(all_panels, ignore_index=True)
    us_panel.to_parquet(OUTPUT_DIR / "us_all.parquet", index=False)
    us_panel.to_csv(OUTPUT_DIR / "us_all.csv", index=False)
    print(f"\nUS total: {len(us_panel):,} rows, {us_panel['ticker'].nunique()} tickers")
    return us_panel

def run_hk_pipeline() -> pd.DataFrame:
    print("\n" + "="*60)
    print("  HK PIPELINE — Hang Seng Tech (30)")
    print("="*60)
    prices    = download_prices(HSTECH_UNIVERSE, "hstech")
    vix       = download_vix()
    rf        = download_rf_hk()
    spy_close = download_spy()
    spy_ret   = np.log(spy_close / spy_close.shift(1))
    sent = download_sentiment_tickers(HSTECH_UNIVERSE, "hstech")
    frames = []
    for ticker in HSTECH_UNIVERSE:
        df_t = build_features_single(
            price_df=prices, rf_df=rf, vix_df=vix,
            spy_ret=spy_ret, sentiment_df=sent,
            ticker=ticker, sector="HStech",
            region="HK", cap_group="large",
        )
        if not df_t.empty:
            frames.append(df_t)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panel.to_parquet(OUTPUT_DIR / "hk_hstech.parquet", index=False)
    panel.to_csv(OUTPUT_DIR / "hk_hstech.csv", index=False)
    print(f"HK total: {len(panel):,} rows")
    return panel

def run_cn_pipeline() -> pd.DataFrame:
    print("\n" + "="*60)
    print("  CN PIPELINE — CSI 300 core (30)")
    print("="*60)
    prices    = download_prices(CSI300_UNIVERSE, "csi300")
    vix       = download_vix()
    rf        = download_rf_cn()
    spy_close = download_spy()
    spy_ret   = np.log(spy_close / spy_close.shift(1))
    frames = []
    empty_sent = pd.DataFrame()
    for ticker in CSI300_UNIVERSE:
        df_t = build_features_single(
            price_df=prices, rf_df=rf, vix_df=vix,
            spy_ret=spy_ret,
            sentiment_df=empty_sent,
            ticker=ticker, sector="CSI300",
            region="CN", cap_group="large",
            sentiment_flag_default=1,
        )
        if not df_t.empty:
            frames.append(df_t)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panel.to_parquet(OUTPUT_DIR / "cn_csi300.parquet", index=False)
    panel.to_csv(OUTPUT_DIR / "cn_csi300.csv", index=False)
    print(f"CN total: {len(panel):,} rows")
    return panel

# ─────────────────────────────────────────────
#  MASTER MERGE + SUMMARY (parquet+csv both saved)
# ─────────────────────────────────────────────
def merge_all(panels: list) -> pd.DataFrame:
    master = pd.concat([p for p in panels if not p.empty], ignore_index=True)
    master.sort_values(["region", "ticker", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)
    master.to_parquet(OUTPUT_DIR / "master_universe.parquet", index=False)
    master.to_csv(OUTPUT_DIR / "master_universe.csv", index=False)
    print("\n" + "="*60)
    print("  MASTER UNIVERSE COMPLETE")
    print(f"  Total rows    : {len(master):,}")
    print(f"  Tickers       : {master['ticker'].nunique()}")
    print(f"  Date range    : {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"  Regions       : {master['region'].value_counts().to_dict()}")
    print(f"  Cap groups    : {master['cap_group'].value_counts().to_dict()}")
    print(f"  Regimes       : {master['regime'].value_counts().to_dict()}")
    sent_cov = (master['sentiment_flag'] == 0).mean()
    print(f"  Sentiment cov : {sent_cov:.1%} of rows have AV sentiment")
    nan_cols = ["rv_lag","sigma_t","sent_lag1","beta","bsm_naive","bsm_oracle"]
    for c in nan_cols:
        if c in master.columns:
            print(f"    {c:20s} : {master[c].isna().sum():>6} NaN ({master[c].isna().mean():.1%})")
    print("="*60)
    return master

# ─────────────────────────────────────────────
#  ENTRYPOINT: DEFAULT RUN ALL WITHOUT ARGS
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Data Pipeline — Part 1")
    parser.add_argument(
        "--region",
        choices=["jpm", "us", "hk", "cn", "all"],
        default="all", # default=all, run full universe directly
        help="Which universe to process(default=all)",
    )
    args = parser.parse_args()
    panels = []
    if args.region == "jpm":
        panels.append(run_jpm_pipeline())
    elif args.region == "us":
        panels.append(run_us_pipeline())
    elif args.region == "hk":
        panels.append(run_hk_pipeline())
    elif args.region == "cn":
        panels.append(run_cn_pipeline())
    elif args.region == "all":
        panels.append(run_jpm_pipeline())
        panels.append(run_us_pipeline())
        panels.append(run_hk_pipeline())
        panels.append(run_cn_pipeline())
        merge_all(panels)
    print("\n Part 1 data pipeline complete.")

if __name__ == "__main__":
    main()