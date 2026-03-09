#!/usr/bin/env python3
"""
fear_greed_strategy.py

Builds a simple stock-market trading signal from:
- CNN Fear & Greed Index (unofficial public JSON endpoint)
- VIX (FRED: VIXCLS)
- S&P 500 (FRED: SP500)
- High Yield OAS (FRED: BAMLH0A0HYM2)
- Cboe Put/Call Ratio (scraped from Cboe Daily Market Statistics page)

Strategy idea:
- Construct a Market Emotion Oscillator (MEO)
- Use MEO + S&P 500 trend filter to create positions
- Backtest on SPY proxy using S&P 500 daily returns

Requirements:
    pip install requests pandas numpy matplotlib beautifulsoup4 lxml python-dateutil

Environment:
    export FRED_API_KEY="your_fred_key"

Run:
    python fear_greed_strategy.py
"""

from __future__ import annotations

import os
import re
import sys
import math
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# -----------------------------
# Configuration
# -----------------------------

FRED_API_KEY = st.secrets["api_keys"]["fred"]

CNN_FGI_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CBOE_DAILY_STATS_URL = "https://www.cboe.com/us/options/market_statistics/daily/"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"

START_DATE = "2020-08-01"
END_DATE = None  # None = today
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


# -----------------------------
# Helpers
# -----------------------------

def require_fred_key() -> None:
    if not FRED_API_KEY:
        raise RuntimeError(
            "Missing FRED_API_KEY environment variable.\n"
            "Get one from FRED, then set:\n"
            '  export FRED_API_KEY="your_key_here"'
        )


def safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        headers=headers or HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def latest_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan


def min_max_clip(x: pd.Series, low: float = -1.0, high: float = 1.0) -> pd.Series:
    return x.clip(lower=low, upper=high)


def zscore_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z


# -----------------------------
# FRED API
# -----------------------------

def fetch_fred_series(series_id: str, start_date: str, end_date: Optional[str] = None) -> pd.Series:
    """
    Download a daily/series history from FRED.
    """
    require_fred_key()

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }
    if end_date:
        params["observation_end"] = end_date

    resp = safe_get(FRED_SERIES_URL, params=params)
    payload = resp.json()

    obs = payload.get("observations", [])
    if not obs:
        raise ValueError(f"No observations returned for FRED series {series_id}")

    rows = []
    for item in obs:
        val = item.get("value")
        if val == ".":
            val = np.nan
        else:
            try:
                val = float(val)
            except Exception:
                val = np.nan
        rows.append((pd.to_datetime(item["date"]), val))

    s = pd.Series(dict(rows), name=series_id).sort_index()
    s.index.name = "date"
    return s


# -----------------------------
# CNN Fear & Greed (unofficial)
# -----------------------------

def fetch_cnn_fear_greed(start_date: str = START_DATE) -> pd.DataFrame:
    """
    Pull historical stock-market Fear & Greed data from CNN's unofficial endpoint.
    """
    url = f"{CNN_FGI_URL}/{start_date}"
    resp = safe_get(
        url,
        headers={
            **HEADERS,
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com",
        },
    )
    data = resp.json()

    hist = data.get("fear_and_greed_historical", {}).get("data", [])
    if not hist:
        # Fallback: some payloads expose only current snapshot
        current = data.get("fear_and_greed")
        if not current:
            raise ValueError("CNN Fear & Greed payload did not contain expected fields.")
        ts = pd.to_datetime(current["timestamp"], unit="ms")
        df = pd.DataFrame(
            [{
                "date": ts.normalize(),
                "fear_greed_score": float(current["score"]),
                "fear_greed_rating": current.get("rating"),
            }]
        )
        return df.set_index("date").sort_index()

    df = pd.DataFrame(hist)
    # Expected keys often include x (unix ms), y (score), rating
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Unexpected CNN history format: columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["x"], unit="ms").dt.normalize()
    df["fear_greed_score"] = pd.to_numeric(df["y"], errors="coerce")
    if "rating" in df.columns:
        df["fear_greed_rating"] = df["rating"]
    else:
        df["fear_greed_rating"] = pd.NA

    df = df[["date", "fear_greed_score", "fear_greed_rating"]].drop_duplicates("date")
    return df.set_index("date").sort_index()


# -----------------------------
# Cboe Put/Call Ratio scrape
# -----------------------------

def fetch_cboe_put_call_ratio() -> float:
    """
    Scrape today's/most recent total put/call ratio from Cboe Daily Market Statistics page.

    This page is not a formal JSON API, so parsing may need occasional maintenance.
    """
    resp = safe_get(CBOE_DAILY_STATS_URL, headers=HEADERS)
    html = resp.text

    # Fast regex attempt
    patterns = [
        r"TOTAL PUT/CALL RATIO[^0-9]*([0-9]+\.[0-9]+)",
        r"PUT/CALL RATIO[^0-9]*([0-9]+\.[0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    # BeautifulSoup fallback
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    raise ValueError("Could not parse put/call ratio from Cboe daily statistics page.")


# -----------------------------
# Build indicator dataset
# -----------------------------

def build_dataset() -> pd.DataFrame:
    print("Downloading CNN Fear & Greed history...")
    fgi = fetch_cnn_fear_greed(START_DATE)

    print("Downloading FRED series...")
    spx = fetch_fred_series("SP500", START_DATE, END_DATE).rename("sp500")
    vix = fetch_fred_series("VIXCLS", START_DATE, END_DATE).rename("vix")
    hy = fetch_fred_series("BAMLH0A0HYM2", START_DATE, END_DATE).rename("hy_oas")

    df = pd.concat([fgi, spx, vix, hy], axis=1).sort_index()

    # Forward-fill slow-moving series where appropriate
    df["fear_greed_score"] = df["fear_greed_score"].ffill()
    df["vix"] = df["vix"].ffill()
    df["hy_oas"] = df["hy_oas"].ffill()

    # Trend filter
    df["sp500_200dma"] = df["sp500"].rolling(200).mean()
    df["trend_up"] = df["sp500"] > df["sp500_200dma"]

    # Returns for proxy backtest
    df["sp500_ret"] = df["sp500"].pct_change()

    return df


# -----------------------------
# Signal engineering
# -----------------------------

def add_realtime_put_call_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    We only have the current/latest put/call ratio from the page scrape.
    For a truly historical backtest you would need a historical PCR source.
    Here we:
      1) store the latest value
      2) create a simple proxy series using VIX regime scaling
    """
    latest_pcr = fetch_cboe_put_call_ratio()
    print(f"Latest Cboe put/call ratio: {latest_pcr:.3f}")

    # Crude historical proxy:
    # anchor at latest PCR, then scale with relative VIX move.
    # This is NOT the same as true historical PCR. It is only a placeholder.
    latest_vix = latest_non_null(df["vix"])
    if pd.isna(latest_vix) or latest_vix == 0:
        raise ValueError("Cannot construct PCR proxy because latest VIX is missing/zero.")

    df["put_call_ratio_proxy"] = latest_pcr * (df["vix"] / latest_vix).clip(lower=0.5, upper=2.0)
    df["put_call_ratio_live"] = np.nan
    df.loc[df.index.max(), "put_call_ratio_live"] = latest_pcr

    return df


def compute_meo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market Emotion Oscillator:
      S1 = normalized Fear & Greed
      S2 = normalized VIX stress
      S3 = normalized Put/Call fear
      S4 = normalized High Yield spread stress
      MEO = average(S1,S2,S3,S4)

    All components clipped to [-1, 1]
    """
    # S1: Fear & Greed 0..100 -> -1..+1
    s1 = (df["fear_greed_score"] - 50.0) / 50.0

    # S2: VIX, centered around ~20
    s2 = -(df["vix"] - 20.0) / 20.0

    # S3: PCR proxy, centered around ~0.70
    s3 = -(df["put_call_ratio_proxy"] - 0.70) / 0.30

    # S4: HY OAS, centered around ~4.0
    s4 = -(df["hy_oas"] - 4.0) / 4.0

    df["S1_fgi"] = min_max_clip(s1)
    df["S2_vix"] = min_max_clip(s2)
    df["S3_pcr"] = min_max_clip(s3)
    df["S4_hy"] = min_max_clip(s4)

    df["MEO"] = df[["S1_fgi", "S2_vix", "S3_pcr", "S4_hy"]].mean(axis=1)

    return df


def generate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Position logic:
      - Strong long when MEO < -0.6 and trend filter is positive
      - Moderate long when -0.6 <= MEO < -0.2 and trend filter positive
      - Neutral otherwise
      - Optional de-risk when MEO > 0.6
    """
    pos = np.zeros(len(df), dtype=float)

    strong_long = (df["MEO"] < -0.6) & (df["trend_up"])
    moderate_long = (df["MEO"] >= -0.6) & (df["MEO"] < -0.2) & (df["trend_up"])
    de_risk = df["MEO"] > 0.6

    pos[strong_long.values] = 1.0
    pos[moderate_long.values] = 0.5
    pos[de_risk.values] = 0.0

    df["position"] = pos
    # Use yesterday's signal for today's return
    df["position_lag"] = df["position"].shift(1).fillna(0.0)

    return df


# -----------------------------
# Backtest
# -----------------------------

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["strategy_ret"] = df["position_lag"] * df["sp500_ret"]
    df["buy_hold_ret"] = df["sp500_ret"]

    df["strategy_curve"] = (1.0 + df["strategy_ret"].fillna(0)).cumprod()
    df["buy_hold_curve"] = (1.0 + df["buy_hold_ret"].fillna(0)).cumprod()

    return df


def performance_stats(ret: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {}

    total_return = float((1.0 + ret).prod() - 1.0)
    ann_return = float((1.0 + total_return) ** (periods_per_year / max(len(ret), 1)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else np.nan

    curve = (1.0 + ret).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

    hit_rate = float((ret > 0).mean())

    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
    }


# -----------------------------
# Plotting / reporting
# -----------------------------

def plot_results(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    df[["strategy_curve", "buy_hold_curve"]].dropna().plot(ax=ax1)
    ax1.set_title("Strategy vs Buy & Hold")
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True)

    ax2 = plt.subplot(3, 1, 2)
    df["MEO"].plot(ax=ax2)
    ax2.axhline(-0.6, linestyle="--")
    ax2.axhline(-0.2, linestyle="--")
    ax2.axhline(0.6, linestyle="--")
    ax2.set_title("Market Emotion Oscillator (MEO)")
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3)
    df["sp500"].plot(ax=ax3, label="S&P 500")
    df["sp500_200dma"].plot(ax=ax3, label="200 DMA")
    ax3.set_title("S&P 500 Trend Filter")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def print_latest_snapshot(df: pd.DataFrame) -> None:
    latest = df.dropna(subset=["sp500"]).iloc[-1]
    print("\nLatest snapshot")
    print("-" * 60)
    print(f"Date:                  {latest.name.date()}")
    print(f"Fear & Greed score:    {latest.get('fear_greed_score', np.nan):.2f}")
    print(f"VIX:                   {latest.get('vix', np.nan):.2f}")
    print(f"HY OAS:                {latest.get('hy_oas', np.nan):.2f}")
    print(f"PCR proxy:             {latest.get('put_call_ratio_proxy', np.nan):.2f}")
    print(f"MEO:                   {latest.get('MEO', np.nan):.3f}")
    print(f"Trend up?              {bool(latest.get('trend_up', False))}")
    print(f"Position for next day: {latest.get('position', np.nan):.2f}")


def print_stats(df: pd.DataFrame) -> None:
    strat = performance_stats(df["strategy_ret"])
    bh = performance_stats(df["buy_hold_ret"])

    print("\nPerformance")
    print("-" * 60)
    print("Strategy")
    for k, v in strat.items():
        print(f"  {k:18s}: {v: .4f}")

    print("\nBuy & Hold")
    for k, v in bh.items():
        print(f"  {k:18s}: {v: .4f}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    try:
        df = build_dataset()
        df = add_realtime_put_call_proxy(df)
        df = compute_meo(df)
        df = generate_positions(df)
        df = backtest(df)

        print_latest_snapshot(df)
        print_stats(df)
        plot_results(df)

        # Save for inspection
        out = "fear_greed_strategy_output.csv"
        df.to_csv(out)
        print(f"\nSaved full dataset to: {out}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
