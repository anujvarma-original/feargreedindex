import json
import math
import re
from datetime import date
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# =============================
# Streamlit page config
# =============================
st.set_page_config(page_title="Fear & Greed Strategy", layout="wide")

st.title("Fear & Greed Strategy Dashboard")
st.caption("Streamlit Cloud version with no extra pip requirements file.")


# =============================
# Configuration
# =============================
FRED_API_KEY = st.secrets["api_keys"]["fred"]

CNN_FGI_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CBOE_DAILY_STATS_URL = "https://www.cboe.com/us/options/market_statistics/daily/"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_START_DATE = "2020-08-01"


# =============================
# Sidebar inputs
# =============================
st.sidebar.header("Settings")

start_date = st.sidebar.date_input(
    "Start date",
    value=pd.to_datetime(DEFAULT_START_DATE).date()
)

use_today_as_end = st.sidebar.checkbox("Use today as end date", value=True)

if use_today_as_end:
    end_date = None
else:
    end_date_input = st.sidebar.date_input("End date", value=date.today())
    end_date = end_date_input.isoformat()

run_button = st.sidebar.button("Run Strategy", type="primary")


# =============================
# Helpers
# =============================
def safe_fetch_text(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> str:
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url, headers=headers or HEADERS)
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def safe_fetch_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    text = safe_fetch_text(url, params=params, headers=headers)
    return json.loads(text)


def latest_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan


def min_max_clip(x: pd.Series, low: float = -1.0, high: float = 1.0) -> pd.Series:
    return x.clip(lower=low, upper=high)


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


# =============================
# Data fetchers
# =============================
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, start_date_str: str, end_date_str: Optional[str] = None) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date_str,
    }
    if end_date_str:
        params["observation_end"] = end_date_str

    payload = safe_fetch_json(FRED_SERIES_URL, params=params)
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


@st.cache_data(ttl=3600)
def fetch_cnn_fear_greed(start_date_str: str) -> pd.DataFrame:
    url = f"{CNN_FGI_URL}/{start_date_str}"
    payload = safe_fetch_json(
        url,
        headers={
            **HEADERS,
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com",
        },
    )

    hist = payload.get("fear_and_greed_historical", {}).get("data", [])
    if not hist:
        current = payload.get("fear_and_greed")
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
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Unexpected CNN history format: columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["x"], unit="ms").dt.normalize()
    df["fear_greed_score"] = pd.to_numeric(df["y"], errors="coerce")
    df["fear_greed_rating"] = df["rating"] if "rating" in df.columns else pd.NA

    df = df[["date", "fear_greed_score", "fear_greed_rating"]].drop_duplicates("date")
    return df.set_index("date").sort_index()


@st.cache_data(ttl=3600)
def fetch_cboe_put_call_ratio() -> float:
    html = safe_fetch_text(CBOE_DAILY_STATS_URL, headers=HEADERS)

    patterns = [
        r"TOTAL PUT/CALL RATIO[^0-9]*([0-9]+\.[0-9]+)",
        r"PUT/CALL RATIO[^0-9]*([0-9]+\.[0-9]+)",
    ]

    for pat in patterns:
        m = re.search(pat, html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return float(m.group(1))

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    raise ValueError("Could not parse put/call ratio from Cboe page.")


# =============================
# Strategy pipeline
# =============================
def build_dataset(start_date_str: str, end_date_str: Optional[str]) -> pd.DataFrame:
    fgi = fetch_cnn_fear_greed(start_date_str)

    spx = fetch_fred_series("SP500", start_date_str, end_date_str).rename("sp500")
    vix = fetch_fred_series("VIXCLS", start_date_str, end_date_str).rename("vix")
    hy = fetch_fred_series("BAMLH0A0HYM2", start_date_str, end_date_str).rename("hy_oas")

    df = pd.concat([fgi, spx, vix, hy], axis=1).sort_index()

    df["fear_greed_score"] = df["fear_greed_score"].ffill()
    df["vix"] = df["vix"].ffill()
    df["hy_oas"] = df["hy_oas"].ffill()

    df["sp500_200dma"] = df["sp500"].rolling(200).mean()
    df["trend_up"] = df["sp500"] > df["sp500_200dma"]
    df["sp500_ret"] = df["sp500"].pct_change()

    return df


def add_realtime_put_call_proxy(df: pd.DataFrame) -> pd.DataFrame:
    latest_pcr = fetch_cboe_put_call_ratio()
    latest_vix = latest_non_null(df["vix"])

    if pd.isna(latest_vix) or latest_vix == 0:
        raise ValueError("Cannot construct PCR proxy because latest VIX is missing/zero.")

    df["put_call_ratio_proxy"] = latest_pcr * (df["vix"] / latest_vix).clip(lower=0.5, upper=2.0)
    df["put_call_ratio_live"] = np.nan
    df.loc[df.index.max(), "put_call_ratio_live"] = latest_pcr

    return df


def compute_meo(df: pd.DataFrame) -> pd.DataFrame:
    s1 = (df["fear_greed_score"] - 50.0) / 50.0
    s2 = -(df["vix"] - 20.0) / 20.0
    s3 = -(df["put_call_ratio_proxy"] - 0.70) / 0.30
    s4 = -(df["hy_oas"] - 4.0) / 4.0

    df["S1_fgi"] = min_max_clip(s1)
    df["S2_vix"] = min_max_clip(s2)
    df["S3_pcr"] = min_max_clip(s3)
    df["S4_hy"] = min_max_clip(s4)

    df["MEO"] = df[["S1_fgi", "S2_vix", "S3_pcr", "S4_hy"]].mean(axis=1)
    return df


def generate_positions(df: pd.DataFrame) -> pd.DataFrame:
    pos = np.zeros(len(df), dtype=float)

    strong_long = (df["MEO"] < -0.6) & (df["trend_up"])
    moderate_long = (df["MEO"] >= -0.6) & (df["MEO"] < -0.2) & (df["trend_up"])
    de_risk = df["MEO"] > 0.6

    pos[strong_long.values] = 1.0
    pos[moderate_long.values] = 0.5
    pos[de_risk.values] = 0.0

    df["position"] = pos
    df["position_lag"] = df["position"].shift(1).fillna(0.0)
    return df


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["strategy_ret"] = df["position_lag"] * df["sp500_ret"]
    df["buy_hold_ret"] = df["sp500_ret"]

    df["strategy_curve"] = (1.0 + df["strategy_ret"].fillna(0)).cumprod()
    df["buy_hold_curve"] = (1.0 + df["buy_hold_ret"].fillna(0)).cumprod()
    return df


# =============================
# App runner
# =============================
def run_app():
    start_date_str = start_date.isoformat()

    with st.spinner("Downloading data and running backtest..."):
        df = build_dataset(start_date_str, end_date)
        df = add_realtime_put_call_proxy(df)
        df = compute_meo(df)
        df = generate_positions(df)
        df = backtest(df)

    latest = df.dropna(subset=["sp500"]).iloc[-1]
    strat = performance_stats(df["strategy_ret"])
    bh = performance_stats(df["buy_hold_ret"])

    st.subheader("Latest Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fear & Greed", f"{latest.get('fear_greed_score', np.nan):.2f}")
    c2.metric("VIX", f"{latest.get('vix', np.nan):.2f}")
    c3.metric("HY OAS", f"{latest.get('hy_oas', np.nan):.2f}")
    c4.metric("MEO", f"{latest.get('MEO', np.nan):.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("PCR Proxy", f"{latest.get('put_call_ratio_proxy', np.nan):.2f}")
    c6.metric("Trend Up?", "Yes" if bool(latest.get("trend_up", False)) else "No")
    c7.metric("Next Position", f"{latest.get('position', np.nan):.2f}")

    st.subheader("Performance")
    perf_df = pd.DataFrame(
        {
            "Strategy": strat,
            "Buy & Hold": bh,
        }
    )
    st.dataframe(perf_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Equity Curves")
    st.line_chart(df[["strategy_curve", "buy_hold_curve"]].dropna())

    st.subheader("Market Emotion Oscillator")
    st.line_chart(df[["MEO"]].dropna())

    st.subheader("S&P 500 vs 200 DMA")
    st.line_chart(df[["sp500", "sp500_200dma"]].dropna())

    st.subheader("Recent Data")
    show_cols = [
        "fear_greed_score", "vix", "hy_oas", "put_call_ratio_proxy",
        "MEO", "trend_up", "position", "strategy_curve", "buy_hold_curve"
    ]
    st.dataframe(df[show_cols].tail(50), use_container_width=True)

    csv_data = df.to_csv().encode("utf-8")
    st.download_button(
        "Download full dataset as CSV",
        data=csv_data,
        file_name="fear_greed_strategy_output.csv",
        mime="text/csv",
    )

    st.info(
        "Note: the historical put/call ratio here is still a proxy derived from the latest scraped Cboe value and VIX. "
        "It is not true historical PCR."
    )


if run_button:
    try:
        run_app()
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Set parameters in the sidebar and click **Run Strategy**.")
