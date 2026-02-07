"""
HFT Algo Scanner - Multi-Strike Analysis
Standalone Streamlit app displaying only the HFT Algo Scanner chart.
"""

import requests
import pandas as pd
import time
import streamlit as st
import os
from datetime import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import deque
from requests.exceptions import RequestException

# Config
IST = pytz.timezone("Asia/Kolkata")

CLIENT_ID = '1100244268'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcwNTI5OTMzLCJpYXQiOjE3NzA0NDM1MzMsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.leZEXQjNx2wlZHbGHre0OKfuvA8qqS22oHUAA4HBhavFyjMbbesqX3CE6FC94LsQCcfaPMfIIBYqAlgagmFzKg'

HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}


def get_expiry_dates():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()['data']
    except RequestException as e:
        st.error(f"Connection error: {e}")
        return []


def fetch_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    response = requests.post(url, json=payload, headers=HEADERS)
    time.sleep(3)
    if response.status_code != 200:
        st.error(f"Failed to fetch option chain")
        return None
    return response.json()


def score_option_sentiment(row):
    score = 0
    if row['OI_Change'] > 0: score += 1
    elif row['OI_Change'] < 0: score -= 1
    if row['LTP_Change'] > 0: score += 1
    elif row['LTP_Change'] < 0: score -= 1
    if row['IV_Change'] > 0: score += 1
    elif row['IV_Change'] < 0: score -= 1
    if row['Theta'] < 0: score += 1
    elif row['Theta'] > 0: score -= 1
    if row['Vega'] > 0: score += 1
    elif row['Vega'] < 0: score -= 1
    if row['IV_Change'] > 0 and row['LTP_Change'] < 0: score -= 1
    elif row['IV_Change'] > 0 and row['LTP_Change'] > 0: score += 1

    if score >= 3: bias = "Aggressive Buying"
    elif score >= 1: bias = "Mild Buying"
    elif score == 0: bias = "Neutral"
    elif score <= -3: bias = "Aggressive Writing"
    else: bias = "Mild Writing"

    return pd.Series([score, bias], index=['SentimentScore', 'SentimentBias'])


def build_strike_sentiment_log(option_chain):
    """Build/update strike_sentiment_log from option chain and return log_df for chart."""
    if "previous_data" not in st.session_state:
        st.session_state.previous_data = {}
    if "rolling_data" not in st.session_state:
        st.session_state.rolling_data = {}
    if "strike_sentiment_log" not in st.session_state:
        try:
            if os.path.exists("sentiment_log_backup.csv"):
                st.session_state.strike_sentiment_log = pd.read_csv("sentiment_log_backup.csv").to_dict("records")
            else:
                st.session_state.strike_sentiment_log = []
        except:
            st.session_state.strike_sentiment_log = []

    previous_data = st.session_state.previous_data
    rolling_data = st.session_state.rolling_data

    if not option_chain or "data" not in option_chain or "oc" not in option_chain["data"]:
        return pd.DataFrame()

    option_chain_data = option_chain["data"]["oc"]
    underlying_price = option_chain["data"]["last_price"]
    atm_strike = float(min(option_chain_data.keys(), key=lambda x: abs(float(x) - underlying_price)))
    min_strike = atm_strike - 5 * 50
    max_strike = atm_strike + 5 * 50

    data_list = []
    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)
        if strike_price < min_strike or strike_price > max_strike:
            continue
        ce_data = contracts.get("ce", {})
        pe_data = contracts.get("pe", {})
        for opt_type, data in [("CE", ce_data), ("PE", pe_data)]:
            greeks = data.get("greeks", {})
            data_list.append({
                "StrikePrice": strike_price, "Type": opt_type,
                "IV": data.get("implied_volatility", 0), "OI": data.get("oi", 0),
                "LTP": data.get("last_price", 0), "Volume": data.get("volume", 0),
                "Delta": greeks.get("delta", 0), "Gamma": greeks.get("gamma", 0),
                "Theta": greeks.get("theta", 0), "Vega": greeks.get("vega", 0)
            })

    df = pd.DataFrame(data_list)
    df["Timestamp"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    # OI/IV/LTP changes
    df["OI_Change"] = df.apply(lambda r: ((r["OI"] - previous_data.get(f'{r["StrikePrice"]}_{r["Type"]}', {}).get("OI", r["OI"])) / r["OI"]) * 100 if r["OI"] else 0, axis=1)
    df["IV_Change"] = df.apply(lambda r: ((r["IV"] - previous_data.get(f'{r["StrikePrice"]}_{r["Type"]}', {}).get("IV", r["IV"])) / r["IV"]) * 100 if r["IV"] else 0, axis=1)
    df["LTP_Change"] = df.apply(lambda r: ((r["LTP"] - previous_data.get(f'{r["StrikePrice"]}_{r["Type"]}', {}).get("LTP", r["LTP"])) / r["LTP"]) * 100 if r["LTP"] else 0, axis=1)
    df[['SentimentScore', 'SentimentBias']] = df.apply(score_option_sentiment, axis=1)

    # Update rolling data
    for _, row in df.iterrows():
        key = f"{row['StrikePrice']}_{row['Type']}"
        if key not in rolling_data:
            rolling_data[key] = deque(maxlen=5)
        rolling_data[key].append({"IV": row["IV"], "OI": row["OI"], "LTP": row["LTP"]})

    # Append to strike_sentiment_log
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    for _, row in df.iterrows():
        st.session_state.strike_sentiment_log.append({
            "Timestamp": timestamp, "StrikePrice": row["StrikePrice"], "Type": row["Type"],
            "LTP": row["LTP"], "OI": row["OI"], "LTP_Change": row["LTP_Change"],
            "OI_Change": row["OI_Change"], "Volume": row["Volume"], "IV": row["IV"],
            "IV_Change": row["IV_Change"], "Theta": row["Theta"], "Vega": row["Vega"],
            "SentimentScore": row["SentimentScore"], "SentimentBias": row["SentimentBias"],
            "UnderlyingValue": underlying_price
        })

    st.session_state.strike_sentiment_log = st.session_state.strike_sentiment_log[-20500:]

    # Save previous_data
    for _, row in df.iterrows():
        key = f"{row['StrikePrice']}_{row['Type']}"
        previous_data[key] = {"IV": row["IV"], "OI": row["OI"], "LTP": row["LTP"], "Vega": row["Vega"], "Theta": row["Theta"], "Delta": row["Delta"], "Gamma": row["Gamma"]}
    previous_data["underlying_price"] = underlying_price

    # Persist log
    pd.DataFrame(st.session_state.strike_sentiment_log).to_csv("sentiment_log_backup.csv", index=False)

    return pd.DataFrame(st.session_state.strike_sentiment_log)


def render_hft_chart(log_df):
    """Render the HFT Algo Scanner - Multi-Strike Analysis chart only."""
    if log_df.empty or len(log_df) < 2:
        st.info("Collecting data... Refresh in a minute to see the chart.")
        return

    log_df = log_df.copy()
    log_df['Exposure'] = log_df['OI'] * log_df['LTP']
    log_df['Volume_Weighted_Exposure'] = log_df.get('Volume', 1) * log_df['Exposure']
    log_df['OI_Change_Pct'] = log_df.get('OI_Change', 0) / (log_df['OI'] + 1) * 100
    log_df['IV_Change_Pct'] = log_df.get('IV_Change', 0) / (log_df.get('IV', 1) + 0.01) * 100
    log_df['LTP_Change_Pct'] = log_df.get('LTP_Change', 0) / (log_df['LTP'] + 0.01) * 100

    # MFI
    typical_price = (log_df['LTP'] + log_df.get('High', log_df['LTP']) + log_df.get('Low', log_df['LTP'])) / 3
    money_flow = typical_price * log_df.get('Volume', 1)
    positive_flow = money_flow.where(log_df['LTP_Change_Pct'] > 0, 0)
    negative_flow = money_flow.where(log_df['LTP_Change_Pct'] < 0, 0)
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    log_df['MFI'] = (100 - (100 / (1 + (positive_mf / (negative_mf + 1))))).fillna(50)

    # Activity metric
    def calc_activity(row):
        base = row['Volume_Weighted_Exposure']
        vf = np.log1p(row.get('Volume', 1) / 50)
        oi_m = min(abs(row['OI_Change_Pct']) / 3, 4)
        pm = abs(row['LTP_Change_Pct']) / 8
        ivf = abs(row['IV_Change_Pct']) / 15
        mfif = abs(row.get('MFI', 50) - 50) / 25
        return base * (1 + vf * 0.3 + oi_m * 0.25 + pm * 0.2 + ivf * 0.15 + mfif * 0.1)

    log_df['Activity_Metric'] = log_df.apply(calc_activity, axis=1)
    log_df['Dark_Pool_Activity'] = log_df.apply(lambda r: r['OI_Change_Pct'] > 5 and abs(r['LTP_Change_Pct']) < 2 and r.get('Volume', 1) > 100, axis=1)

    # Enhanced sentiment
    spot_prices = log_df.groupby('Timestamp')['UnderlyingValue'].first().reset_index()
    spot_prices['Price_Change'] = spot_prices['UnderlyingValue'].pct_change() * 100
    spot_prices['Price_Velocity'] = spot_prices['Price_Change'].diff()

    def enhanced_sentiment(row):
        pc, v = row['Price_Change'], row.get('Price_Velocity', 0)
        if pc > 0.1 and v > 0: return 'Strong Bullish'
        elif pc > 0.05: return 'Bullish'
        elif pc < -0.1 and v < 0: return 'Strong Bearish'
        elif pc < -0.05: return 'Bearish'
        return 'Neutral'

    spot_prices['Enhanced_Sentiment'] = spot_prices.apply(enhanced_sentiment, axis=1)
    log_df = log_df.merge(spot_prices[['Timestamp', 'Enhanced_Sentiment']], on='Timestamp', how='left')

    # Flow classification
    def classify_flow(row):
        activity = row['Activity_Metric']
        oi_c = row['OI_Change_Pct']
        lt_c = row['LTP_Change_Pct']
        iv_c = row['IV_Change_Pct']
        vol = row.get('Volume', 1)
        sent = row.get('Enhanced_Sentiment', 'Neutral')
        opt = row['Type']
        mfi = row.get('MFI', 50)
        dp = row.get('Dark_Pool_Activity', False)

        act_pct = np.percentile(log_df['Activity_Metric'], 65)
        oi_vals = log_df['OI_Change_Pct'].replace([np.inf, -np.inf], np.nan).dropna()
        sig_oi = np.percentile(oi_vals, 75) if not oi_vals.empty else 5
        sig_vol = 75

        if dp: return f"Dark Pool {opt}", "#800080", activity * 1.5
        if activity < act_pct: return "Low Activity", "#333333", activity * 0.1

        if opt == "CE":
            if mfi > 70 and vol > sig_vol and lt_c > 2 and sent in ['Bullish', 'Strong Bullish']:
                return "Aggressive Call Buy", "#006400", activity * 1.2
            elif oi_c > sig_oi and lt_c < -2 and mfi < 30 and vol > sig_vol:
                return "Heavy Call Short", "#DC143C", activity * 1.1
            elif vol > sig_vol and lt_c > 1 and (sent in ['Bullish', 'Strong Bullish'] or iv_c > 3):
                return "Call Buy", "#2E8B57", activity
            elif oi_c > sig_oi and lt_c < -1 and vol > sig_vol:
                return "Call Short", "#FF6B6B", activity
            return "Call Activity", "#90EE90", activity * 0.7
        else:
            if mfi < 30 and vol > sig_vol and lt_c > 2 and sent in ['Bearish', 'Strong Bearish']:
                return "Aggressive Put Buy", "#8B0000", activity * 1.2
            elif oi_c > sig_oi and lt_c < -2 and mfi > 70 and vol > sig_vol:
                return "Heavy Put Write", "#228B22", activity * 1.1
            elif vol > sig_vol and lt_c > 1 and (sent in ['Bearish', 'Strong Bearish'] or iv_c > 3):
                return "Put Buy", "#8B0000", activity
            elif oi_c > sig_oi and lt_c < -1 and vol > sig_vol:
                return "Put Write", "#90EE90", activity
            return "Put Activity", "#FFB6C1", activity * 0.7

    cls = log_df.apply(lambda r: pd.Series(classify_flow(r)), axis=1)
    log_df[['Flow_Type', 'Color', 'Weighted_Activity']] = cls

    # Time aggregation
    log_df['TimeSlot_dt'] = pd.to_datetime(log_df['Timestamp']).dt.floor('1min')
    flow_agg = log_df.groupby(['TimeSlot_dt', 'Flow_Type', 'Color']).agg({
        'Weighted_Activity': 'sum', 'UnderlyingValue': 'last', 'Volume': 'sum', 'MFI': 'mean'
    }).reset_index().sort_values('TimeSlot_dt')
    flow_agg['TimeSlot'] = flow_agg['TimeSlot_dt'].dt.strftime('%H:%M')

    price_data = log_df.groupby('TimeSlot_dt').agg({'UnderlyingValue': 'last', 'MFI': 'mean'}).reset_index().sort_values('TimeSlot_dt')

    # Chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.1, 0.3],
        subplot_titles=("📈 Spot Price Movement", "💹 Money Flow Index (MFI)", "🔍 HFT Algo Scanner - Multi-Strike Analysis"))

    fig.add_trace(go.Scatter(x=price_data['TimeSlot_dt'], y=price_data['UnderlyingValue'], mode='lines',
        name='Spot Price', line=dict(color='#00BFFF', width=2),
        hovertemplate="<b>%{x|%H:%M}</b><br>Price: ₹%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data['TimeSlot_dt'], y=price_data['MFI'], mode='lines',
        name='MFI', line=dict(color='#FFD700', width=2), fill='tonexty', fillcolor='rgba(255, 215, 0, 0.1)',
        hovertemplate="<b>%{x|%H:%M}</b><br>MFI: %{y:.1f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    excluded = ['Call Activity', 'Put Activity', 'Call Buy', 'Put Buy', 'Low Activity']
    for ft in flow_agg['Flow_Type'].unique():
        if ft in excluded:
            continue
        fd = flow_agg[flow_agg['Flow_Type'] == ft]
        if not fd.empty:
            fig.add_trace(go.Bar(x=fd['TimeSlot_dt'], y=fd['Weighted_Activity'], name=ft,
                marker_color=fd['Color'].iloc[0], opacity=0.8,
                hovertemplate=f"<b>{ft}</b><br>Time: %{{x|%H:%M}}<br>Flow: %{{y:,.0f}}<extra></extra>"), row=3, col=1)

    for r in [1, 2, 3]:
        fig.update_xaxes(rangeslider_visible=False, fixedrange=False, type='date', tickformat='%H:%M', row=r, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, row=r, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="MFI", row=2, col=1)
    fig.update_yaxes(title_text="Big Money Flow", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(title=dict(text="🎯 HFT Algo Scanner - Multi-Strike Analysis", x=0.5, font=dict(size=18)),
        barmode='stack', height=900, plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color='black', size=11), legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.02))

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False})


# --- Main ---
st.set_page_config(page_title="HFT Algo Scanner", layout="wide")

expiry_dates = get_expiry_dates()
if not expiry_dates:
    st.error("Could not fetch expiry dates.")
    st.stop()

nearest_expiry = expiry_dates[0]

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

with st.spinner("Fetching data..."):
    option_chain = fetch_option_chain(nearest_expiry)
    if option_chain:
        log_df = build_strike_sentiment_log(option_chain)
        render_hft_chart(log_df)
    else:
        st.error("Failed to fetch option chain. Try again.")
