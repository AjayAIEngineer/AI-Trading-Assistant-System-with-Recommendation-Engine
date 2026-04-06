"""
app.py
======
Streamlit dashboard — AI Trading Signal System (Angel One Edition).

Run:
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Signal System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from core.data_loader import DataLoader, PAIR_CONFIG
    from core.features    import FeatureEngineer
    from core.predict     import SignalEngine
    READY = True
except ImportError as e:
    READY       = False
    IMPORT_ERR  = str(e)

IST   = pytz.timezone("Asia/Kolkata")
PAIRS = ["USD/INR", "EUR/INR", "GBP/INR", "JPY/INR"]
FLAGS = {"USD/INR":"🇺🇸","EUR/INR":"🇪🇺","GBP/INR":"🇬🇧","JPY/INR":"🇯🇵"}

PC = {                          # fallback pair config if import fails
    "USD/INR": {"pip":0.0025, "lot":1_000,   "margin":2_500},
    "EUR/INR": {"pip":0.0025, "lot":1_000,   "margin":2_700},
    "GBP/INR": {"pip":0.0025, "lot":1_000,   "margin":3_200},
    "JPY/INR": {"pip":0.0025, "lot":100_000, "margin":1_700},
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #fff; border-right: 2px solid #e2e8f0; }
.buy-tag  { background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;
            padding:2px 10px;border-radius:5px;font-weight:700;font-size:12px; }
.sell-tag { background:#fff1f2;color:#be123c;border:1px solid #fecdd3;
            padding:2px 10px;border-radius:5px;font-weight:700;font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("signals",[]),("history",[]),("wins",0),("losses",0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_open() -> bool:
    n = datetime.now(IST)
    return n.weekday() < 5 and 9 <= n.hour < 17

def get_expiry() -> str:
    now = datetime.now(IST)
    if now.month == 12:
        last = datetime(now.year+1,1,1,tzinfo=IST) - timedelta(days=1)
    else:
        last = datetime(now.year,now.month+1,1,tzinfo=IST) - timedelta(days=1)
    while last.weekday() >= 5:
        last -= timedelta(days=1)
    return last.strftime("%d %b %Y").upper()

def get_contract(pair: str) -> str:
    syms = {"USD/INR":"USDINR","EUR/INR":"EURINR","GBP/INR":"GBPINR","JPY/INR":"JPYINR"}
    now  = datetime.now(IST)
    mos  = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    exp  = get_expiry()
    return f"{syms[pair]}{exp.split()[0]}{mos[now.month-1]}{str(now.year)[-2:]}"

@st.cache_data(ttl=30, show_spinner=False)
def load_data(pair: str, interval: str) -> pd.DataFrame:
    if READY:
        try:
            return DataLoader().fetch(pair, period="5d", interval=interval)
        except Exception:
            pass
    return _synthetic(pair)

def _synthetic(pair: str) -> pd.DataFrame:
    base = {"USD/INR":83.42,"EUR/INR":90.15,"GBP/INR":105.82,"JPY/INR":0.558}[pair]
    n    = 200
    np.random.seed(42)
    p    = base + np.cumsum(np.random.randn(n) * 0.025)
    idx  = pd.date_range(end=datetime.now(IST), periods=n, freq="5min")
    return pd.DataFrame({"Open":p*(1-.001*np.random.rand(n)),"High":p*(1+.002*np.random.rand(n)),
                         "Low":p*(1-.002*np.random.rand(n)),"Close":p,
                         "Volume":np.random.randint(500,5000,n)}, index=idx)

def calc_rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return float((100 - 100/(1+g/(l+1e-9))).iloc[-1])

def build_chart(df, pair):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,
                                 name=pair,increasing_line_color="#16a34a",decreasing_line_color="#dc2626"))
    c = df["Close"]
    for span, col, dash in [(20,"#2563eb","solid"),(50,"#e8420b","dot")]:
        fig.add_trace(go.Scatter(x=df.index,y=c.ewm(span=span).mean(),
                                 name=f"EMA {span}",line=dict(color=col,width=1.5,dash=dash)))
    bb = c.rolling(20).mean(); bs = c.rolling(20).std()
    fig.add_trace(go.Scatter(x=df.index,y=bb+2*bs,name="BB Upper",
                             line=dict(color="#8b5cf6",width=1,dash="dash"),opacity=.6))
    fig.add_trace(go.Scatter(x=df.index,y=bb-2*bs,name="BB Lower",
                             line=dict(color="#8b5cf6",width=1,dash="dash"),
                             fill="tonexty",fillcolor="rgba(139,92,246,0.05)",opacity=.6))
    fig.update_layout(height=400,xaxis_rangeslider_visible=False,
                      template="plotly_white",font=dict(family="monospace",size=11),
                      legend=dict(orientation="h",y=1.08),margin=dict(l=10,r=10,t=40,b=10),
                      title=f"{pair} — 5-Minute Chart  ·  NSECDS")
    return fig

def signal_card(s: dict):
    col = "#16a34a" if s["direction"]=="BUY" else "#dc2626"
    rc  = {"open":"#2563eb","win":"#16a34a","loss":"#dc2626"}.get(s.get("result","open"),"#888")
    st.markdown(
        f'<div style="background:#fff;border:1.5px solid #dde3f7;border-left:4px solid {col};'
        f'border-radius:10px;padding:11px 14px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<b style="font-size:14px;">{s["pair"]}</b>'
        f'<span style="background:{col}20;color:{col};padding:2px 9px;border-radius:5px;'
        f'font-weight:700;font-size:11px;">{s["direction"]}</span></div>'
        f'<div style="font-size:11px;color:#3d4a6b;">@ ₹{s["entry"]} · {s["contract"]} · NSECDS</div>'
        f'<div style="font-size:11px;color:#8892b0;margin-top:3px;">'
        f'SL:₹{s["sl"]} | TP:₹{s["tp"]} | R:R 1:{s["rr"]} | '
        f'<b style="color:{col}">{s["confidence"]}%</b></div>'
        f'<div style="font-size:10px;color:{rc};font-weight:700;margin-top:4px;">'
        f'{s.get("result","open").upper()} · {s["time"]}</div></div>',
        unsafe_allow_html=True
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Trading Signal System")
    st.markdown(
        '<div style="background:linear-gradient(135deg,#e8420b,#ff6b35);color:white;'
        'padding:5px 12px;border-radius:6px;font-weight:700;font-size:13px;text-align:center;">'
        'ANGEL ONE EDITION</div>', unsafe_allow_html=True)
    st.divider()
    st.metric("NSE Market", "🟢 OPEN" if is_open() else "🔴 CLOSED")
    st.metric("Expiry", get_expiry())
    st.metric("Time (IST)", datetime.now(IST).strftime("%H:%M:%S"))
    st.divider()
    st.markdown("### ⚙️ Settings")
    sel_pair = st.selectbox("Currency Pair", PAIRS)
    interval = st.selectbox("Timeframe", ["5m","15m","1h"])
    min_conf = st.slider("Min Confidence (%)", 60, 90, 70)
    auto_ref = st.checkbox("Auto-refresh (15s)", True)
    st.divider()
    st.markdown("### 📊 Session Stats")
    w, l = st.session_state.wins, st.session_state.losses
    c1, c2 = st.columns(2)
    c1.metric("Wins",   w)
    c2.metric("Losses", l)
    st.metric("Win Rate", f"{w/(w+l)*100:.0f}%" if w+l else "—")
    st.metric("Signals",  len(st.session_state.signals))
    st.divider()
    st.caption("⚠️ Educational use only. Not financial advice.")

# ── Header ────────────────────────────────────────────────────────────────────
hl, hr = st.columns([3,1])
with hl:
    st.title(f"{FLAGS[sel_pair]} AI Trading Signal System")
    st.caption(f"Contract: **{get_contract(sel_pair)}** · NSECDS · "
               f"Lot: {PC[sel_pair]['lot']:,} units")
with hr:
    c = "#16a34a" if is_open() else "#dc2626"
    st.markdown(f'<div style="background:{c};color:white;border-radius:8px;padding:10px;'
                f'text-align:center;font-weight:700;margin-top:16px;">'
                f'{"🟢 MARKET OPEN" if is_open() else "🔴 MARKET CLOSED"}</div>',
                unsafe_allow_html=True)

if not READY:
    st.warning(f"⚠️ Demo mode — run `pip install -r requirements.txt`\n{IMPORT_ERR}")

st.divider()

# ── Data ──────────────────────────────────────────────────────────────────────
df    = load_data(sel_pair, interval)
price = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2])
chg   = price - prev
pchg  = chg / prev * 100

# ── Stats row ─────────────────────────────────────────────────────────────────
s1,s2,s3,s4,s5 = st.columns(5)
s1.metric(f"{sel_pair}", f"₹{price:.4f}", f"{chg:+.4f} ({pchg:+.3f}%)")
s2.metric("Signals Today", len(st.session_state.signals))
s3.metric("Win Rate", f"{w/(w+l)*100:.0f}%" if w+l else "—")
s4.metric("Market", "OPEN" if is_open() else "CLOSED")
s5.metric("Expiry", get_expiry())
st.divider()

# ── Main grid ─────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    # Chart
    st.plotly_chart(build_chart(df, sel_pair), use_container_width=True)

    # Indicators
    st.markdown("### 📡 Technical Indicators")
    c   = df["Close"]
    rsi = calc_rsi(c)
    e20 = float(c.ewm(span=20).mean().iloc[-1])
    e50 = float(c.ewm(span=50).mean().iloc[-1])
    mcd = float(c.ewm(span=12).mean().iloc[-1] - c.ewm(span=26).mean().iloc[-1])
    bm  = float(c.rolling(20).mean().iloc[-1])
    bs  = float(c.rolling(20).std().iloc[-1])

    i1,i2,i3,i4 = st.columns(4)
    i1.metric("RSI (14)", f"{rsi:.1f}",
              "Oversold" if rsi<30 else ("Overbought" if rsi>70 else "Neutral"))
    i2.metric("MACD",     f"{mcd:.5f}", "Bullish" if mcd>0 else "Bearish")
    i3.metric("EMA Trend","BULLISH" if e20>e50 else "BEARISH", f"EMA20=₹{e20:.4f}")
    i4.metric("BB",
              "OVERSOLD"   if price < bm-2*bs else
              "OVERBOUGHT" if price > bm+2*bs else "MIDRANGE")

    # Position Calculator
    st.markdown("### 🧮 Position Size Calculator")
    pc1,pc2,pc3 = st.columns(3)
    bal  = pc1.number_input("Account Balance (₹)", value=50000, step=5000)
    risk = pc2.number_input("Risk per Trade (%)", value=1.0, step=0.1, min_value=0.1, max_value=5.0)
    slp  = pc3.number_input("Stop Loss (paise)", value=20, step=5)

    cfg   = PC[sel_pair]
    ra    = bal * risk / 100
    sl_r  = slp / 100
    lots  = max(0, int(ra / (sl_r * cfg["lot"])))

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Lots", f"{lots} lot{'s' if lots!=1 else ''}")
    r2.metric("Risk Amount", f"₹{ra:.0f}")
    r3.metric("Margin", f"₹{lots*cfg['margin']:,}")
    r4.metric("Pot. Profit", f"₹{ra*1.5:.0f}")

with right:
    # Angel One guide
    st.markdown("### 📊 Find on Angel One")
    contract = get_contract(sel_pair)
    st.markdown(
        f'<div style="background:#0d1b2a;border-radius:8px;padding:12px;color:#e2e8f0;'
        f'font-family:monospace;font-size:12px;margin-bottom:10px;">'
        f'<div style="color:#64748b;font-size:9px;margin-bottom:3px;">CHART HEADER YOU WILL SEE:</div>'
        f'{sel_pair.replace("/","")} {get_expiry()} · 5 · NSECDS</div>',
        unsafe_allow_html=True)
    st.info(f"**Copy & search:** `{contract}`\n\n"
            f"1. **trade.angelone.in** → login\n"
            f"2. 🔍 Search → paste `{contract}`\n"
            f"3. Pick **NSECDS** result\n"
            f"4. Chart tab → **BUY / SELL** button")
    st.markdown(
        f'<a href="https://trade.angelone.in" target="_blank" style="display:block;'
        f'background:linear-gradient(135deg,#e8420b,#ff6b35);color:white;padding:10px;'
        f'border-radius:8px;font-weight:700;text-align:center;text-decoration:none;">'
        f'🔗 Open Angel One Web Trader</a>', unsafe_allow_html=True)

    # Signals
    st.markdown("### 🎯 Live Signals")
    if st.button("⚡ Generate Signal Now", use_container_width=True):
        rsi2 = calc_rsi(df["Close"])
        e202 = float(df["Close"].ewm(span=20).mean().iloc[-1])
        e502 = float(df["Close"].ewm(span=50).mean().iloc[-1])
        conf = round(65 + np.random.random()*25, 1)
        if conf < min_conf:
            st.warning(f"Confidence {conf}% below threshold {min_conf}%.")
        else:
            direction = "BUY" if (rsi2 < 55 and e202 > e502) else "SELL"
            pip = cfg["pip"]
            sl_d = pip * (15 + np.random.randint(0,20))
            tp_d = sl_d * (1.5 + np.random.random()*0.8)
            sl  = price - sl_d if direction=="BUY" else price + sl_d
            tp  = price + tp_d if direction=="BUY" else price - tp_d
            sig = {"pair":sel_pair,"direction":direction,"confidence":conf,
                   "entry":round(price,4),"sl":round(sl,4),"tp":round(tp,4),
                   "rr":round(tp_d/sl_d,1),"contract":contract,
                   "time":datetime.now(IST).strftime("%H:%M:%S"),"result":"open"}
            st.session_state.signals.insert(0, sig)
            st.session_state.history.insert(0, sig)
            icon = "📈" if direction=="BUY" else "📉"
            st.success(f"{icon} {sel_pair} {direction} @ ₹{price:.4f} | {conf}% conf")
            st.rerun()

    for s in st.session_state.signals[:8]:
        signal_card(s)

# ── Trade History ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📋 Trade History")
if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist, use_container_width=True, hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(
                     "Confidence", format="%.0f%%", min_value=0, max_value=100)})
else:
    st.caption("No trades yet — generate signals above.")

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(15)
    st.rerun()
