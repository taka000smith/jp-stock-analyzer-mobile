
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="日本株分析 Mobile", layout="centered")

st.markdown("""
<style>
/* Mobile first */
.block-container {
    padding-top: 4.2rem;
    padding-bottom: 1.5rem;
    padding-left: 0.8rem;
    padding-right: 0.8rem;
    max-width: 760px;
}
h1 {
    font-size: 1.55rem !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-size: 1.25rem !important;
    margin-top: 0.7rem !important;
    margin-bottom: 0.3rem !important;
}
h3 {
    font-size: 1.05rem !important;
    margin-top: 0.65rem !important;
    margin-bottom: 0.25rem !important;
}
div[data-testid="stMetric"] {
    padding: 0.28rem 0.35rem;
    border-radius: 0.55rem;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}
button[kind="primary"] {
    min-height: 2.8rem;
    font-size: 1rem !important;
}
.stTextArea textarea {
    font-size: 1rem !important;
}
@media (max-width: 640px) {
    div[data-testid="column"] {
        min-width: 0 !important;
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def jp_ticker(code: str) -> str:
    code = str(code).strip().upper()
    if not code:
        return ""
    return code if code.endswith(".T") else f"{code}.T"

def sf(x):
    try:
        if x is None:
            return np.nan
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan

def fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "N/A"

def fmt_num(x, d=1):
    return f"{x:.{d}f}" if pd.notna(x) else "N/A"

def fmt_yen(x):
    return f"¥{x:,.0f}" if pd.notna(x) else "N/A"

def growth(cur, prev):
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return np.nan
    return cur / prev - 1

def cagr(cur, old, years):
    if pd.isna(cur) or pd.isna(old) or old <= 0 or cur <= 0 or years <= 0:
        return np.nan
    return (cur / old) ** (1 / years) - 1

def row_values(df, candidates):
    if df is None or df.empty:
        return []
    idx_lower = {str(i).lower(): i for i in df.index}
    for cand in candidates:
        key = cand.lower()
        if key in idx_lower:
            row = df.loc[idx_lower[key]]
            return [sf(v) for v in row.values]
    for cand in candidates:
        key = cand.lower()
        for idx in df.index:
            if key in str(idx).lower():
                row = df.loc[idx]
                return [sf(v) for v in row.values]
    return []

def row_latest(df, candidates, pos=0):
    vals = row_values(df, candidates)
    return vals[pos] if len(vals) > pos else np.nan

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def score_band(x, bands):
    """
    bands: list of tuples (threshold, score), descending threshold
    """
    if pd.isna(x):
        return None
    for threshold, score in bands:
        if x >= threshold:
            return score
    return 0

def weighted_score(items):
    vals = [(s, w) for s, w in items if s is not None and pd.notna(s)]
    if not vals:
        return np.nan
    wsum = sum(w for _, w in vals)
    return sum(s*w for s, w in vals) / wsum * 10

def grade(score):
    if pd.isna(score): return "N/A"
    if score >= 85: return "S"
    if score >= 75: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    return "D"

def entry_label(score):
    if pd.isna(score): return "判定不能"
    if score >= 80: return "買い候補"
    if score >= 65: return "条件付き監視"
    if score >= 50: return "待ち"
    return "見送り"

def trend_label(price, ma25, ma75):
    if all(pd.notna(x) for x in [price, ma25, ma75]):
        if price > ma25 > ma75: return "上昇トレンド"
        if price < ma25 < ma75: return "下降トレンド"
        return "中立"
    return "不明"

# =========================================================
# Analysis
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def analyze_one(code):
    symbol = jp_ticker(code)
    t = yf.Ticker(symbol)

    hist = t.history(period="2y", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError("株価データを取得できません。")

    hist = hist.dropna(subset=["Close"]).copy()
    hist["MA25"] = hist["Close"].rolling(25).mean()
    hist["MA75"] = hist["Close"].rolling(75).mean()
    hist["VOL20"] = hist["Volume"].rolling(20).mean()
    hist["RSI14"] = calc_rsi(hist["Close"], 14)
    hist["High20_prev"] = hist["High"].rolling(20).max().shift(1)
    hist["Low20_prev"] = hist["Low"].rolling(20).min().shift(1)

    tr = pd.concat([
        hist["High"] - hist["Low"],
        (hist["High"] - hist["Close"].shift(1)).abs(),
        (hist["Low"] - hist["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    hist["ATR14"] = tr.rolling(14).mean()

    last = hist.iloc[-1]
    price = sf(last["Close"])
    ma25 = sf(last["MA25"])
    ma75 = sf(last["MA75"])
    rsi = sf(last["RSI14"])
    vol20 = sf(last["VOL20"])
    vol_ratio = sf(last["Volume"] / vol20) if pd.notna(vol20) and vol20 > 0 else np.nan
    high20 = sf(last["High20_prev"])
    low20 = sf(last["Low20_prev"])
    atr = sf(last["ATR14"])

    h252 = hist.tail(252)
    high52 = sf(h252["High"].max()) if not h252.empty else np.nan
    dist_high52 = price/high52 - 1 if pd.notna(high52) and high52 else np.nan
    ma25_gap = price/ma25 - 1 if pd.notna(ma25) and ma25 else np.nan

    # Financials
    try: income = t.income_stmt
    except Exception: income = pd.DataFrame()
    try: balance = t.balance_sheet
    except Exception: balance = pd.DataFrame()
    try: cashflow = t.cashflow
    except Exception: cashflow = pd.DataFrame()
    try: q_income = t.quarterly_income_stmt
    except Exception: q_income = pd.DataFrame()

    revs = row_values(income, ["Total Revenue", "Operating Revenue"])
    ops = row_values(income, ["Operating Income"])
    epss = row_values(income, ["Diluted EPS", "Basic EPS"])
    nets = row_values(income, ["Net Income", "Net Income Common Stockholders"])

    rev0 = revs[0] if len(revs) > 0 else np.nan
    rev1 = revs[1] if len(revs) > 1 else np.nan
    op0 = ops[0] if len(ops) > 0 else np.nan
    op1 = ops[1] if len(ops) > 1 else np.nan
    eps0 = epss[0] if len(epss) > 0 else np.nan
    eps1 = epss[1] if len(epss) > 1 else np.nan
    net0 = nets[0] if len(nets) > 0 else np.nan
    net1 = nets[1] if len(nets) > 1 else np.nan

    rev_yoy = growth(rev0, rev1)
    op_yoy = growth(op0, op1)
    eps_yoy = growth(eps0, eps1)
    if pd.isna(eps_yoy):
        eps_yoy = growth(net0, net1)

    rev_cagr3 = cagr(revs[0], revs[3], 3) if len(revs) >= 4 else np.nan
    op_cagr3 = cagr(ops[0], ops[3], 3) if len(ops) >= 4 and ops[0] > 0 and ops[3] > 0 else np.nan
    eps_cagr3 = cagr(epss[0], epss[3], 3) if len(epss) >= 4 and epss[0] > 0 and epss[3] > 0 else np.nan

    op_margin = op0/rev0 if pd.notna(op0) and pd.notna(rev0) and rev0 else np.nan
    op_margin_prev = op1/rev1 if pd.notna(op1) and pd.notna(rev1) and rev1 else np.nan
    margin_change = op_margin - op_margin_prev if pd.notna(op_margin) and pd.notna(op_margin_prev) else np.nan

    equity = row_latest(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    assets = row_latest(balance, ["Total Assets"])
    equity_ratio = equity/assets if pd.notna(equity) and pd.notna(assets) and assets else np.nan

    op_cf = row_latest(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    free_cf = row_latest(cashflow, ["Free Cash Flow"])
    if pd.isna(free_cf):
        capex = row_latest(cashflow, ["Capital Expenditure", "Capital Expenditures"])
        if pd.notna(op_cf) and pd.notna(capex):
            free_cf = op_cf + capex  # yfinance capex often negative

    # Quarterly growth (latest vs same q last year, if available)
    q_revs = row_values(q_income, ["Total Revenue", "Operating Revenue"])
    q_ops = row_values(q_income, ["Operating Income"])
    q_rev_yoy = growth(q_revs[0], q_revs[4]) if len(q_revs) >= 5 else np.nan
    q_op_yoy = growth(q_ops[0], q_ops[4]) if len(q_ops) >= 5 else np.nan

    try:
        info = t.info or {}
    except Exception:
        info = {}

    name = info.get("shortName") or info.get("longName") or symbol
    sector = info.get("sector") or "N/A"
    industry = info.get("industry") or "N/A"

    trailing_per = sf(info.get("trailingPE"))
    forward_per = sf(info.get("forwardPE"))
    pbr = sf(info.get("priceToBook"))
    roe = sf(info.get("returnOnEquity"))
    if pd.isna(roe) and pd.notna(net0) and pd.notna(equity) and equity:
        roe = net0/equity

    forward_eps = sf(info.get("forwardEps"))
    trailing_eps = sf(info.get("trailingEps"))
    implied_eps_growth = growth(forward_eps, trailing_eps)

    # =====================================================
    # Company quality / fundamental score
    # =====================================================
    s_rev_yoy = score_band(rev_yoy, [(0.30,10),(0.20,9),(0.10,7),(0.05,5),(0,3)])
    s_op_yoy = score_band(op_yoy, [(0.30,10),(0.20,9),(0.10,7),(0.05,5),(0,3)])
    s_eps_yoy = score_band(eps_yoy, [(0.30,10),(0.20,9),(0.10,7),(0.05,5),(0,3)])
    s_rev_cagr = score_band(rev_cagr3, [(0.20,10),(0.15,9),(0.10,8),(0.05,6),(0,3)])
    s_op_cagr = score_band(op_cagr3, [(0.25,10),(0.20,9),(0.15,8),(0.10,7),(0.05,5),(0,3)])
    s_eps_cagr = score_band(eps_cagr3, [(0.25,10),(0.20,9),(0.15,8),(0.10,7),(0.05,5),(0,3)])

    if pd.isna(op_margin):
        s_margin = None
    else:
        if op_margin >= .20: s_margin = 10
        elif op_margin >= .15: s_margin = 9
        elif op_margin >= .10: s_margin = 8
        elif op_margin >= .05: s_margin = 6
        elif op_margin > 0: s_margin = 4
        else: s_margin = 0
        if pd.notna(margin_change):
            if margin_change >= .02: s_margin = min(10, s_margin+2)
            elif margin_change > 0: s_margin = min(10, s_margin+1)
            elif margin_change <= -.02: s_margin = max(0, s_margin-2)
            elif margin_change < 0: s_margin = max(0, s_margin-1)

    if pd.isna(roe): s_roe = None
    elif roe >= .20: s_roe = 10
    elif roe >= .15: s_roe = 9
    elif roe >= .10: s_roe = 7
    elif roe >= .08: s_roe = 5
    elif roe > 0: s_roe = 3
    else: s_roe = 0

    if pd.isna(equity_ratio): s_equity = None
    elif equity_ratio >= .60: s_equity = 10
    elif equity_ratio >= .40: s_equity = 8
    elif equity_ratio >= .30: s_equity = 6
    elif equity_ratio >= .20: s_equity = 4
    else: s_equity = 2

    s_cf = None
    if pd.notna(op_cf):
        s_cf = 8 if op_cf > 0 else 0
        if pd.notna(free_cf):
            if free_cf > 0: s_cf = min(10, s_cf+2)
            else: s_cf = max(0, s_cf-2)

    # Valuation score: prefer forward PER if available
    per_used = forward_per if pd.notna(forward_per) and forward_per > 0 else trailing_per
    growth_for_valuation = implied_eps_growth
    if pd.isna(growth_for_valuation):
        growth_for_valuation = eps_cagr3 if pd.notna(eps_cagr3) else eps_yoy

    if pd.isna(per_used) or per_used <= 0:
        s_val = None
    elif pd.notna(growth_for_valuation) and growth_for_valuation > 0:
        peg_like = per_used / (growth_for_valuation*100)
        if peg_like <= .8: s_val = 10
        elif peg_like <= 1.2: s_val = 8
        elif peg_like <= 1.8: s_val = 6
        elif peg_like <= 2.5: s_val = 4
        else: s_val = 2
    else:
        if per_used <= 15: s_val = 8
        elif per_used <= 25: s_val = 6
        elif per_used <= 40: s_val = 4
        else: s_val = 2

    fund_score = weighted_score([
        (s_rev_yoy, 0.8), (s_op_yoy, 1.0), (s_eps_yoy, 1.0),
        (s_rev_cagr, 0.9), (s_op_cagr, 1.1), (s_eps_cagr, 1.1),
        (s_margin, 1.0), (s_roe, 1.0), (s_cf, 0.8),
        (s_equity, 0.5), (s_val, 1.0),
    ])

    # =====================================================
    # Entry / technical score
    # =====================================================
    # trend
    if pd.notna(ma25) and pd.notna(ma75):
        if price > ma25 > ma75: s_trend = 10
        elif price > ma25 and ma25 <= ma75: s_trend = 7
        elif price < ma25 and ma25 > ma75: s_trend = 5
        else: s_trend = 2
    else:
        s_trend = None

    # RSI: good momentum but penalize overheating
    if pd.isna(rsi): s_rsi = None
    elif 50 <= rsi <= 68: s_rsi = 10
    elif 45 <= rsi < 50 or 68 < rsi <= 72: s_rsi = 8
    elif 40 <= rsi < 45 or 72 < rsi <= 76: s_rsi = 6
    elif 30 <= rsi < 40 or 76 < rsi <= 80: s_rsi = 3
    else: s_rsi = 1

    # Volume
    if pd.isna(vol_ratio): s_vol = None
    elif vol_ratio >= 2.0: s_vol = 10
    elif vol_ratio >= 1.5: s_vol = 9
    elif vol_ratio >= 1.2: s_vol = 8
    elif vol_ratio >= .8: s_vol = 5
    else: s_vol = 3

    # 52w high proximity
    if pd.isna(high52): s_high52 = None
    else:
        ratio52 = price/high52
        if ratio52 >= .97: s_high52 = 10
        elif ratio52 >= .92: s_high52 = 9
        elif ratio52 >= .85: s_high52 = 7
        elif ratio52 >= .75: s_high52 = 5
        else: s_high52 = 3

    # 20d breakout setup
    if pd.isna(high20): s_break = None
    else:
        r20 = price/high20
        if 0.985 <= r20 <= 1.02: s_break = 10
        elif .95 <= r20 < .985: s_break = 8
        elif 1.02 < r20 <= 1.06: s_break = 6
        elif .90 <= r20 < .95: s_break = 5
        else: s_break = 3

    # MA25 gap: avoid chasing
    if pd.isna(ma25_gap): s_gap = None
    elif .00 <= ma25_gap <= .06: s_gap = 10
    elif -.03 <= ma25_gap < 0: s_gap = 8
    elif .06 < ma25_gap <= .10: s_gap = 6
    elif .10 < ma25_gap <= .15: s_gap = 3
    elif ma25_gap > .15: s_gap = 1
    else: s_gap = 4

    tech_score = weighted_score([
        (s_trend, 1.5),
        (s_rsi, 0.8),
        (s_vol, 0.8),
        (s_high52, 0.9),
        (s_break, 1.2),
        (s_gap, 1.0),
    ])

    # =====================================================
    # Entry candidates / stop
    # =====================================================
    breakout_entry = high20 * 1.003 if pd.notna(high20) else np.nan
    pullback_low = ma25 * .985 if pd.notna(ma25) else np.nan
    pullback_high = ma25 * 1.015 if pd.notna(ma25) else np.nan

    stop_candidates = []
    if pd.notna(ma25) and pd.notna(atr):
        v = ma25 - 2*atr
        if v < price:
            stop_candidates.append(v)
    if pd.notna(low20) and low20 < price:
        stop_candidates.append(low20*.995)
    stop_price = max(stop_candidates) if stop_candidates else np.nan

    risk_break = breakout_entry - stop_price if pd.notna(breakout_entry) and pd.notna(stop_price) else np.nan
    target_2r = breakout_entry + 2*risk_break if pd.notna(risk_break) and risk_break > 0 else np.nan

    # =====================================================
    # Decision explanation
    # =====================================================
    strengths, cautions = [], []

    if pd.notna(rev_cagr3) and rev_cagr3 >= .10: strengths.append(f"売上3年CAGR {rev_cagr3*100:.1f}%")
    if pd.notna(op_cagr3) and op_cagr3 >= .10: strengths.append(f"営業利益3年CAGR {op_cagr3*100:.1f}%")
    if pd.notna(eps_cagr3) and eps_cagr3 >= .10: strengths.append(f"EPS3年CAGR {eps_cagr3*100:.1f}%")
    if pd.notna(margin_change) and margin_change > 0: strengths.append("営業利益率改善")
    if pd.notna(roe) and roe >= .15: strengths.append(f"ROE {roe*100:.1f}%")
    if pd.notna(op_cf) and op_cf > 0: strengths.append("営業CFプラス")
    if price > ma25 > ma75: strengths.append("25日線＞75日線の上昇トレンド")
    if pd.notna(vol_ratio) and vol_ratio >= 1.2: strengths.append(f"出来高 {vol_ratio:.1f}倍")

    if pd.notna(op_yoy) and op_yoy < 0: cautions.append(f"営業利益YoY {op_yoy*100:.1f}%")
    if pd.notna(eps_yoy) and eps_yoy < 0: cautions.append(f"EPS YoY {eps_yoy*100:.1f}%")
    if pd.notna(rsi) and rsi >= 75: cautions.append(f"RSI {rsi:.1f}で過熱")
    if pd.notna(ma25_gap) and ma25_gap > .10: cautions.append(f"25日線から+{ma25_gap*100:.1f}%乖離")
    if pd.notna(per_used) and per_used >= 40: cautions.append(f"PER {per_used:.1f}倍")
    if pd.notna(op_cf) and op_cf <= 0: cautions.append("営業CFマイナス")

    company_grade = grade(fund_score)
    entry = entry_label(tech_score)

    # Combined interpretation
    if fund_score >= 75 and tech_score >= 70:
        overall = "有力候補"
    elif fund_score >= 75 and tech_score < 70:
        overall = "企業は良好・Entry待ち"
    elif fund_score < 60 and tech_score >= 75:
        overall = "チャート先行・ファンダ確認"
    elif fund_score >= 60 and tech_score >= 60:
        overall = "監視候補"
    else:
        overall = "見送り寄り"

    return {
        "code": str(code).strip(),
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "industry": industry,
        "price": price,
        "rev_yoy": rev_yoy,
        "op_yoy": op_yoy,
        "eps_yoy": eps_yoy,
        "rev_cagr3": rev_cagr3,
        "op_cagr3": op_cagr3,
        "eps_cagr3": eps_cagr3,
        "q_rev_yoy": q_rev_yoy,
        "q_op_yoy": q_op_yoy,
        "op_margin": op_margin,
        "op_margin_prev": op_margin_prev,
        "margin_change": margin_change,
        "roe": roe,
        "equity_ratio": equity_ratio,
        "op_cf": op_cf,
        "free_cf": free_cf,
        "trailing_per": trailing_per,
        "forward_per": forward_per,
        "per_used": per_used,
        "pbr": pbr,
        "implied_eps_growth": implied_eps_growth,
        "ma25": ma25,
        "ma75": ma75,
        "ma25_gap": ma25_gap,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "high20": high20,
        "high52": high52,
        "dist_high52": dist_high52,
        "atr": atr,
        "fund_score": fund_score,
        "company_grade": company_grade,
        "tech_score": tech_score,
        "entry_label": entry,
        "overall": overall,
        "trend": trend_label(price, ma25, ma75),
        "breakout_entry": breakout_entry,
        "pullback_low": pullback_low,
        "pullback_high": pullback_high,
        "stop_price": stop_price,
        "target_2r": target_2r,
        "strengths": strengths,
        "cautions": cautions,
        "hist": hist,
    }

# =========================================================
# Session state
# =========================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "analysis_errors" not in st.session_state:
    st.session_state.analysis_errors = []
if "analysis_has_run" not in st.session_state:
    st.session_state.analysis_has_run = False

# =========================================================
# UI
# =========================================================
st.markdown("""
<div style="
    padding: 0.15rem 0 0.35rem 0;
    margin: 0;
">
  <div style="
      font-size: 1.65rem;
      font-weight: 800;
      line-height: 1.2;
      color: inherit;
      margin-bottom: 0.15rem;
  ">
    📈 日本株分析
  </div>
  <div style="
      font-size: 0.95rem;
      opacity: 0.78;
      line-height: 1.25;
  ">
    ファンダ＋テクニカル
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("企業評価とEntry評価を分けて確認できます。")

st.subheader("銘柄入力")
codes_text = st.text_area(
    "コード（カンマ/空白/改行区切り）",
    value="",
    height=92,
    placeholder="例：7203, 6758, 8035",
    label_visibility="collapsed"
)
run = st.button("分析する", type="primary", use_container_width=True)



if run:
    raw = codes_text.replace(",", " ").replace("、", " ").split()
    codes = list(dict.fromkeys([x.strip() for x in raw if x.strip()]))

    if not codes:
        st.warning("銘柄コードを入力してください。")
        st.stop()

    results, errors = [], []
    prog = st.progress(0, text="データ取得中...")

    for i, code in enumerate(codes):
        try:
            results.append(analyze_one(code))
        except Exception as e:
            errors.append((code, str(e)))
        prog.progress((i+1)/len(codes), text=f"{code} を分析中...")
    prog.empty()

    st.session_state.analysis_results = results
    st.session_state.analysis_errors = errors
    st.session_state.analysis_has_run = True

results = st.session_state.analysis_results
errors = st.session_state.analysis_errors

if st.session_state.analysis_has_run:
    if errors:
        st.warning("一部銘柄のデータ取得に失敗しました。")
        for c, e in errors:
            st.write(f"- {c}: {e}")

    if results:
        rows = []
        for r in results:
            rows.append({
                "コード": r["code"],
                "銘柄名": r["name"],
                "株価": round(r["price"],1) if pd.notna(r["price"]) else np.nan,
                "企業Grade": r["company_grade"],
                "企業点": round(r["fund_score"],1) if pd.notna(r["fund_score"]) else np.nan,
                "Entry点": round(r["tech_score"],1) if pd.notna(r["tech_score"]) else np.nan,
                "総合判断": r["overall"],
                "売上YoY": fmt_pct(r["rev_yoy"]),
                "営利YoY": fmt_pct(r["op_yoy"]),
                "EPS YoY": fmt_pct(r["eps_yoy"]),
                "売上3Y CAGR": fmt_pct(r["rev_cagr3"]),
                "営利3Y CAGR": fmt_pct(r["op_cagr3"]),
                "EPS3Y CAGR": fmt_pct(r["eps_cagr3"]),
                "ROE": fmt_pct(r["roe"]),
                "予想PER": round(r["forward_per"],1) if pd.notna(r["forward_per"]) else np.nan,
                "RSI": round(r["rsi"],1) if pd.notna(r["rsi"]) else np.nan,
                "出来高倍率": round(r["vol_ratio"],2) if pd.notna(r["vol_ratio"]) else np.nan,
                "25日線乖離": fmt_pct(r["ma25_gap"]),
                "Entry判定": r["entry_label"],
                "ブレイクEntry": round(r["breakout_entry"],1) if pd.notna(r["breakout_entry"]) else np.nan,
                "押し目下限": round(r["pullback_low"],1) if pd.notna(r["pullback_low"]) else np.nan,
                "押し目上限": round(r["pullback_high"],1) if pd.notna(r["pullback_high"]) else np.nan,
                "損切り参考": round(r["stop_price"],1) if pd.notna(r["stop_price"]) else np.nan,
            })

        out = pd.DataFrame(rows).sort_values(
            by=["企業点","Entry点"],
            ascending=False,
            na_position="last"
        )

        st.subheader("📋 銘柄一覧")
        for rr in results:
            with st.container(border=True):
                st.markdown(f"**{rr['code']} {rr['name']}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("企業", rr["company_grade"])
                c2.metric("企業点", fmt_num(rr["fund_score"]))
                c3.metric("Entry", fmt_num(rr["tech_score"]))
                st.caption(f"{fmt_yen(rr['price'])} ｜ {rr['overall']} ｜ {rr['entry_label']}")

        csv = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSVをダウンロード",
            csv,
            "stock_screening_v2.csv",
            "text/csv"
        )

        st.divider()
        st.subheader("🔎 個別分析")

        selected = st.selectbox(
            "詳しく見る銘柄",
            [r["code"] for r in results],
            format_func=lambda x: next(f"{r['code']} {r['name']}" for r in results if r["code"] == x)
        )
        r = next(x for x in results if x["code"] == selected)

        st.markdown(f"## {r['code']}  {r['name']}")
        st.caption(f"{r['sector']} / {r['industry']}")

        st.markdown(f"### {r['overall']}")
        c1,c2 = st.columns(2)
        c1.metric("現在値", fmt_yen(r["price"]))
        c2.metric("企業Grade", r["company_grade"])
        c3,c4 = st.columns(2)
        c3.metric("企業点", fmt_num(r["fund_score"]))
        c4.metric("Entry点", fmt_num(r["tech_score"]))

        st.markdown("### 🏢 企業評価")
        a1,a2 = st.columns(2)
        a1.metric("売上YoY", fmt_pct(r["rev_yoy"]))
        a2.metric("営利YoY", fmt_pct(r["op_yoy"]))
        a3,a4 = st.columns(2)
        a3.metric("EPS YoY", fmt_pct(r["eps_yoy"]))
        a4.metric("売上3Y CAGR", fmt_pct(r["rev_cagr3"]))
        a5,a6 = st.columns(2)
        a5.metric("営利3Y CAGR", fmt_pct(r["op_cagr3"]))
        a6.metric("EPS3Y CAGR", fmt_pct(r["eps_cagr3"]))

        b1,b2 = st.columns(2)
        b1.metric("営業利益率", fmt_pct(r["op_margin"]))
        b2.metric("利益率変化", fmt_pct(r["margin_change"]))
        b3,b4 = st.columns(2)
        b3.metric("ROE", fmt_pct(r["roe"]))
        b4.metric("自己資本比率", fmt_pct(r["equity_ratio"]))
        b5,b6 = st.columns(2)
        b5.metric("予想PER", fmt_num(r["forward_per"]))
        b6.metric("PBR", fmt_num(r["pbr"]))

        if pd.notna(r["q_rev_yoy"]) or pd.notna(r["q_op_yoy"]):
            q1,q2 = st.columns(2)
            q1.metric("最新四半期 売上YoY", fmt_pct(r["q_rev_yoy"]))
            q2.metric("最新四半期 営利YoY", fmt_pct(r["q_op_yoy"]))

        st.markdown("### 📊 Entry評価")
        d1,d2 = st.columns(2)
        d1.metric("トレンド", r["trend"])
        d2.metric("25日線乖離", fmt_pct(r["ma25_gap"]))
        d3,d4 = st.columns(2)
        d3.metric("RSI(14)", fmt_num(r["rsi"]))
        d4.metric("出来高/20日平均", fmt_num(r["vol_ratio"],2))
        with st.expander("移動平均線を表示"):
            m1,m2 = st.columns(2)
            m1.metric("25日線", fmt_yen(r["ma25"]))
            m2.metric("75日線", fmt_yen(r["ma75"]))

        st.markdown(f"### Entry判定：**{r['entry_label']}**")

        e1,e2 = st.columns(2)
        e1.metric("ブレイクEntry", fmt_yen(r["breakout_entry"]))
        e2.metric("損切り参考", fmt_yen(r["stop_price"]))
        e3,e4 = st.columns(2)
        e3.metric(
            "押し目Entry",
            f"{fmt_yen(r['pullback_low'])}〜{fmt_yen(r['pullback_high'])}"
            if pd.notna(r["pullback_low"]) else "N/A"
        )
        e4.metric("2R目標", fmt_yen(r["target_2r"]))

        with st.expander("✅ 強み", expanded=True):
            if r["strengths"]:
                for x in r["strengths"]:
                    st.write(f"• {x}")
            else:
                st.write("明確な強みを自動検出できませんでした。")

        with st.expander("⚠️ 注意点", expanded=True):
            if r["cautions"]:
                for x in r["cautions"]:
                    st.write(f"• {x}")
            else:
                st.write("大きな注意点を自動検出できませんでした。")

        st.markdown("### 📈 株価・出来高チャート")
        chart_hist = r["hist"].tail(180).copy()

        # ローソク足と出来高を同じ時間軸にまとめる
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.76, 0.24]
        )

        fig.add_trace(
            go.Candlestick(
                x=chart_hist.index,
                open=chart_hist["Open"],
                high=chart_hist["High"],
                low=chart_hist["Low"],
                close=chart_hist["Close"],
                name="株価"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=chart_hist.index,
                y=chart_hist["MA25"],
                mode="lines",
                name="25日線",
                line=dict(width=1.5)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=chart_hist.index,
                y=chart_hist["MA75"],
                mode="lines",
                name="75日線",
                line=dict(width=1.5)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=chart_hist.index,
                y=chart_hist["Volume"],
                name="出来高",
                opacity=0.65
            ),
            row=2, col=1
        )

        # Entry / Stop ラインは株価パネルだけに表示
        if pd.notna(r["breakout_entry"]):
            fig.add_hline(
                y=r["breakout_entry"],
                line_dash="dash",
                annotation_text="ブレイクEntry",
                annotation_position="top left",
                row=1, col=1
            )
        if pd.notna(r["pullback_low"]):
            fig.add_hline(
                y=r["pullback_low"],
                line_dash="dot",
                annotation_text="押し目下限",
                annotation_position="bottom left",
                row=1, col=1
            )
        if pd.notna(r["pullback_high"]):
            fig.add_hline(
                y=r["pullback_high"],
                line_dash="dot",
                annotation_text="押し目上限",
                annotation_position="top left",
                row=1, col=1
            )
        if pd.notna(r["stop_price"]):
            fig.add_hline(
                y=r["stop_price"],
                line_dash="dashdot",
                annotation_text="損切り参考",
                annotation_position="bottom left",
                row=1, col=1
            )

        fig.update_layout(
            height=560,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=25, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
            bargap=0.15
        )

        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            showgrid=False,
            row=1, col=1
        )
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            showgrid=False,
            row=2, col=1
        )

        fig.update_yaxes(title_text="株価", row=1, col=1)
        fig.update_yaxes(title_text="出来高", row=2, col=1)

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "scrollZoom": True,
                "displaylogo": False
            }
        )

        st.warning(
            "V2はスクリーニング支援用です。yfinanceの財務データは欠損・更新遅延・定義差があり得ます。"
            "決算短信、決算説明資料、TDnet等の一次情報で必ず確認してください。"
        )
elif not st.session_state.analysis_has_run:
    st.info("銘柄コードを入力して「分析する」を押してください。")
