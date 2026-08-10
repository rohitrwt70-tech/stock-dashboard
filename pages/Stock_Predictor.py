import os
import re
import math
import uuid
import json
import random as _random
import datetime
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD as TA_MACD, ADXIndicator, EMAIndicator
    from ta.volatility import BollingerBands
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="Stock Predictor", layout="wide")

# ── yfinance safe wrappers (catch network timeouts gracefully) ────────────────
def _yf_info(sym: str) -> dict:
    """Fetch yfinance .info with timeout protection. Returns {} on failure."""
    try:
        return yf.Ticker(sym).info or {}
    except Exception as e:
        _em = str(e).lower()
        if any(k in _em for k in ("timeout", "timed out", "connection", "curl")):
            st.warning(f"⚠️ Yahoo Finance timed out for {sym}. Data may be incomplete — try refreshing in a moment.", icon="🌐")
        return {}

def _yf_fast_price(sym: str) -> float:
    """Fetch last price via fast_info with timeout protection. Returns 0.0 on failure."""
    try:
        return float(yf.Ticker(sym).fast_info.last_price or 0)
    except Exception:
        return 0.0

def _yf_history(sym: str, **kwargs):
    """Fetch yfinance history with timeout protection. Returns empty DataFrame on failure."""
    try:
        return yf.Ticker(sym).history(**kwargs)
    except Exception as e:
        _em = str(e).lower()
        if any(k in _em for k in ("timeout", "timed out", "connection", "curl")):
            st.warning(f"⚠️ Yahoo Finance timed out fetching history for {sym}. Try refreshing.", icon="🌐")
        import pandas as _pd_safe
        return _pd_safe.DataFrame()

# ── Sidebar (top-level) ───────────────────────────────────────────────────────
MARKETS = {
    "🇮🇳 NSE (India)": {
        "symbol": "₹",
        "default": "RELIANCE.NS",
        "exchanges": {"NSI", "BSE"},
        "placeholder": "Search e.g. Reliance, Infosys, TCS…",
    },
    "🇺🇸 NYSE / NASDAQ (US)": {
        "symbol": "$",
        "default": "AAPL",
        "exchanges": {"NMS", "NYQ", "NGM", "NCM", "ASE", "NYM", "PCX"},
        "placeholder": "Search e.g. Apple, Tesla, Microsoft…",
    },
}

default_market = st.session_state.get("shared_market_label", "🇮🇳 NSE (India)")
market_idx     = list(MARKETS.keys()).index(default_market) if default_market in MARKETS else 0

market_label = st.sidebar.selectbox("Market", list(MARKETS.keys()), index=market_idx)
market       = MARKETS[market_label]
curr         = market["symbol"]


def search_stocks(query: str):
    if not query or len(query.strip()) < 1:
        return []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "lang": "en-US", "region": "US",
                    "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=4,
        )
        quotes  = resp.json().get("quotes", [])
        allowed = market["exchanges"]
        results = []
        for q in quotes:
            if q.get("quoteType") not in ("EQUITY", "ETF"):
                continue
            if q.get("exchange") not in allowed:
                continue
            sym  = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or sym
            results.append(f"{sym} — {name}")
        return results
    except Exception:
        return []


with st.sidebar:
    _india_fallback = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","LT.NS","KOTAKBANK.NS","AXISBANK.NS","WIPRO.NS","HCLTECH.NS","SUNPHARMA.NS","MARUTI.NS","TITAN.NS","BAJFINANCE.NS","NTPC.NS","POWERGRID.NS","TATAMOTORS.NS","TATASTEEL.NS","ADANIPORTS.NS","ONGC.NS","COALINDIA.NS","DRREDDY.NS","CIPLA.NS","ZOMATO.NS","IRCTC.NS","HAL.NS","BEL.NS","TATAPOWER.NS","POLYCAB.NS","HAVELLS.NS","PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","LTIM.NS"]
    _us_fallback = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","V","MA","UNH","XOM","LLY","AVGO","JNJ","PG","HD","MRK","ABBV","COST","CVX","CRM","BAC","AMD","PEP","NFLX","KO","WMT","CSCO","MCD","ORCL","CAT","TXN","ADBE","QCOM","GS","AMGN","MS","INTU","BKNG","SNDK","PLTR","COIN","SOFI","HOOD","RIVN","SNOW","DDOG","NET","CRWD","PANW","NOW","WDAY","SHOP","RKLB","IONQ","ARM","APP","SMCI","DELL","SERV","UBER","LYFT","ABNB","DASH","RDDT","PINS","SNAP"]
    _sidebar_tickers = _us_fallback if "🇺🇸" in market_label else _india_fallback
    raw = st.selectbox(
        "Search Stock",
        options=[""] + _sidebar_tickers,
        index=0,
        placeholder=market["placeholder"],
        key=f"pred_search_{market_label}",
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Risk Settings")
    _acct_size = st.number_input(
        "Account size ($)",
        min_value=1000, max_value=10_000_000,
        value=st.session_state.get("acct_size", 10000),
        step=1000, format="%d",
        key="acct_size_input",
        help="Your total trading capital",
    )
    st.session_state["acct_size"] = _acct_size
    _risk_pct = st.slider(
        "Max risk per trade (%)",
        min_value=0.5, max_value=3.0,
        value=st.session_state.get("risk_pct", 1.0),
        step=0.5, format="%.1f%%",
        key="risk_pct_input",
        help="1% = lose max 1% of account on any single trade",
    )
    st.session_state["risk_pct"] = _risk_pct
    _max_loss_usd = _acct_size * _risk_pct / 100
    st.caption(f"Max loss per trade: **${_max_loss_usd:,.0f}**")

if raw and " — " in raw:
    ticker = raw.split(" — ")[0].strip()
elif raw:
    ticker = raw.strip()
else:
    ticker = st.session_state.get("shared_ticker", market["default"])


@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def load_price_data(sym):
    return _yf_history(sym, period="2y")


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR        = Path("./screener_models")
PREDICTIONS_FILE = Path("./screener_predictions.json")
UNIVERSE_CACHE   = Path("./screener_universe_cache.json")
WATCHLIST_FILE   = Path("./watchlist.json")
PORTFOLIO_FILE   = Path("./portfolio_holdings.json")
PAPER_TRADES_FILE = Path("./paper_trades.json")
PRICE_HISTORY_FILE = Path("./price_history.json")
VALIDATION_LOG_FILE = Path("./validation_log.json")
CALIBRATION_FILE   = Path("./model_calibration.json")
MODEL_DIR.mkdir(exist_ok=True)

BUCKET_CONFIG = {
    "anchor": {
        "name":         "🔒 Anchor",
        "allocation":   0.30,
        "count":        5,
        "color":        "#00c853",
        "bg":           "#002010",
        "border":       "#00c853",
        "target_min":   0.18,
        "target_max":   0.25,
        "horizon":      "1 Year",
        "horizon_days": 252,
        "rebalance":    "Quarterly",
        "max_beta":     1.0,
        "description":  "Capital preservation + steady 18-25% annual returns. Sleep-well-at-night stocks.",
    },
    "growth": {
        "name":         "📈 Growth",
        "allocation":   0.40,
        "count":        8,
        "color":        "#2979ff",
        "bg":           "#001233",
        "border":       "#2979ff",
        "target_min":   0.30,
        "target_max":   0.45,
        "horizon":      "6 Months",
        "horizon_days": 126,
        "rebalance":    "Monthly",
        "max_beta":     1.5,
        "description":  "Sector momentum + earnings acceleration. 30-45% in 6 months.",
    },
    "rotational": {
        "name":         "🚀 Rotational",
        "allocation":   0.30,
        "count":        8,
        "color":        "#ff9100",
        "bg":           "#1a0d00",
        "border":       "#ff9100",
        "target_min":   0.40,
        "target_max":   0.80,
        "horizon":      "3 Months",
        "horizon_days": 63,
        "rebalance":    "Weekly",
        "max_beta":     3.0,
        "description":  "High-momentum breakout plays for capital rotation. 40-80% in 3 months.",
    },
}

VIX_REGIMES = {
    # Counts = max stocks shown per bucket in each market regime
    "calm":    {"max": 15,  "anchor": 8, "growth": 10, "rotational": 8, "ondeck": 6},
    "normal":  {"max": 25,  "anchor": 7, "growth": 8,  "rotational": 7, "ondeck": 5},
    "fearful": {"max": 35,  "anchor": 7, "growth": 5,  "rotational": 3, "ondeck": 4},
    "crisis":  {"max": 999, "anchor": 7, "growth": 2,  "rotational": 0, "ondeck": 3},
}

INDIA_THEMES = [
    "Quantum Computing India", "Defence Technology India PLI",
    "EV Battery Components India", "Specialty Chemicals India Export",
    "Contract Manufacturing Electronics India", "Space Technology India ISRO",
    "Agri Tech India", "Water Treatment Infrastructure India",
    "Semiconductor India Fab", "Renewable Energy Storage India",
    "Diagnostics Healthcare India Tier2", "Logistics Supply Chain India",
    "Digital Lending Fintech India", "Gaming Esports India",
    "Hotel Tourism India Recovery", "Railway Infrastructure India",
    "Cold Chain Agriculture India", "Drone Technology India",
    "Smart Meter AMI India", "Generic Pharma API Export India",
]

US_THEMES = [
    "Quantum Computing Hardware", "AI Infrastructure Data Centers",
    "Nuclear Energy Small Modular Reactor", "Defence Cybersecurity",
    "Space Technology Commercial", "Humanoid Robotics",
    "GLP-1 Drug Supply Chain", "Water Technology Desalination",
    "Carbon Capture Technology", "Rare Earth Mining Processing",
    "Autonomous Vehicle Sensors", "Precision Agriculture Technology",
    "Gene Editing Biotech", "Satellite Internet Infrastructure",
    "Battery Recycling Technology", "Neuromorphic Computing",
    "Vertical Farming Technology", "Hydrogen Fuel Cell Infrastructure",
    "Anti-Aging Longevity Biotech", "Photonics Optical Computing",
]

# Standard GICS sectors + India-specific additions for industry filter
INDUSTRY_OPTIONS = [
    "Technology", "Information Technology", "Software",
    "Healthcare", "Pharmaceuticals", "Biotechnology",
    "Financials", "Banking", "Insurance",
    "Consumer Discretionary", "Consumer Staples", "Retail", "FMCG",
    "Industrials", "Manufacturing", "Defence",
    "Energy", "Oil & Gas", "Renewable Energy",
    "Materials", "Chemicals", "Metals & Mining",
    "Utilities", "Real Estate",
    "Communication Services", "Media & Entertainment",
    "Auto & Auto Components", "IT Services",
    "Logistics & Supply Chain", "Infrastructure",
]

DISCOVERY_LABELS = {
    "undiscovered": ("🔭 UNDISCOVERED", "#9c27b0", "#1a0033"),
    "hidden_gem":   ("💎 HIDDEN GEM",   "#00bcd4", "#001a1f"),
    "theme_play":   ("🚀 THEME PLAY",   "#ff9100", "#1a0d00"),
    "momentum":     ("📈 MOMENTUM",     "#00c853", "#002010"),
    "quality":      ("🏦 QUALITY",      "#2979ff", "#001233"),
}

# ── India universe (Nifty 500 representative) ─────────────────────────────────
INDIA_UNIVERSE = [
    # Nifty 50
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","NTPC.NS","POWERGRID.NS","NESTLEIND.NS",
    "WIPRO.NS","HCLTECH.NS","TECHM.NS","SUNPHARMA.NS","DRREDDY.NS",
    "CIPLA.NS","ONGC.NS","COALINDIA.NS","ADANIPORTS.NS","ULTRACEMCO.NS",
    "GRASIM.NS","TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
    "INDUSINDBK.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","M&M.NS",
    "BPCL.NS","BRITANNIA.NS","DIVISLAB.NS","HDFCLIFE.NS","SBILIFE.NS",
    "APOLLOHOSP.NS","DMART.NS","PIDILITIND.NS","TATACONSUM.NS","MUTHOOTFIN.NS",
    # Midcap
    "PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","LTIM.NS","ZOMATO.NS",
    "IRCTC.NS","IRFC.NS","RVNL.NS","HAL.NS","BEL.NS","BHEL.NS",
    "TATAPOWER.NS","TORNTPOWER.NS","POLYCAB.NS","HAVELLS.NS","VOLTAS.NS",
    "ASTRAL.NS","SUPREMEIND.NS","ESCORTS.NS","TIINDIA.NS","CUMMINSIND.NS",
    "THERMAX.NS","BALKRISIND.NS","MRF.NS","APOLLOTYRE.NS","PIIND.NS",
    "AARTIIND.NS","DEEPAKNITR.NS","NAVINFLUO.NS","TATACHEM.NS","BIOCON.NS",
    "AUROPHARMA.NS","ALKEM.NS","TORNTPHARM.NS","LAURUSLABS.NS","GRANULES.NS",
    "METROPOLIS.NS","KIMS.NS","MAXHEALTH.NS","KALYANKJIL.NS","SENCO.NS",
    "ANGELONE.NS","CDSL.NS","MOTILALOFS.NS","EMAMILTD.NS","MARICO.NS",
    "GODREJCP.NS","DABUR.NS","COLPAL.NS","PGHH.NS","BERGEPAINT.NS",
    # Smallcap / Emerging
    "TANLA.NS","NEWGEN.NS","KPITTECH.NS","TATAELXSI.NS","ZENSAR.NS",
    "KAYNES.NS","AMBER.NS","APLAPOLLO.NS","GRAVITA.NS","RATEGAIN.NS",
    "INDIAMART.NS","JUSTDIAL.NS","POLICYBZR.NS","NYKAA.NS","DELHIVERY.NS",
    "PAYTM.NS","CAMPUS.NS","BIKAJI.NS","HAPPSTMNDS.NS","LTTS.NS",
    "CYIENT.NS","MASTEK.NS","QUICKHEAL.NS","NAZARA.NS","OPTIEMUS.NS",
    "RAILVIKAS.NS","RITES.NS","IRCON.NS","NBCC.NS","HFCL.NS",
    "CLEAN.NS","FLUOROCHEM.NS","ALKYLAMINE.NS","GNFC.NS","PCBL.NS",
    "SUMICHEM.NS","VINDHYATEL.NS","STLTECH.NS","TEJASNET.NS","RAILTEL.NS",
]

# ── Universe fetchers ─────────────────────────────────────────────────────────
def get_india_universe():
    try:
        nse_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        r = requests.get(nse_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col:
            tickers = [s.strip() + ".NS" for s in df[sym_col].dropna().tolist()]
            if len(tickers) > 50:
                return tickers
    except Exception:
        pass
    return INDIA_UNIVERSE


@st.cache_data(ttl=86400)
def get_us_universe():
    # Primary: SEC EDGAR — all ~10k US-listed companies, no API key needed
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "StockDashboard rohitrwt70@gmail.com"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            tickers = sorted({v["ticker"] for v in data.values() if v.get("ticker")})
            if len(tickers) > 1000:
                return tickers
    except Exception:
        pass

    # Fallback: S&P 500 core
    sp500 = []
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        sp500_df = pd.read_csv(url, timeout=8)
        sp500 = sp500_df["Symbol"].tolist()
    except Exception:
        pass

    # Comprehensive US ticker list: S&P 500, Nasdaq 100, Russell 1000 leaders,
    # popular mid/small caps, sector ETFs, and recent spinoffs/IPOs
    builtin = [
        # Mega cap tech
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ORCL",
        "CRM","ADBE","QCOM","AMD","INTC","TXN","MU","KLAC","LRCX","AMAT",
        "ASML","TSM","SNPS","CDNS","ANSS","PTC","ANSYS","FTNT","ZBRA","FFIV",
        # Semiconductors
        "NVDA","AMD","INTC","QCOM","TXN","MU","KLAC","LRCX","AMAT","MRVL",
        "MPWR","ON","SWKS","QRVO","WOLF","AMBA","CRUS","SLAB","DIOD","POWI",
        "SIMO","FORM","ACLS","COHU","ICHR","MKSI","UCTT","AXTI","AOSL","RMBS",
        "SNDK","WDC","STX","NTAP","PSTG","DELL","HPE","HPQ","NCR","CDW",
        # Storage & hardware
        "SNDK","WDC","STX","NTAP","PSTG","DELL","HPE","HPQ","IBM","EMC",
        # Financials
        "JPM","BAC","WFC","GS","MS","C","BLK","AXP","V","MA",
        "COF","DFS","SYF","ALLY","LC","SOFI","HOOD","COIN","MSTR","PYPL",
        "SQ","AFRM","UPST","OPEN","OPFI","ENOVA","QFIN","FUTU","TIGR","LU",
        "ICE","CME","NDAQ","CBOE","MKTX","LPLA","RJF","SF","STIFEL","PIPR",
        "BRK-B","BRK-A","MET","PRU","LNC","UNM","AFL","GL","PFG","VOYA",
        "AIG","ALL","PGR","TRV","CB","HIG","CNA","WRB","CINF","MKL",
        # Healthcare & Biotech
        "UNH","JNJ","LLY","ABBV","MRK","PFE","BMY","AMGN","GILD","BIIB",
        "REGN","VRTX","MRNA","BNTX","NVAX","SGEN","ALNY","IONS","SRPT","RARE",
        "BLUE","EDIT","CRSP","BEAM","NTLA","FATE","PACB","RXRX","AGEN","IMMU",
        "TMO","DHR","A","BIO","ILMN","HOLX","BAX","BDX","COO","HSIC",
        "MDT","BSX","EW","SYK","ZBH","ABMD","ISRG","DXCM","IDXX","PODD",
        "HCA","THC","UHS","CYH","ENSG","NHC","ACAD","JAZZ","INCY","EXEL",
        "CVS","WBA","RAD","CANO","OSCR","ACMR","ACHC","HUM","ELV","CNC",
        # Consumer
        "AMZN","WMT","COST","TGT","HD","LOW","TJX","ROST","BURL","OLLI",
        "NKE","LULU","UA","UAA","SKX","CROX","DECK","BOOT","ONON","HBI",
        "MCD","SBUX","YUM","QSR","DPZ","PZZA","WING","SHAK","TXRH","DINE",
        "PG","KO","PEP","MDLZ","GIS","K","CAG","CPB","MKC","SJM",
        "CL","CHD","ELF","ULTA","COTY","REVG","IPAR","EL","SBH","PRGO",
        "GM","F","STLA","TM","HMC","NIO","XPEV","LI","RIVN","LCID",
        "TSLA","FSR","NKLA","WKHS","RIDE","GOEV","SOLO","AYRO","HYLN","CEV",
        # Energy
        "XOM","CVX","COP","EOG","PXD","DVN","FANG","MRO","APA","HES",
        "SLB","HAL","BKR","FTI","NOV","RIG","VAL","NE","DO","OIS",
        "PSX","VLO","MPC","PBF","HFC","DKL","CVRR","PARR","CLMT","CAPL",
        "OKE","WMB","KMI","ET","EPD","MMP","MPLX","PAA","SHLX","NS",
        "LNG","TELL","NEXT","GLNG","GMLP","COOL","AR","RRC","SWN","CNX",
        # Industrials
        "CAT","DE","EMR","HON","GE","RTX","LMT","NOC","GD","BA",
        "HII","TDG","HEI","AXON","LDOS","SAIC","BAH","CACI","PAE","KTOS",
        "UPS","FDX","XPO","CHRW","EXPD","JBHT","LSTR","ODFL","SAIA","WERN",
        "CSX","NSC","UNP","CP","CNI","KSU","GATX","RAIL","TRN","GBX",
        "MMM","ITW","PH","ROK","AME","FTV","ROP","IEX","FLOW","XYL",
        "GNRC","REXNORD","WTS","WATTS","FELE","AAON","AIRC","AWI","APOG","TREX",
        # REITs
        "AMT","PLD","EQIX","CCI","DLR","PSA","EXR","CUBE","LSI","NSA",
        "SPG","O","NNN","ADC","STOR","EPRT","GTY","AGREE","NETSTREIT","PINE",
        "EQR","AVB","ESS","MAA","CPT","UDR","NVR","DHI","LEN","PHM",
        "TOL","MDC","TMHC","KBH","MHO","SGH","BZH","HOV","WLH","LGIH",
        # Utilities
        "NEE","DUK","SO","D","AEP","EXC","XEL","ES","WEC","ETR",
        "PPL","FE","CNP","AES","NRG","PNW","POR","CLNE","BE","FCEL",
        # Materials
        "LIN","APD","DD","EMN","PPG","SHW","RPM","AXTA","HXL","KWR",
        "NEM","GOLD","AEM","KGC","HL","PAAS","AG","CDE","EXK","SILV",
        "FCX","TECK","AA","CENX","KALU","ARNC","ATI","HWM","MTRN","MXCY",
        "NUE","STLD","CLF","CMC","RS","WOR","ZEUS","HZNP","TS","GGB",
        # Cloud & SaaS
        "SNOW","MDB","DDOG","NET","HUBS","ZS","OKTA","CRWD","S","PANW",
        "SMAR","COUP","PLAN","PCTY","PAYC","WK","APPN","PEGA","BPTH","FROG",
        "TWLO","ZI","BILL","MNTV","BRZE","AMPL","SPRK","RDVT","WEAV","CFLT",
        "GTLB","SDLP","ESTC","SUMO","NEWR","DT","FSLY","FIVN","NICE","NICE",
        "NOW","WDAY","VEEV","INTU","ADSK","ANSS","PTC","MANH","EPAM","GLOB",
        # Cybersecurity
        "CRWD","PANW","FTNT","ZS","OKTA","S","TENB","RPD","VRNT","QLYS",
        "CYBR","SAIL","SSTK","CSCO","CHKP","PFPT","MIME","OSPN","EVBG","SCWX",
        # Digital media & streaming
        "NFLX","DIS","PARA","WBD","CMCSA","FOX","FOXA","AMCX","LGF-A","LGF-B",
        "SPOT","IHRT","SIRI","LSXMA","LSXMB","LSXMK","FWONA","FWONK","BATRK","BATRA",
        "RBLX","U","EA","TTWO","ATVI","NTDOY","GME","GAMB","DKNG","PENN",
        # E-commerce & marketplace
        "AMZN","SHOP","EBAY","ETSY","WISH","POSH","REAL","RENT","RDFN","OPEN",
        "ABNB","BKNG","EXPE","TRIP","LIND","VCNX","YELP","ANGI","IAC","MTCH",
        # EV & clean energy
        "TSLA","RIVN","LCID","NIO","XPEV","LI","NKLA","RIDE","GOEV","WKHS",
        "ENPH","SEDG","FSLR","RUN","NOVA","ARRY","SPWR","NEP","BEPC","CWEN",
        "BLNK","EVGO","CHPT","SBE","HYLN","SOLO","AYRO","KNDI","IDEX","WORX",
        "PLUG","FCEL","BLOOM","BE","BLDP","ITM","NEL","HTOO","HPNN","HYSR",
        # Space & defense tech
        "RKLB","ASTS","ASTR","SPIR","MNTS","KTOS","MAXR","DigitalBridge","SATL","OSAT",
        "LMT","NOC","RTX","GD","BA","HII","TDG","HEI","LDOS","SAIC",
        "AXON","OSIS","VSE","DRS","CACI","BAH","KEYW","MANT","FLIR","DRS",
        # AI & machine learning
        "NVDA","AMD","INTC","GOOGL","MSFT","META","AMZN","IBM","PLTR","AI",
        "BBAI","SOUN","GFAI","DTCK","NABL","EXAI","AEYE","OPAD","BDAI","AIOT",
        "SMCI","DELL","HPE","NTAP","PSTG","NTNX","PYCR","NCNO","RDWR","CEVA",
        # Biotech & genomics
        "ILMN","PACB","RXRX","SEER","OMIC","NVTA","GNMK","FLGT","EXAS","NTRA",
        "GH","CDNA","VCNX","FATE","BLUE","CRSP","EDIT","BEAM","NTLA","ALNY",
        "IONS","SRPT","RARE","ACAD","JAZZ","INCY","EXEL","HALO","MRUS","ARQT",
        # Finance & fintech
        "PYPL","SQ","AFRM","UPST","SOFI","HOOD","COIN","MSTR","LC","OPEN",
        "OPFI","ENOVA","DAVE","MQ","PAYO","DH","FLYW","PLTK","IIIV","RPAY",
        "NRDS","PRFT","GTPAY","RELY","GREE","CURO","WRLD","EZCORP","FCFS","QCR",
        # Recent IPOs & spinoffs (2023-2025)
        "SNDK","ARM","BIRK","CART","KVYO","CLBT","IREN","MARA","RIOT","HUT",
        "BTDR","CIFR","CORZ","WULF","BTBT","SDIG","HIVE","DGHI","POWI","POWL",
        "APP","APGE","ALAB","ACHR","JOBY","LILM","EVE","BLDE","BETA","SKYH",
        "IONQ","QUBT","RGTI","QBTS","IQM","ARQQ","CNXT","SPAC","PSFE","GREE",
        # Consumer tech & devices
        "AAPL","MSFT","GOOGL","AMZN","META","SNAP","PINS","TWTR","RDDT","MTCH",
        "ZM","DOCU","DOCN","BOX","DRPBX","WORK","TEAM","ATLS","FROG","CFLT",
        "SONO","HEAR","KOSS","GPRO","ROKU","VUZI","IMMR","MKSI","UEIC","SMTC",
        # Retail & restaurants
        "COST","WMT","TGT","KR","SFM","WINN","CHEF","GO","UNFI","SPTN",
        "MCD","SBUX","YUM","QSR","DPZ","PZZA","WING","SHAK","TXRH","FAT",
        "CHUY","BJRI","DINE","EAT","DRI","CAKE","BLMN","RUTH","FWRG","ARCO",
        "LULU","NKE","UA","SKX","CROX","DECK","BOOT","ONON","CURV","WHLM",
        # Telecoms & infrastructure
        "T","VZ","TMUS","LUMN","FYBR","CCOI","SHEN","ATNI","OOMA","BAND",
        "AMT","CCI","SBAC","SBA","UNIT","UNITI","ATUS","CABO","LBRDA","CHTR",
        "DISH","SATS","VSAT","GSAT","ORBC","GILT","IRDM","MAXR","SPOK","NTGR",
        # Misc popular tickers
        "GME","AMC","BB","BBBY","KOSS","EXPR","CLOV","WKHS","RIDE","NKLA",
        "WISH","CLOV","MVIS","OCGN","ATOS","HIMS","BODY","FATH","OWLET","BARK",
        "SPCE","RKT","UWMC","GHVI","PSTH","AJAX","COVA","GS","IPOF","FTIV",
    ]

    all_tickers = list(dict.fromkeys(sp500 + builtin))
    # Clean up any non-string or empty entries
    return [t for t in all_tickers if isinstance(t, str) and t.strip()]


# ── Live VIX ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_live_vix():
    try:
        vix_df = yf.Ticker("^VIX").history(period="2d")
        vix_val = float(vix_df["Close"].dropna().iloc[-1])
    except Exception:
        vix_val = 18.0
    for regime, cfg in VIX_REGIMES.items():
        if vix_val <= cfg["max"]:
            return vix_val, regime
    return vix_val, "crisis"


# ── Single-stock analyser ─────────────────────────────────────────────────────
def analyze_stock(ticker_sym, curr_sym):
    """Returns dict of metrics or None on any failure. Never raises."""
    try:
        df = yf.Ticker(ticker_sym).history(period="6mo")
        if df.empty or len(df) < 30:
            return None

        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        vol   = df["Volume"].values.astype(float)
        price = close[-1]

        s = pd.Series(close)

        # RSI
        delta    = s.diff()
        gain     = delta.clip(lower=0).rolling(14).mean()
        loss     = (-delta.clip(upper=0)).rolling(14).mean()
        rs       = gain / loss
        rsi_s    = 100 - (100 / (1 + rs))
        rsi_now  = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

        # MACD
        ema12      = s.ewm(span=12, adjust=False).mean()
        ema26      = s.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        sig_line   = macd_line.ewm(span=9, adjust=False).mean()
        macd_now   = float(macd_line.iloc[-1])
        sig_now    = float(sig_line.iloc[-1])
        macd_bull  = macd_now > sig_now and macd_now > 0

        # MAs
        ma50   = float(s.rolling(50).mean().iloc[-1])
        ma200  = float(s.rolling(min(200, len(s))).mean().iloc[-1])
        ema20  = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50v = float(s.ewm(span=50, adjust=False).mean().iloc[-1])

        above_ma50  = price > ma50
        above_ma200 = price > ma200
        ema_bull    = ema20 > ema50v

        # Volume surge vs 20-day average
        vol_s     = pd.Series(vol)
        vol_20avg = float(vol_s.rolling(20).mean().iloc[-1])
        vol_surge = float(vol[-1] / vol_20avg) if vol_20avg > 0 else 1.0

        # 52W range
        high_52w      = float(max(high))
        low_52w       = float(min(low))
        pct_from_high = (high_52w - price) / high_52w * 100 if high_52w > 0 else 0.0

        # Bollinger %B
        bb_mid = float(s.rolling(20).mean().iloc[-1])
        bb_std = float(s.rolling(20).std().iloc[-1])
        bb_hi  = bb_mid + 2 * bb_std
        bb_lo  = bb_mid - 2 * bb_std
        bb_rng = bb_hi - bb_lo
        bb_pct = float((price - bb_lo) / bb_rng) if bb_rng > 0 else 0.5

        # ATR
        ph = pd.Series(high); pl = pd.Series(low); pc = pd.Series(close).shift()
        tr  = pd.concat([ph - pl, (ph - pc).abs(), (pl - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # Beta vs index (30-day correlation approximation)
        beta = 1.0
        try:
            idx_sym = "^NSEI" if ".NS" in ticker_sym else "^GSPC"
            idx_df  = yf.Ticker(idx_sym).history(period="3mo")
            if not idx_df.empty:
                stk_ret = pd.Series(close).pct_change().dropna()
                idx_ret = idx_df["Close"].pct_change().dropna()
                min_len = min(len(stk_ret), len(idx_ret))
                if min_len > 20:
                    beta_val, _, _, _, _ = stats.linregress(
                        idx_ret.values[-min_len:], stk_ret.values[-min_len:]
                    )
                    beta = float(beta_val)
        except Exception:
            pass

        # ── Composite score ───────────────────────────────────────────────────
        # RSI (sweet-spot 45–65 is best)
        if 45 <= rsi_now <= 65:    rsi_score = 85
        elif 35 <= rsi_now < 45:   rsi_score = 65
        elif 65 < rsi_now <= 75:   rsi_score = 60
        elif rsi_now < 35:         rsi_score = 45  # oversold — potential hidden gem
        else:                       rsi_score = 25  # overbought

        macd_score = 80 if macd_bull else (50 if macd_now > sig_now else 20)

        if above_ma50 and above_ma200:     ma_score = 90
        elif above_ma50:                   ma_score = 65
        elif above_ma200:                  ma_score = 50
        else:                              ma_score = 15

        vol_score = min(100.0, vol_surge * 40)

        mom_rng   = high_52w - low_52w
        mom_score = float((price - low_52w) / mom_rng * 100) if mom_rng > 0 else 50.0

        bb_score  = bb_pct * 100
        ema_score = 80 if ema_bull else 25

        composite = (
            rsi_score  * 0.15 +
            macd_score * 0.20 +
            ma_score   * 0.20 +
            vol_score  * 0.15 +
            mom_score  * 0.15 +
            bb_score   * 0.10 +
            ema_score  * 0.05
        )

        # Entry & stop
        entry     = round(price * 0.990, 2)
        stop_loss = round(entry - 1.5 * atr, 2)

        # Expected return % (rough: composite → return mapping)
        exp_ret = round((composite - 50) * 0.8, 1)  # 80% score ≈ 24% expected return

        # Top 3 human-readable signals
        raw_signals = []
        if macd_bull:               raw_signals.append((macd_score, f"MACD bullish ({macd_now:+.3f})"))
        if rsi_now < 35:            raw_signals.append((70, f"RSI {rsi_now:.1f} — oversold, potential bounce"))
        if rsi_now > 55 and rsi_now < 70: raw_signals.append((rsi_score, f"RSI {rsi_now:.1f} — strong momentum zone"))
        if vol_surge > 1.5:         raw_signals.append((vol_score, f"Volume surge {vol_surge:.1f}× 20-day avg"))
        if above_ma200:             raw_signals.append((60, "Price above MA200 — long-term uptrend"))
        if above_ma50:              raw_signals.append((55, "Price above MA50 — medium-term support"))
        if ema_bull:                raw_signals.append((60, "EMA20 > EMA50 — short-term momentum bullish"))
        if bb_pct > 0.65:           raw_signals.append((50, f"Bollinger %B {bb_pct*100:.0f}% — strong position"))
        if pct_from_high < 5:       raw_signals.append((75, f"Near 52W high ({pct_from_high:.1f}% away) — breakout zone"))
        if not above_ma200 and rsi_now < 40: raw_signals.append((65, "Below MA200 + oversold — hidden gem candidate"))

        top3 = [sig for _, sig in sorted(raw_signals, key=lambda x: -x[0])[:3]]

        return {
            "ticker":        ticker_sym,
            "price":         round(float(price), 2),
            "rsi":           round(rsi_now, 1),
            "macd_bull":     bool(macd_bull),
            "above_ma50":    bool(above_ma50),
            "above_ma200":   bool(above_ma200),
            "ema_bull":      bool(ema_bull),
            "vol_surge":     round(vol_surge, 2),
            "pct_from_high": round(pct_from_high, 1),
            "bb_pct":        round(bb_pct, 3),
            "atr":           round(atr, 2),
            "beta":          round(beta, 2),
            "composite":     round(composite, 1),
            "entry":         entry,
            "stop_loss":     stop_loss,
            "exp_ret":       exp_ret,
            "top_signals":   top3,
            "curr":          curr_sym,
        }
    except Exception:
        return None


# ── Discovery label assignment ────────────────────────────────────────────────
THEME_KEYWORDS = {
    "Technology": "theme_play", "Information Technology": "theme_play",
    "Healthcare": "theme_play", "Industrials": "theme_play",
    "Consumer Discretionary": "momentum", "Energy": "theme_play",
    "Materials": "theme_play", "Utilities": "quality",
    "Consumer Staples": "quality", "Financials": "quality",
    "Real Estate": "quality",
}

def assign_discovery_label(s):
    comp    = s.get("composite", 50)
    rsi     = s.get("rsi", 50)
    beta    = s.get("beta", 1.0)
    am200   = s.get("above_ma200", False)
    sector  = s.get("sector", "")

    if not am200 and rsi < 40:                      return "hidden_gem"
    if comp > 70 and beta < 1.2:                    return "undiscovered"
    if beta < 0.8 and am200:                        return "quality"
    if rsi > 60 and s.get("macd_bull") and am200:   return "momentum"
    if sector in THEME_KEYWORDS:
        return THEME_KEYWORDS.get(sector, "momentum")
    return "momentum"


# ── Bucket assignment ─────────────────────────────────────────────────────────
def assign_to_buckets(stocks, vix_regime):
    cfg = VIX_REGIMES.get(vix_regime, VIX_REGIMES["normal"])

    anchor_pool = [s for s in stocks
                   if s["beta"] <= BUCKET_CONFIG["anchor"]["max_beta"]
                   and s["above_ma200"]
                   and 45 <= s["rsi"] <= 65]
    growth_pool = [s for s in stocks
                   if s["macd_bull"]
                   and s["above_ma50"]
                   and s["vol_surge"] >= 1.2
                   and 50 <= s["rsi"] <= 72]
    rotat_pool  = [s for s in stocks
                   if s["composite"] >= 65
                   and s["pct_from_high"] <= 10
                   and s["beta"] >= 1.3]

    anchor_pool.sort(key=lambda x: -x["composite"])
    growth_pool.sort(key=lambda x: -x["composite"])
    rotat_pool.sort(key=lambda x: -x["composite"])

    return {
        "anchor":     anchor_pool[: cfg["anchor"]],
        "growth":     growth_pool[: cfg["growth"]],
        "rotational": rotat_pool[: cfg["rotational"]],
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────
def save_cache(results, buckets, market_key, vix_val, regime):
    payload = {
        "ts":      pd.Timestamp.now().isoformat(),
        "market":  market_key,
        "vix":     vix_val,
        "regime":  regime,
        "stocks":  results,
        "buckets": {k: v for k, v in buckets.items()},
    }
    try:
        PREDICTIONS_FILE.write_text(json.dumps(payload, default=str))
    except Exception:
        pass


def load_cache(market_key):
    try:
        if not PREDICTIONS_FILE.exists():
            return None
        data = json.loads(PREDICTIONS_FILE.read_text())
        if data.get("market") != market_key:
            return None
        ts   = pd.Timestamp(data["ts"])
        age  = (pd.Timestamp.now() - ts).total_seconds()
        if age > 86400:   # 24 hours
            return None
        return data
    except Exception:
        return None


# ── Full scan ─────────────────────────────────────────────────────────────────
def run_scan(tickers, curr_sym, progress_bar, status_txt, time_left_txt):
    results  = []
    total    = len(tickers)
    BATCH    = 20

    for batch_start in range(0, total, BATCH):
        batch = tickers[batch_start: batch_start + BATCH]
        with ThreadPoolExecutor(max_workers=BATCH) as ex:
            futs = {ex.submit(analyze_stock, t, curr_sym): t for t in batch}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                except Exception:
                    pass

        done     = min(batch_start + BATCH, total)
        pct      = done / total
        remaining = total - done
        secs_left = int(remaining * 0.6)
        mins_left, s_left = divmod(secs_left, 60)

        progress_bar.progress(pct)
        status_txt.markdown(
            f"**Scanned:** {done} / {total} stocks &nbsp;·&nbsp; "
            f"**Found:** {len(results)} candidates &nbsp;·&nbsp; "
            f"**Last batch:** {', '.join(batch[:3])}…"
        )
        time_left_txt.caption(
            f"⏱ Estimated time remaining: {mins_left}m {s_left:02d}s"
        )

    return results


# ── Stock card renderer ───────────────────────────────────────────────────────
def render_card(s, bucket_key):
    bc   = BUCKET_CONFIG[bucket_key]
    dlbl, dfg, dbg = DISCOVERY_LABELS.get(s.get("discovery", "momentum"),
                                           DISCOVERY_LABELS["momentum"])
    signals_html = "".join(
        f"<div style='color:#aaa;font-size:0.78rem;margin:2px 0'>• {sig}</div>"
        for sig in s.get("top_signals", [])
    )
    exp_color = "#00c853" if s.get("exp_ret", 0) >= 0 else "#ff5252"
    stop_pct  = ((s["entry"] - s["stop_loss"]) / s["entry"] * 100) if s["entry"] else 0

    return f"""
    <div style="border:1px solid {bc['border']};border-radius:10px;
        background:{bc['bg']};padding:16px 18px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div>
          <span style="font-size:1.2rem;font-weight:800;color:#fff">{s['ticker'].replace('.NS','')}</span>
          <span style="margin-left:10px;font-size:0.75rem;background:{dbg};
            color:{dfg};border:1px solid {dfg};padding:2px 8px;border-radius:4px">{dlbl}</span>
        </div>
        <div style="text-align:right">
          <div style="font-size:1.1rem;font-weight:700;color:#fff">
            {s['curr']}{s['price']:.2f}</div>
          <div style="font-size:0.8rem;color:{bc['color']}">
            Score: {s['composite']:.1f}</div>
        </div>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap">
        <div><span style="color:#888;font-size:0.72rem">RSI</span>
          <div style="color:#fff;font-weight:600">{s['rsi']:.1f}</div></div>
        <div><span style="color:#888;font-size:0.72rem">Beta</span>
          <div style="color:#fff;font-weight:600">{s['beta']:.2f}</div></div>
        <div><span style="color:#888;font-size:0.72rem">Vol Surge</span>
          <div style="color:#fff;font-weight:600">{s['vol_surge']:.1f}×</div></div>
        <div><span style="color:#888;font-size:0.72rem">From 52W High</span>
          <div style="color:#fff;font-weight:600">{s['pct_from_high']:.1f}%</div></div>
        <div><span style="color:#888;font-size:0.72rem">Exp. Return</span>
          <div style="color:{exp_color};font-weight:700">{s['exp_ret']:+.1f}%</div></div>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:10px">
        <div><span style="color:#888;font-size:0.72rem">Entry</span>
          <div style="color:#00c853;font-weight:700">{s['curr']}{s['entry']:.2f}</div></div>
        <div><span style="color:#888;font-size:0.72rem">Stop Loss</span>
          <div style="color:#ff5252;font-weight:700">
            {s['curr']}{s['stop_loss']:.2f}
            <span style="color:#888;font-size:0.7rem"> ({stop_pct:.1f}%↓)</span>
          </div></div>
        <div><span style="color:#888;font-size:0.72rem">Horizon</span>
          <div style="color:{bc['color']};font-weight:600">{bc['horizon']}</div></div>
        <div><span style="color:#888;font-size:0.72rem">Rebalance</span>
          <div style="color:#aaa;font-weight:600">{bc['rebalance']}</div></div>
      </div>
      <div>{signals_html}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1 — VIX Fetch
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def fetch_vix():
    """Fetch current VIX. Returns float."""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        return float(vix["Close"].dropna().iloc[-1])
    except Exception:
        return 18.0


def get_vix_regime(vix_val):
    """Return (regime_name, regime_config) for current VIX."""
    for name, cfg in VIX_REGIMES.items():
        if vix_val <= cfg["max"]:
            return name, cfg
    return "crisis", VIX_REGIMES["crisis"]


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTOR  — multi-signal classification: BULL/HIGH_VOL/BEAR/CRISIS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def detect_market_regime(market_key="US"):
    """
    Classify the current market environment using 5 signals:
      1. VIX level        — fear gauge
      2. Index vs 200MA   — structural trend
      3. 3M momentum      — intermediate direction
      4. 20-day realised vol — turbulence level
      5. 50/200MA cross   — golden / death cross

    Returns a dict with regime name, display values, and adjustment multipliers
    that feed into Kelly sizing and MC drift.
    """
    try:
        idx_sym = "^NSEI" if market_key == "IN" else "^GSPC"
        idx_df  = yf.Ticker(idx_sym).history(period="1y",  interval="1d")
        vix_df  = yf.Ticker("^VIX").history(period="3mo", interval="1d")

        if idx_df.empty:
            raise ValueError("No index data")

        close    = idx_df["Close"].dropna()
        vix_curr = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 20.0

        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else float(close.mean())
        cur   = float(close.iloc[-1])
        vs200 = (cur - ma200) / ma200
        mom3m = (cur / float(close.iloc[-63]) - 1) if len(close) >= 63 else 0.0

        daily_rets = close.pct_change().dropna()
        vol20 = float(daily_rets.tail(20).std() * np.sqrt(252)) if len(daily_rets) >= 20 else 0.20
        golden = ma50 > ma200

        # Composite bull/bear score
        s = 0
        if   vix_curr < 15:   s += 2
        elif vix_curr < 20:   s += 1
        elif vix_curr > 40:   s -= 4
        elif vix_curr > 30:   s -= 2
        elif vix_curr > 22:   s -= 1

        if   vs200 >  0.05:   s += 2
        elif vs200 >  0:      s += 1
        elif vs200 < -0.15:   s -= 3
        elif vs200 < -0.08:   s -= 2
        elif vs200 < -0.03:   s -= 1

        if   mom3m >  0.08:   s += 2
        elif mom3m >  0.02:   s += 1
        elif mom3m < -0.15:   s -= 3
        elif mom3m < -0.05:   s -= 1

        if   vol20 < 0.12:    s += 1
        elif vol20 > 0.35:    s -= 2
        elif vol20 > 0.25:    s -= 1

        s += 1 if golden else -1

        if vix_curr > 40 or s <= -6:
            return dict(regime="CRISIS",   color="#ff1744", bg="#1a0000", emoji="🔴",
                        desc="Market in CRISIS — extreme fear, capital preservation first",
                        action="Raise cash. Hold only anchors. No new positions.",
                        kelly_mult=0.20, mc_adj=-0.0004,
                        vix=round(vix_curr,1), mom_3m=round(mom3m*100,1),
                        vs_200ma=round(vs200*100,1), vol_20d=round(vol20*100,1),
                        score=s, golden=golden)
        elif s <= -3 or (vix_curr > 28 and mom3m < -0.05):
            return dict(regime="BEAR",     color="#ff5252", bg="#1a0000", emoji="🐻",
                        desc="BEAR market — trend down, reduce exposure",
                        action="Cut sizes 40-60%. Tighten stops. Favour quality.",
                        kelly_mult=0.40, mc_adj=-0.0002,
                        vix=round(vix_curr,1), mom_3m=round(mom3m*100,1),
                        vs_200ma=round(vs200*100,1), vol_20d=round(vol20*100,1),
                        score=s, golden=golden)
        elif s <= 0 or vol20 > 0.22:
            return dict(regime="HIGH_VOL", color="#ffcc00", bg="#1a1000", emoji="⚡",
                        desc="HIGH VOLATILITY — choppy, size down, wait for clarity",
                        action="Half-size positions. Wait for VIX < 20 for full sizing.",
                        kelly_mult=0.65, mc_adj=-0.0001,
                        vix=round(vix_curr,1), mom_3m=round(mom3m*100,1),
                        vs_200ma=round(vs200*100,1), vol_20d=round(vol20*100,1),
                        score=s, golden=golden)
        else:
            return dict(regime="BULL",     color="#00c853", bg="#001a00", emoji="🐂",
                        desc="BULL market — trend up, full sizing appropriate",
                        action="Full Kelly sizing. Ride momentum. Trailing stops.",
                        kelly_mult=1.0,  mc_adj=0.0001,
                        vix=round(vix_curr,1), mom_3m=round(mom3m*100,1),
                        vs_200ma=round(vs200*100,1), vol_20d=round(vol20*100,1),
                        score=s, golden=golden)
    except Exception:
        return dict(regime="UNKNOWN", color="#888", bg="#111", emoji="❓",
                    desc="Regime unknown — check connectivity",
                    action="Use conservative sizing until regime is clear.",
                    kelly_mult=0.75, mc_adj=0.0,
                    vix=20.0, mom_3m=0.0, vs_200ma=0.0, vol_20d=20.0,
                    score=0, golden=True)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2 — Dynamic Universe Builder
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def build_universe_india():
    """
    Dynamically fetch Nifty 500 + Smallcap 250 + Midcap 150
    constituents from NSE. Returns list of .NS ticker symbols.
    Falls back to a minimal seed list if fetch fails.
    """
    all_symbols = set()
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            lines = resp.text.strip().split("\n")
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 2:
                    sym = parts[1].strip().strip('"')
                    if sym and len(sym) > 1:
                        all_symbols.add(sym + ".NS")
        except Exception:
            continue

    if len(all_symbols) < 50:
        fallback = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS",
            "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "SUNPHARMA.NS",
            "TATAMOTORS.NS", "ADANIENT.NS", "ONGC.NS", "NTPC.NS",
            "ZOMATO.NS", "NYKAA.NS", "DIXON.NS", "APOLLOHOSP.NS",
            "TATAPOWER.NS", "JSWSTEEL.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
        ]
        all_symbols.update(fallback)

    return list(all_symbols)


@st.cache_data(ttl=86400, show_spinner=False)
@st.cache_data(ttl=86400, show_spinner=False)
def build_universe_us_smallcap():
    """
    Fetch S&P 600 (small-cap index) from Wikipedia.
    Falls back to a curated list of ~150 actively-traded US small caps.
    Returns list of ticker symbols.
    """
    all_symbols = set()

    # S&P 600 Small Cap from Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
        for t in tables:
            for col in ("Ticker symbol", "Symbol", "Ticker"):
                if col in t.columns:
                    all_symbols.update(t[col].dropna().astype(str).tolist())
                    break
    except Exception:
        pass

    # Curated small-cap seeds (always included — well-known actively-traded names)
    smallcap_seeds = [
        # Tech / Software small-caps
        "CLFD", "JAMF", "NTNX", "PEGA", "VRNS", "QLYS", "EVBG",
        "ALRM", "ENFN", "PRGS", "CSGS", "KFRC", "CWAN",
        # Biotech / Healthcare
        "NVCR", "ACAD", "SEER", "RXRX", "BMRN", "FOLD",
        "SNDX", "ACLS", "INVA", "CNMD", "MMSI", "MLAB",
        # Fintech / Financial
        "CURO", "ENVA", "WRLD", "OPRX", "KXIN", "NRDS",
        "TASK", "PAID", "FLYW",
        # Industrial / Defence
        "KTOS", "AVAV", "RCAT", "CACI", "DRS", "PLBY",
        "HURN", "HAYN", "MTRX", "GHM",
        # Consumer / Retail
        "XPOF", "GRIL", "PRTH", "RICK", "VTOL", "TPVG",
        "TPIC", "IRBT", "SMTK",
        # Energy / Materials
        "CUEN", "BATL", "REX", "USPH", "GATO", "HUSA",
        "MARPS", "AMPY", "MNTK",
        # Real estate / REIT small-caps
        "PINE", "GOOD", "LAND", "GIPR", "PLYM", "NXRT",
        # India-related US small-caps
        "SIFY", "INTT", "AMRN",
        # EV / Clean energy
        "NKLA", "SOLO", "WKHS", "AYRO", "BLNK", "CHPT",
        # AI / Robotics emerging
        "AITX", "BBAI", "SOUN", "PRCT", "CEVA", "IPIX",
    ]
    all_symbols.update(smallcap_seeds)

    return [s for s in all_symbols if s and len(s) <= 6 and s.isalnum()]


def build_universe_us():
    """
    Dynamically fetch S&P 500 + Nasdaq 100 (large/mid cap).
    Returns list of ticker symbols.
    """
    all_symbols = set()

    # S&P 500 from Wikipedia
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        sp500 = tables[0]["Symbol"].tolist()
        all_symbols.update([s.replace(".", "-") for s in sp500])
    except Exception:
        pass

    # Nasdaq 100 from Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            if "Ticker" in t.columns:
                all_symbols.update(t["Ticker"].dropna().tolist())
                break
            if "Symbol" in t.columns:
                all_symbols.update(t["Symbol"].dropna().tolist())
                break
    except Exception:
        pass

    if len(all_symbols) < 50:
        fallback = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
            "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "MA", "PG", "AVGO",
            "HD", "CVX", "ABBV", "COST", "BAC", "TMO", "CSCO", "ACN",
            "PANW", "CRWD", "NET", "ZS", "DDOG", "SNOW", "PLTR",
        ]
        all_symbols.update(fallback)

    # Extended mid-cap / high-growth universe always added regardless of S&P scrape
    extended = [
        # Cloud & SaaS mid-caps often missed by S&P/Nasdaq-100
        "FRSH", "BILL", "ZI", "GTLB", "MNDY", "HUBS", "DOCN", "CFLT",
        "ESTC", "ELASTIC", "APPN", "PCTY", "PAYC", "COUP",
        # Tech hardware / storage
        "DELL", "HPE", "WDC", "STX", "NTAP", "PSTG",
        # Semiconductor mid-caps
        "WOLF", "SWKS", "QRVO", "MPWR", "CRUS", "ACLS", "ONTO",
        # Fintech
        "SQ", "AFRM", "UPST", "SOFI", "HOOD", "NRDS",
        # Healthcare / biotech
        "CELH", "HIMS", "NVAX", "MRNA", "BNTX", "REGN", "VRTX",
        # Retail / consumer growth
        "DUOL", "RBLX", "U", "UNITY", "TTWO", "EA", "ZNGA",
        # Industrial / defence small-cap
        "KTOS", "RCAT", "JOBY", "ACHR", "LILM", "EVTL",
        # India ADRs / US-listed India exposure
        "INFY", "WIT", "HDB", "IBN", "SIFY", "VNET",
        # Quantum / AI pure-plays
        "IONQ", "RGTI", "QUBT", "ARQQ", "QMCO", "IQM",
        "BBAI", "SOUN", "AITX", "PRCT",
        # Emerging US growth
        "NOW", "TTD", "DKNG", "PENN", "WYNN", "MGM",
        "RIVN", "LCID", "FSR", "NKLA",
    ]
    all_symbols.update(extended)

    return list(all_symbols)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3 — Theme Discovery via Gemini
# ══════════════════════════════════════════════════════════════════════════════

def discover_theme_stocks(theme_query, gemini_api_key, market="both"):
    """
    Use Gemini to find company names for a theme,
    then resolve to valid yfinance tickers.
    Returns list of dicts: {symbol, name, theme}
    """
    if not gemini_api_key:
        return []
    try:
        from google import genai as genai_sdk
        client = genai_sdk.Client(api_key=gemini_api_key)

        if market == "india":
            market_context = "listed on NSE or BSE India"
        elif market == "us":
            market_context = "listed on NYSE or NASDAQ"
        else:
            market_context = "listed on NSE, BSE, NYSE, or NASDAQ"

        prompt = f"""Find 12 publicly listed companies {market_context}
that are directly involved in: {theme_query}

Focus on SMALL and MID cap companies that most retail investors
don't know about. Avoid obvious large caps unless they are pure-play.

Return ONLY a valid JSON array like this, no other text:
[
  {{"name": "Company Name", "ticker_hint": "SYMBOL", "exchange": "NSE/NYSE/NASDAQ/BSE"}},
  ...
]"""

        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        companies = json.loads(text)

        discovered = []
        for c in companies[:12]:
            hint     = c.get("ticker_hint", "")
            exchange = c.get("exchange", "")
            name     = c.get("name", "")

            if hint:
                sym = hint.strip()
                if exchange in ("NSE", "BSE") and not sym.endswith(".NS"):
                    sym = sym + ".NS"
                try:
                    test  = yf.Ticker(sym).fast_info
                    price = getattr(test, "last_price", None)
                    if price and price > 0:
                        discovered.append({"symbol": sym, "name": name, "theme": theme_query})
                        continue
                except Exception:
                    pass

            try:
                resp = requests.get(
                    "https://query2.finance.yahoo.com/v1/finance/search",
                    params={"q": name, "quotesCount": 3, "newsCount": 0},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=5,
                )
                for q in resp.json().get("quotes", []):
                    if q.get("quoteType") == "EQUITY":
                        sym = q.get("symbol", "")
                        if sym:
                            discovered.append({"symbol": sym, "name": name, "theme": theme_query})
                            break
            except Exception:
                pass

        return discovered
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 4 — Feature Computation
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(v, default=None):
    """Return float(v) or default if v is None/NaN/invalid."""
    try:
        f = float(v)
        return default if (f != f) else f   # f != f is True only for NaN
    except Exception:
        return default


def fetch_stock_features(sym):
    """
    Compute all 4-layer features for a single stock.
    FIX 1: individual try/except per indicator block — one failure
            never drops the whole stock; returns partial dict with None.
    Returns partial-or-full feature dict, or None only if price data missing.
    """
    # ── Price data — mandatory; bail if unavailable ───────────────────────────
    try:
        t  = yf.Ticker(sym)
        df = t.history(period="1y")   # 1y is enough for all indicators; 2x faster
        if df is None or len(df) < 60:
            return None
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()
        price = float(close.iloc[-1])
        if price <= 0:
            return None
    except Exception:
        return None

    # ── Company info (optional) ───────────────────────────────────────────────
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    def pct(days):
        try:
            if len(close) <= days:
                return None
            return _safe_float((close.iloc[-1] / close.iloc[-days - 1]) - 1)
        except Exception:
            return None

    def sf(key):
        return _safe_float(info.get(key))

    # ── Technical block (FIX 1: each indicator has its own try/except) ────────
    mom_1w = pct(5);   mom_1m = pct(21)
    mom_3m = pct(63);  mom_6m = pct(126); mom_1y = pct(252)

    atr = None; atr_pct = None
    try:
        h_l  = high - low
        h_pc = (high - close.shift()).abs()
        l_pc = (low  - close.shift()).abs()
        tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        atr  = _safe_float(tr.rolling(14).mean().iloc[-1], 1.0)
        atr_pct = (atr / price * 100) if price > 0 else 2.0
    except Exception:
        atr = 1.0; atr_pct = 2.0

    vol_surge = 1.0
    try:
        v20 = _safe_float(vol.rolling(20).mean().iloc[-1]) or 1.0
        vol_surge = _safe_float(float(vol.iloc[-5:].mean()) / v20, 1.0)
    except Exception:
        pass

    high_52w = price; low_52w = price
    try:
        high_52w = _safe_float(close.rolling(min(252, len(close))).max().iloc[-1], price)
        low_52w  = _safe_float(close.rolling(min(252, len(close))).min().iloc[-1], price)
    except Exception:
        pass

    rsi = 50.0
    try:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = _safe_float((100 - (100 / (1 + rs))).iloc[-1], 50.0)
    except Exception:
        pass

    macd_hist = 0.0
    try:
        ema12     = close.ewm(span=12).mean()
        ema26     = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9).mean()
        macd_hist = _safe_float((macd_line - macd_sig).iloc[-1], 0.0)
    except Exception:
        pass

    bb_pct = 0.5
    try:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_h   = bb_mid + 2 * bb_std
        bb_l   = bb_mid - 2 * bb_std
        bb_rng = _safe_float((bb_h - bb_l).iloc[-1])
        if bb_rng and bb_rng > 0:
            bb_pct = _safe_float((price - bb_l.iloc[-1]) / bb_rng, 0.5)
    except Exception:
        pass

    obv_slope = 0.0
    try:
        obv      = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        obv_ref  = _safe_float(obv.iloc[-21]) or 1.0
        obv_slope= _safe_float((obv.iloc[-1] - obv.iloc[-21]) / abs(obv_ref), 0.0)
    except Exception:
        pass

    ma50 = price; ma200 = price
    try:
        ma50  = _safe_float(close.rolling(50).mean().iloc[-1],  price)
        ma200 = _safe_float(
            close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50,
            ma50,
        )
    except Exception:
        pass

    hist_mean = 0.0; hist_std = 0.01; hist_skew = 0.0; hist_kurt = 0.0
    max_dd = -0.1; sharpe = 0.0
    try:
        daily_ret = close.pct_change().dropna()
        hist_mean = _safe_float(daily_ret.mean(), 0.0)
        hist_std  = _safe_float(daily_ret.std(),  0.01) or 0.01
        hist_skew = _safe_float(stats.skew(daily_ret), 0.0)
        hist_kurt = _safe_float(stats.kurtosis(daily_ret), 0.0)
        roll_max  = close.cummax()
        max_dd    = _safe_float(((close - roll_max) / roll_max).min(), -0.1)
        rf_daily  = 0.06 / 252 if sym.endswith(".NS") else 0.05 / 252
        excess    = daily_ret - rf_daily
        sharpe    = _safe_float(
            excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0,
            0.0,
        )
    except Exception:
        pass

    # ── Technical-only return (fundamentals fetched separately for final picks)
    # Fundamentals are left as None here — they are filled by
    # enrich_with_fundamentals() which runs only on the small final pick list.
    currency = "₹" if sym.endswith(".NS") else "$"
    exchange = "NSE" if sym.endswith(".NS") else "US"

    # Discovery label from technicals only (refined after fundamental enrich)
    if (mom_3m or 0) > 0.15 and rsi < 65:  disc_label = "momentum"
    elif rsi < 35:                           disc_label = "hidden_gem"
    else:                                    disc_label = "theme_play"

    return {
        "symbol": sym,        "name": sym,          "sector": "Unknown",
        "price": price,       "currency": currency,  "exchange": exchange,
        "mktcap": None,       "n_analysts": None,
        "mom_1w": mom_1w,     "mom_1m": mom_1m,     "mom_3m": mom_3m,
        "mom_6m": mom_6m,     "mom_1y": mom_1y,
        "rsi": rsi,           "macd_hist": macd_hist,
        "bb_pct": bb_pct,     "atr_pct": atr_pct,
        "vol_surge": vol_surge, "obv_slope": obv_slope,
        "above_ma50":   1 if price > ma50  else 0,
        "above_ma200":  1 if price > ma200 else 0,
        "golden_cross": 1 if ma50  > ma200 else 0,
        "pct_from_52w_high": _safe_float(price / high_52w - 1, 0.0),
        "pct_from_52w_low":  _safe_float(price / low_52w  - 1, 0.0),
        "high_52w": high_52w, "low_52w": low_52w,
        "hist_mean": hist_mean, "hist_std": hist_std,
        "hist_skew": hist_skew, "hist_kurt": hist_kurt,
        "max_dd": max_dd,     "sharpe": sharpe,
        # Fundamentals — populated by enrich_with_fundamentals()
        "pe": None,  "fwd_pe": None, "peg": None,   "pb": None,
        "ev_ebitda": None,  "roe": None,
        "net_margin": None, "op_margin": None,
        "rev_growth": None, "earn_growth": None,
        "de_ratio": None,   "curr_ratio": None,
        "fcf_yield": None,  "beta": None,
        "insider": None,    "inst_holding": None,
        "short_pct": None,  "analyst_rec": None,
        "analyst_upside": None,
        "div_yield": None,  "w52_change": None,
        "discovery_bonus": 5,
        "discovery_label": disc_label,
        "theme": None,
    }


def enrich_with_fundamentals(f):
    """
    Fetch yfinance .info for a single stock and merge into feature dict.
    Called ONLY for final picks (~15 stocks) — not during the mass scan.
    Returns updated f dict in place.
    """
    sym = f.get("symbol", "")
    try:
        info = yf.Ticker(sym).info or {}

        def sf(key):
            return _safe_float(info.get(key))

        mktcap      = sf("marketCap")
        n_analysts  = sf("numberOfAnalystOpinions")
        target_mean = sf("targetMeanPrice")
        price       = f["price"]

        f.update({
            "name":          info.get("shortName") or info.get("longName") or sym,
            "sector":        info.get("sector") or "Unknown",
            "mktcap":        mktcap,
            "n_analysts":    n_analysts,
            "pe":            sf("trailingPE"),
            "fwd_pe":        sf("forwardPE"),
            "peg":           sf("pegRatio"),
            "pb":            sf("priceToBook"),
            "ev_ebitda":     sf("enterpriseToEbitda"),
            "roe":           sf("returnOnEquity"),
            "net_margin":    sf("profitMargins"),
            "op_margin":     sf("operatingMargins"),
            "rev_growth":    sf("revenueGrowth"),
            "earn_growth":   sf("earningsGrowth"),
            "de_ratio":      sf("debtToEquity"),
            "curr_ratio":    sf("currentRatio"),
            "beta":          sf("beta"),
            "insider":       sf("heldPercentInsiders"),
            "inst_holding":  sf("heldPercentInstitutions"),
            "short_pct":     sf("sharesPercentSharesOut"),
            "analyst_rec":   sf("recommendationMean"),
            "analyst_upside":((target_mean / price) - 1) if (target_mean and price > 0) else None,
            "div_yield":     sf("dividendYield"),
            "w52_change":    sf("52WeekChange"),
            "short_ratio":   sf("shortRatio"),           # days to cover
            "short_float":   sf("shortPercentOfFloat"),  # % of float shorted
            "exchange":      info.get("exchange", ""),
        })

        # Derived
        fcf   = sf("freeCashflow")
        f["fcf_yield"] = (fcf / mktcap) if (fcf and mktcap and mktcap > 0) else None

        # Refine discovery label with real data
        n_ana_int       = int(n_analysts) if n_analysts else 0
        discovery_bonus = 0
        if   n_ana_int == 0: discovery_bonus += 15
        elif n_ana_int <= 3: discovery_bonus += 10
        elif n_ana_int <= 8: discovery_bonus += 5
        if mktcap:
            thr = 50e9 if sym.endswith(".NS") else 2e9
            if mktcap < thr:
                discovery_bonus += 5
        f["discovery_bonus"] = min(20, discovery_bonus)

        roe = f.get("roe")
        mom_3m = f.get("mom_3m")
        if   n_ana_int <= 3:                               f["discovery_label"] = "undiscovered"
        elif n_ana_int <= 8 and mktcap and mktcap < 50e9: f["discovery_label"] = "hidden_gem"
        elif mom_3m and mom_3m > 0.15:                    f["discovery_label"] = "momentum"
        elif roe and roe > 0.20:                           f["discovery_label"] = "quality"
        else:                                              f["discovery_label"] = "theme_play"

    except Exception:
        pass   # keep whatever partial data exists
    return f


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 5 — Parallel Universe Scanner
# ══════════════════════════════════════════════════════════════════════════════

def scan_universe_parallel(symbols, progress_bar=None, status_text=None,
                            max_workers=20):
    """
    Fetch features for all symbols in parallel.
    Returns dict {symbol: feature_dict}
    """
    results = {}
    total   = len(symbols)
    done    = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(fetch_stock_features, sym): sym
            for sym in symbols
        }
        for future in as_completed(future_to_sym):
            sym   = future_to_sym[future]
            done += 1
            try:
                data = future.result(timeout=8)   # 8 s hard limit per stock
                if data is not None:
                    results[sym] = data
            except Exception:
                pass
            if progress_bar:
                progress_bar.progress(min(1.0, done / total))
            if status_text:
                status_text.text(
                    f"Scanning {done}/{total} stocks... "
                    f"({len(results)} qualified so far)"
                )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — constants + helpers
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "mom_1w", "mom_1m", "mom_3m", "mom_6m", "mom_1y",
    "rsi", "macd_hist", "bb_pct", "atr_pct", "vol_surge",
    "obv_slope", "above_ma50", "above_ma200", "golden_cross",
    "pct_from_52w_high", "pct_from_52w_low",
    "hist_mean", "hist_std", "hist_skew", "hist_kurt",
    "max_dd", "sharpe",
    "pe", "fwd_pe", "peg", "pb", "ev_ebitda", "roe",
    "net_margin", "op_margin", "rev_growth", "earn_growth",
    "de_ratio", "curr_ratio", "fcf_yield", "beta",
    "insider", "inst_holding", "short_pct",
    "analyst_rec", "analyst_upside", "div_yield", "w52_change",
]


def normalise_universe(features_dict):
    """
    Compute per-column percentile ranks across the full universe.
    Returns dict {symbol: {col: percentile_0_to_1}}
    """
    rows = []
    syms = list(features_dict.keys())
    for sym in syms:
        f   = features_dict[sym]
        row = [f.get(col) for col in FEATURE_COLS]
        rows.append(row)

    df_u = pd.DataFrame(rows, columns=FEATURE_COLS, index=syms)
    df_u = df_u.apply(pd.to_numeric, errors="coerce")
    df_u = df_u.fillna(df_u.median())
    ranked = df_u.rank(pct=True)
    return ranked.to_dict(orient="index")


def score_stock(f, ranked, bucket_key, risk_pref="Moderate", inv_horizon="1 Year"):
    """
    Score a stock 0-100 for a specific bucket.
    risk_pref:   "Conservative" | "Moderate" | "Aggressive"
    inv_horizon: "3 Months" | "6 Months" | "1 Year" | "3 Years"
    Returns (total_score, breakdown_dict)
    """
    if f is None or ranked is None:
        return 0, {}

    sym = f.get("symbol", "")
    r   = ranked.get(sym, {})

    # Base weights per bucket
    BASE_WEIGHTS = {
        "anchor":     {"tech": 0.20, "fund": 0.50, "risk": 0.20, "sent": 0.10},
        "growth":     {"tech": 0.35, "fund": 0.40, "risk": 0.15, "sent": 0.10},
        "rotational": {"tech": 0.50, "fund": 0.20, "risk": 0.20, "sent": 0.10},
    }
    w = dict(BASE_WEIGHTS.get(bucket_key, BASE_WEIGHTS["growth"]))

    # ── Risk preference adjustments ───────────────────────────────────────────
    # Conservative: more fundamental + risk, less technical momentum
    # Aggressive: more technical + sentiment, less risk emphasis
    if risk_pref == "Conservative":
        w["tech"] = max(0.10, w["tech"] - 0.12)
        w["fund"] = min(0.65, w["fund"] + 0.08)
        w["risk"] = min(0.35, w["risk"] + 0.08)
        w["sent"] = max(0.05, w["sent"] - 0.04)
    elif risk_pref == "Aggressive":
        w["tech"] = min(0.65, w["tech"] + 0.12)
        w["fund"] = max(0.15, w["fund"] - 0.08)
        w["risk"] = max(0.08, w["risk"] - 0.08)
        w["sent"] = min(0.20, w["sent"] + 0.04)

    # ── Investment horizon adjustments ────────────────────────────────────────
    # Short horizon → momentum matters more; long horizon → fundamentals matter more
    if inv_horizon == "3 Months":
        w["tech"] = min(0.70, w["tech"] + 0.10)
        w["fund"] = max(0.10, w["fund"] - 0.10)
    elif inv_horizon in ("1 Year", "3 Years"):
        w["tech"] = max(0.10, w["tech"] - 0.08)
        w["fund"] = min(0.70, w["fund"] + 0.08)

    # Normalise so weights sum to 1
    wsum = sum(w.values())
    w = {k: v / wsum for k, v in w.items()}

    def rv(col, default=0.5):
        """Return percentile rank for col; substitute default if NaN/missing."""
        v = r.get(col, default)
        try:
            f_v = float(v)
            return default if (f_v != f_v) else f_v   # NaN check
        except Exception:
            return default

    def safe_mean(lst):
        """np.nanmean with a 0.5 fallback if all values are NaN."""
        arr = np.array([float(x) for x in lst], dtype=float)
        return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.5

    # ── Technical ────────────────────────────────────────────────────────────
    tech = []
    if bucket_key == "rotational":
        tech += [rv("mom_1m") * 1.5, rv("mom_3m") * 2.0, rv("mom_1w") * 1.0]
    elif bucket_key == "growth":
        tech += [rv("mom_1m") * 1.2, rv("mom_3m") * 1.5, rv("mom_6m") * 1.0]
    else:
        tech += [rv("mom_3m") * 0.8, rv("mom_6m") * 1.2, rv("mom_1y") * 1.5]

    tech += [rv("above_ma50"), rv("above_ma200"), rv("golden_cross"),
             rv("vol_surge"), rv("obv_slope"), rv("sharpe") * 1.5]

    rsi_val = float(f.get("rsi") or 50)
    if   rsi_val < 30: rsi_score = 0.85
    elif rsi_val < 45: rsi_score = 0.75
    elif rsi_val < 60: rsi_score = 0.65
    elif rsi_val < 70: rsi_score = 0.45
    else:              rsi_score = 0.20
    tech.append(rsi_score)
    tech_score = float(np.clip(safe_mean(tech) * 100, 0, 100))

    # ── Fundamental ───────────────────────────────────────────────────────────
    fund = [
        1 - rv("pe"), 1 - rv("fwd_pe"), 1 - rv("peg"),
        rv("roe"), rv("net_margin"), rv("op_margin"),
        rv("rev_growth"), rv("earn_growth"), rv("fcf_yield"),
        1 - rv("de_ratio"), rv("curr_ratio"),
    ]
    fund_score = float(np.clip(safe_mean(fund) * 100, 0, 100))

    # ── Risk ──────────────────────────────────────────────────────────────────
    beta_val = float(f.get("beta") or 1.0)
    if   bucket_key == "anchor":  beta_s = max(0.0, 1 - (beta_val - 0.5) / 1.5)
    elif bucket_key == "growth":  beta_s = 1 - abs(beta_val - 1.1) / 1.5
    else:                         beta_s = min(1.0, beta_val / 2.0)

    risk = [float(np.clip(beta_s, 0, 1)),
            1 - rv("max_dd"), rv("analyst_upside"),
            rv("w52_change"), rv("curr_ratio")]
    risk_score = float(np.clip(safe_mean(risk) * 100, 0, 100))

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sent = [1 - rv("analyst_rec"), rv("insider"),
            rv("inst_holding"), 1 - rv("short_pct")]
    sent_score = float(np.clip(safe_mean(sent) * 100, 0, 100))

    # ── Hard disqualifiers ────────────────────────────────────────────────────
    penalty = 0
    if (f.get("net_margin") or 0) < -0.10:                          penalty += 25
    if (f.get("de_ratio")   or 0) > 300 and bucket_key=="anchor":   penalty += 25
    if (f.get("beta")       or 0) > 3.0 and bucket_key=="anchor":   penalty += 20
    if (f.get("rev_growth") or 0) < -0.25:                          penalty += 20
    if (f.get("short_pct")  or 0) > 0.25:                           penalty += 15
    if (f.get("mktcap")     or 1e12) < 5e8 and bucket_key=="anchor":penalty += 15
    # Risk-preference extra penalties
    beta_val_p = float(f.get("beta") or 1.0)
    if risk_pref == "Conservative":
        if beta_val_p > 1.5:  penalty += 20   # conservative: penalise high-beta heavily
        if beta_val_p > 2.0:  penalty += 20
        if (f.get("max_dd") or 0) < -0.40: penalty += 15   # deep drawdown = bad
    elif risk_pref == "Aggressive":
        if beta_val_p < 0.5:  penalty += 10   # aggressive: penalise very low-beta

    total = (
        tech_score  * w["tech"] +
        fund_score  * w["fund"] +
        risk_score  * w["risk"] +
        sent_score  * w["sent"]
    ) - penalty + float(f.get("discovery_bonus") or 0)

    total = max(0.0, min(100.0, float(total) if total == total else 50.0))

    def _r(v):  # round, NaN-safe
        try:
            fv = float(v)
            return round(fv if fv == fv else 50.0, 1)
        except Exception:
            return 50.0

    return total, {
        "technical":   _r(tech_score),
        "fundamental": _r(fund_score),
        "risk":        _r(risk_score),
        "sentiment":   _r(sent_score),
        "penalty":     penalty,
        "discovery":   float(f.get("discovery_bonus") or 0),
        "total":       _r(total),
    }


def classify_bucket(f, ranked):
    """
    Return which bucket (anchor/growth/rotational) best fits the stock.
    Thresholds deliberately inclusive so all three buckets are well populated.
    """
    beta   = float(f.get("beta",      1.0) or 1.0)
    mom_3m = float(f.get("mom_3m",    0.0) or 0.0)
    mom_1m = float(f.get("mom_1m",    0.0) or 0.0)
    sharpe = float(f.get("sharpe",    0.0) or 0.0)
    max_dd = float(f.get("max_dd",   -0.3) or -0.3)
    vol    = float(f.get("vol_surge", 1.0) or 1.0)
    rsi    = float(f.get("rsi",        50) or 50)
    above_ma200 = int(f.get("above_ma200", 0) or 0)

    # Anchor: low-beta, low drawdown, consistent trend, decent Sharpe
    if beta < 1.1 and abs(max_dd) < 0.30 and sharpe > 0.3 and above_ma200:
        return "anchor"

    # Rotational: any meaningful positive momentum + elevated beta or volume
    # Lowered thresholds so 7-8 stocks can qualify
    if (mom_3m > 0.05 or mom_1m > 0.08) and (vol > 1.2 or beta > 1.1) and rsi > 45:
        return "rotational"

    return "growth"


def assign_percentile(score, all_scores):
    """Return P99/P95/P90/P80/WATCH based on score distribution."""
    if not all_scores or len(all_scores) < 5:
        return "P80"
    pct = stats.percentileofscore(all_scores, score)
    if pct >= 99: return "P99"
    if pct >= 95: return "P95"
    if pct >= 90: return "P90"
    if pct >= 80: return "P80"
    return "WATCH"


# ══════════════════════════════════════════════════════════════════════════════
# RETURN OBJECTIVE ENGINE  — score stocks for a specific return target + duration
# ══════════════════════════════════════════════════════════════════════════════

# Map horizon label → MC key and trading days
OBJECTIVE_HORIZON_MAP = {
    "1M":  ("1M",  21),
    "3M":  ("3M",  63),
    "6M":  ("1Y",  126),   # MC doesn't have 6M; use 1Y and scale
    "1Y":  ("1Y",  252),
}

# MC probability field nearest to each target pct
_PROB_FIELDS = {
    0:   "prob_positive",
    5:   "prob_5pct",
    10:  "prob_10pct",
    15:  "prob_15pct",
    20:  "prob_20pct",
    25:  "prob_25pct",
    30:  "prob_30pct",
    40:  "prob_40pct",
    50:  "prob_50pct",
    75:  "prob_75pct",
    100: "prob_100pct",
}

def get_prob_at_target(mc_horizon_dict, target_pct):
    """
    Return probability (0-100) of achieving >= target_pct return at a given horizon.
    Interpolates between the nearest stored thresholds.
    """
    if not mc_horizon_dict:
        return 0.0
    thresholds = sorted(_PROB_FIELDS.keys())
    # Find surrounding thresholds
    lo = max((t for t in thresholds if t <= target_pct), default=0)
    hi = min((t for t in thresholds if t >= target_pct), default=100)
    if lo == hi:
        return float(mc_horizon_dict.get(_PROB_FIELDS[lo], 0) or 0)
    p_lo = float(mc_horizon_dict.get(_PROB_FIELDS[lo], 0) or 0)
    p_hi = float(mc_horizon_dict.get(_PROB_FIELDS[hi], 0) or 0)
    frac = (target_pct - lo) / (hi - lo) if hi > lo else 0
    return round(p_lo + (p_hi - p_lo) * frac, 1)


def score_for_objective(f, mc_result, target_pct, horizon_label):
    """
    Score a stock specifically for the objective: achieve target_pct% in horizon_label.

    This replaces the standard 4-layer quality score when the user sets a return objective.
    Returns (score 0-100, breakdown_dict, prob_of_target).

    The objective mode rewards:
      1. Probability of hitting the target (from Monte Carlo) — 40%
      2. Momentum strength and direction                      — 25%
      3. Volatility adequacy (stock must be volatile enough)  — 20%
      4. Catalyst potential (squeeze, breakout, recovery)     — 15%
    """
    if not mc_result:
        return 0, {}, 0

    mc_key, _ = OBJECTIVE_HORIZON_MAP.get(horizon_label, ("3M", 63))
    mc_h = mc_result.get(mc_key, {})

    # 1. Probability of hitting the target
    prob_target = get_prob_at_target(mc_h, target_pct)
    # Normalise to 0-100 score (cap at 40% probability = full score for this component)
    prob_score = min(100, prob_target / max(target_pct * 0.05, 1) * 100)

    # 2. Momentum score
    mom3  = float(f.get("mom_3m") or 0)
    mom1  = float(f.get("mom_1m") or 0)
    mom6  = float(f.get("mom_6m") or 0)
    if   mom3 > 0.20: mom_score = 100
    elif mom3 > 0.10: mom_score = 80
    elif mom3 > 0.05: mom_score = 65
    elif mom3 > 0:    mom_score = 45
    elif mom3 > -0.10: mom_score = 25  # slight dip — contrarian
    else:              mom_score = max(0, 10 + mom3 * 100)

    # 3. Volatility adequacy — stock MUST be volatile enough to hit target in time
    sigma = float(f.get("hist_std") or 0.015) or 0.015
    _, td = OBJECTIVE_HORIZON_MAP.get(horizon_label, ("3M", 63))
    # 1-sigma range at horizon ≈ sigma * sqrt(td) * 100%
    one_sigma_range = sigma * np.sqrt(td) * 100
    # Need sigma range to be at least 50% of target to have meaningful probability
    vol_score = min(100, one_sigma_range / max(target_pct * 0.5, 1) * 100)

    # 4. Catalyst / squeeze potential
    catalyst = 0
    rsi   = float(f.get("rsi") or 50)
    short = float(f.get("short_pct") or 0)
    vsurg = float(f.get("vol_surge") or 1)
    beta  = float(f.get("beta") or 1)

    if rsi < 30:            catalyst += 25   # deeply oversold = bounce fuel
    elif rsi < 45:          catalyst += 12   # mildly oversold
    if short > 0.15:        catalyst += 25   # high short interest = squeeze potential
    elif short > 0.08:      catalyst += 12
    if vsurg > 2.5:         catalyst += 25   # unusual volume = something happening
    elif vsurg > 1.5:       catalyst += 12
    if beta > 2.0:          catalyst += 15   # amplified mover
    elif beta > 1.5:        catalyst += 8
    if not f.get("above_ma200") and f.get("above_ma50"):
        catalyst += 15   # recovering from downtrend
    catalyst = min(100, catalyst)

    total = (
        prob_score  * 0.40 +
        mom_score   * 0.25 +
        vol_score   * 0.20 +
        catalyst    * 0.15
    )

    # Penalise structurally incapable stocks (too stable to ever hit target)
    if one_sigma_range < target_pct * 0.30:
        total *= 0.4   # stock can't mathematically hit target in this timeframe

    total = round(min(100, max(0, total)), 1)

    return total, {
        "prob_target": round(prob_target, 1),
        "momentum":    round(mom_score, 0),
        "volatility":  round(vol_score, 0),
        "catalyst":    round(catalyst, 0),
    }, prob_target


def get_objective_scoring_weights(target_pct, horizon_label):
    """
    Return score_stock weights tuned for the return objective.
    Higher targets → more weight on technical/momentum, less on fundamentals.
    """
    if target_pct >= 40 or horizon_label == "1M":
        return {"tech": 0.65, "fund": 0.10, "risk": 0.10, "sent": 0.15}
    elif target_pct >= 25 or horizon_label == "3M":
        return {"tech": 0.50, "fund": 0.20, "risk": 0.15, "sent": 0.15}
    elif target_pct >= 15:
        return {"tech": 0.40, "fund": 0.30, "risk": 0.18, "sent": 0.12}
    else:
        return {"tech": 0.35, "fund": 0.35, "risk": 0.20, "sent": 0.10}


# ══════════════════════════════════════════════════════════════════════════════
# SHORT CANDIDATE SCORER  — inverted quality model for identifying shorts
# ══════════════════════════════════════════════════════════════════════════════

def score_short_candidate(f):
    """
    Score a stock as a SHORT candidate (0–100, higher = better short).
    Looks for: overbought technicals + weak fundamentals + stretched valuation + downtrend.
    """
    score = 0.0
    bkdn  = {}

    # 1. Technical signals (max 40 pts) — overbought & in downtrend
    tech = 0
    rsi  = float(f.get("rsi") or 50)
    if   rsi > 75: tech += 15
    elif rsi > 65: tech += 8
    elif rsi < 35: tech -= 10   # oversold = dangerous short

    if not f.get("above_ma200", False): tech += 12  # below 200MA = downtrend
    if not f.get("above_ma50",  False): tech += 5

    mom3 = float(f.get("mom_3m") or 0)
    mom1 = float(f.get("mom_1m") or 0)
    if   mom3 < -0.12: tech += 15
    elif mom3 < -0.05: tech += 8
    elif mom3 >  0.12: tech -= 10  # strong uptrend = risky to short

    tech = max(0, min(40, tech))
    bkdn["technical"] = tech
    score += tech

    # 2. Fundamental weakness (max 40 pts)
    fund = 0
    roe  = _safe_float(f.get("roe"))
    if roe is not None:
        if   roe < 0:    fund += 12
        elif roe < 0.05: fund += 6

    nm = _safe_float(f.get("net_margin"))
    if nm is not None:
        if   nm < 0:    fund += 12
        elif nm < 0.03: fund += 5

    rg = _safe_float(f.get("rev_growth"))
    if rg is not None:
        if   rg < -0.10: fund += 12
        elif rg < 0:     fund += 6
        elif rg > 0.20:  fund -= 8   # strong growth = risky short

    de = _safe_float(f.get("de_ratio"))
    if de is not None and de > 2.0: fund += 8

    fund = max(0, min(40, fund))
    bkdn["fundamental"] = fund
    score += fund

    # 3. Valuation stretched (max 20 pts)
    val = 0
    pe  = _safe_float(f.get("pe"))
    if pe and pe > 0:
        if   pe > 60: val += 20
        elif pe > 40: val += 12
        elif pe > 25: val += 5
        elif pe < 12: val -= 5  # cheap = bad short

    val = max(0, min(20, val))
    bkdn["valuation"] = val
    score += val

    return round(float(score), 1), bkdn


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def compute_correlation_matrix(symbols_tuple):
    """
    Fetch 6-month daily returns for all symbols and return correlation matrix.
    Accepts a tuple (hashable, required for st.cache_data).
    Returns (corr_df, warning_pairs) or (None, []).
    """
    symbols = list(symbols_tuple)
    try:
        if len(symbols) < 2:
            return None, []
        raw = yf.download(symbols, period="6mo", interval="1d", progress=False)
        if "Close" not in raw.columns and isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        elif isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
            prices = raw[["Close"]].rename(columns={"Close": symbols[0]})
        else:
            prices = raw.get("Close", raw)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=symbols[0])
        prices = prices.dropna(how="all", axis=1)
        if prices.shape[1] < 2:
            return None, []
        rets = prices.pct_change().dropna()
        corr = rets.corr().round(2)
        # Flag highly correlated pairs (risk: not truly diversified)
        warn = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if abs(v) > 0.70:
                    warn.append((cols[i], cols[j], round(float(v), 2)))
        return corr, warn
    except Exception:
        return None, []


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS BULLET GENERATOR  (used in both Tab 1 and Tab 2 stock cards)
# ══════════════════════════════════════════════════════════════════════════════

def generate_analysis_bullets(f, mc, score, bkdn, dcf=None, curr="$"):
    """
    Synthesise all model signals into ≤5 plain-English bullet points.
    Each bullet covers one dimension, ending with a clear action.
    """
    bullets = []
    price       = float(f.get("price")        or 0)
    rsi         = float(f.get("rsi")          or 50)
    above_ma50  = int(f.get("above_ma50")     or 0)
    above_ma200 = int(f.get("above_ma200")    or 0)
    mom_3m      = float(f.get("mom_3m")       or 0)
    mom_1m      = float(f.get("mom_1m")       or 0)
    rev_growth  = float(f.get("rev_growth")   or 0)
    net_margin  = float(f.get("net_margin")   or 0)
    beta        = float(f.get("beta")         or 1.0)
    n_ana       = int(f.get("n_analysts")     or 0)
    au          = float(f.get("analyst_upside") or 0)
    fund_score  = float(bkdn.get("fundamental", 50) or 50)
    tech_score  = float(bkdn.get("technical",  50) or 50)

    rr          = float(mc.get("rr_3m",     0) or 0) if mc else 0
    stop        = float(mc.get("stop_loss", price*0.93) or price*0.93) if mc else price*0.93
    risk_p      = float(mc.get("risk_pct",  7.0) or 7.0) if mc else 7.0
    mc3_base    = float(mc["3M"].get("ret_base",  0) or 0) if (mc and "3M" in mc) else 0
    mc3_p90     = float(mc["3M"].get("ret_p90",   0) or 0) if (mc and "3M" in mc) else 0
    mc3_prob    = float(mc["3M"].get("prob_positive", 50) or 50) if (mc and "3M" in mc) else 50
    prob_loss   = float(mc["3M"].get("prob_loss10", 0) or 0) if (mc and "3M" in mc) else 0
    tgt3m       = float(mc["3M"].get("price_base", price) or price) if (mc and "3M" in mc) else price
    tgt1y       = float(mc["1Y"].get("price_base", price) or price) if (mc and "1Y" in mc) else price
    mc1y_ret    = float(mc["1Y"].get("ret_base", 0) or 0) if (mc and "1Y" in mc) else 0

    ma50_v  = float(f.get("ma50")  or price)
    ma200_v = float(f.get("ma200") or price)

    # ── BULLET 1: Entry timing & technical stance ─────────────────────────────
    if rsi > 72:
        bullets.append(
            f"⚠️ **Overbought — wait for a pullback.** RSI {rsi:.0f} is above 70. "
            f"Price has moved too fast in the short term. A healthier entry is near "
            f"{curr}{max(stop*1.02, price*0.94):.2f}–{curr}{price*0.97:.2f}. "
            f"Chasing here risks buying the top of a short-term spike."
        )
    elif rsi < 32:
        bullets.append(
            f"✅ **Oversold — potential bounce zone.** RSI {rsi:.0f} is below 30. "
            f"Price is statistically stretched to the downside. "
            f"If the business is sound (check fundamentals), this is often a high-reward entry point. "
            f"Current price {curr}{price:.2f} is near short-term support."
        )
    elif above_ma50 and above_ma200:
        bullets.append(
            f"✅ **Clean technical setup.** Price {curr}{price:.2f} is above both MA50 and MA200 — "
            f"both short and long-term trends are aligned upward. RSI {rsi:.0f} is in the "
            f"healthy 40–65 range with room to run. Trend is your friend here."
        )
    elif not above_ma200 and above_ma50:
        bullets.append(
            f"⚠️ **Mixed signals — recovery in progress but not confirmed.** "
            f"Price {curr}{price:.2f} broke above MA50 ({curr}{ma50_v:.2f}) but is still below "
            f"MA200 ({curr}{ma200_v:.2f}). The long-term trend is still recovering. "
            f"A break above MA200 would be the stronger buy signal."
        )
    else:
        bullets.append(
            f"🔴 **Downtrend — price below both moving averages.** "
            f"Below MA50 ({curr}{ma50_v:.2f}) and MA200 ({curr}{ma200_v:.2f}). "
            f"Momentum is against you. Either wait for MA50 reclaim, or "
            f"use a very small starter position only."
        )

    # ── BULLET 2: Risk/Reward & position sizing ───────────────────────────────
    if rr >= 2.5:
        bullets.append(
            f"✅ **Excellent risk/reward at {rr:.1f}:1.** Stop loss at {curr}{stop:.2f} "
            f"({risk_p:.1f}% below entry) limits your downside. The 3M statistical base "
            f"case is {curr}{tgt3m:.2f} ({mc3_base:+.1f}%). "
            f"Sizing: risk no more than 1–2% of your portfolio — "
            f"that means a position of approximately "
            f"{curr}{(price * 0.02 / max(price - stop, 0.01) * price):.0f} total."
        )
    elif 1.0 <= rr < 2.5:
        bullets.append(
            f"⚠️ **Marginal risk/reward at {rr:.1f}:1** (minimum is 2:1). "
            f"Stop at {curr}{stop:.2f}, 3M base {curr}{tgt3m:.2f}. "
            f"If you enter, use a half-size position. "
            f"There is a {prob_loss:.0f}% statistical chance of a >10% loss in 3 months."
        )
    else:
        bullets.append(
            f"🔴 **Poor risk/reward at {rr:.1f}:1** — the statistical base case ({mc3_base:+.1f}% in 3M) "
            f"does not justify the risk of a {risk_p:.1f}% stop loss. "
            f"This is not a good entry point for new money. "
            f"Wait for either price to pull back or fundamentals to improve."
        )

    # ── BULLET 3: Fundamental quality & growth ────────────────────────────────
    if fund_score >= 60 and rev_growth > 0.10:
        bullets.append(
            f"✅ **Business quality supports the investment.** Revenue growing "
            f"{rev_growth*100:.0f}%+ YoY with net margin {net_margin*100:.0f}%. "
            f"Fundamental score {fund_score:.0f}/100. This is not a momentum gamble — "
            f"the underlying business justifies a long-term position. "
            f"{f'{n_ana} analysts cover this' if n_ana > 0 else 'Limited analyst coverage — potential discovery opportunity'}."
        )
    elif fund_score >= 40:
        bullets.append(
            f"⚪ **Adequate fundamentals ({fund_score:.0f}/100) — neither a red flag nor a standout.** "
            f"Revenue growth {rev_growth*100:+.0f}%, margin {net_margin*100:.0f}%. "
            f"The investment thesis relies more on price momentum than deep business quality. "
            f"Treat this as a technical trade with a strict stop, not a long-term hold."
        )
    else:
        bullets.append(
            f"🔴 **Weak fundamentals ({fund_score:.0f}/100) — momentum play only.** "
            f"Revenue growth {rev_growth*100:+.0f}%, margin {net_margin*100:.0f}%. "
            f"Only enter if the technical setup is perfect and with a tight stop. "
            f"Do not hold through a drawdown hoping fundamentals will bail you out."
        )

    # ── BULLET 4: Valuation & long-term upside ────────────────────────────────
    if dcf and dcf.get("upside_pct") is not None:
        upside = dcf["upside_pct"]
        intrinsic = dcf["intrinsic"]
        if upside > 20:
            bullets.append(
                f"✅ **Valuation: {upside:+.0f}% margin of safety.** "
                f"DCF intrinsic value is {curr}{intrinsic:.2f} vs current price {curr}{price:.2f}. "
                f"You are buying {abs(upside):.0f}% below what the business is worth based on "
                f"future cash flows. This is the strongest long-term buy signal. "
                f"Even if the stock stays flat for 6 months, the value is here."
            )
        elif -10 <= upside <= 20:
            bullets.append(
                f"⚪ **Fairly valued — limited margin of safety.** "
                f"DCF intrinsic {curr}{intrinsic:.2f}, current {curr}{price:.2f} ({upside:+.0f}%). "
                f"The stock is roughly fairly priced. "
                f"Upside depends on growth exceeding current expectations, not just mean reversion. "
                f"Analysts see {au*100:.0f}% upside to mean target."
            )
        else:
            bullets.append(
                f"🔴 **Overvalued by DCF — {abs(upside):.0f}% above intrinsic value.** "
                f"Model fair value {curr}{intrinsic:.2f} vs current {curr}{price:.2f}. "
                f"Premium may be justified by growth expectations, but leaves little room for error. "
                f"Any earnings miss could trigger a sharp de-rating."
            )
    else:
        # Use Monte Carlo 1Y + analyst consensus as proxy
        if mc1y_ret > 15:
            bullets.append(
                f"✅ **1-year outlook positive.** Monte Carlo median path: {mc1y_ret:+.1f}% in 12 months. "
                f"{'Analysts see ' + str(round(au*100,0)) + '% upside to mean target. ' if n_ana > 3 else ''}"
                f"The statistical distribution favours a recovery over this horizon."
            )
        elif mc1y_ret > -10:
            bullets.append(
                f"⚪ **1-year outlook: flat to modest.** Blended MC median path: {mc1y_ret:+.1f}% "
                f"over 12 months. The statistical base case suggests patience rather than quick profits. "
                f"{'Analyst consensus: ' + str(round(au*100,0)) + '% implied upside.' if n_ana > 3 else ''}"
            )
        else:
            bullets.append(
                f"⚠️ **1-year statistical base case is negative ({mc1y_ret:+.1f}%).** "
                f"The blended Monte Carlo (historical + fundamental drift) predicts the median path "
                f"declines over 12 months. Fundamental recovery needs to outpace this drag. "
                f"Requires high conviction in the business thesis to hold for a year."
            )

    # ── BULLET 5: Final recommended action ───────────────────────────────────
    is_buy  = score >= 60 and rr >= 2.0 and (above_ma50 or rsi < 50) and fund_score >= 40
    is_wait = score >= 55 and (rsi > 68 or rr < 2.0 or (not above_ma50 and mom_1m < -0.05))
    is_dca  = score >= 60 and not above_ma200 and fund_score >= 55

    if is_buy:
        bullets.append(
            f"🎯 **RECOMMENDED ACTION: BUY** — Enter near {curr}{price:.2f} with stop at "
            f"{curr}{stop:.2f}. Use 1–2% of portfolio risk. "
            f"Score {score:.0f}/100 with aligned signals across technicals and fundamentals. "
            f"Review position if price falls below stop or at next earnings."
        )
    elif is_dca:
        bullets.append(
            f"🎯 **RECOMMENDED ACTION: PHASE IN (DCA)** — Business quality justifies entry "
            f"but price is still recovering (below MA200 {curr}{ma200_v:.2f}). "
            f"Buy 1/3 now at {curr}{price:.2f}, add 1/3 on MA50 reclaim, final 1/3 on MA200 breakout. "
            f"This spreads your risk across the recovery. Stop on full position: {curr}{stop:.2f}."
        )
    elif is_wait:
        reclaim = max(ma50_v, price * 0.97)
        bullets.append(
            f"🎯 **RECOMMENDED ACTION: WAIT** — Setup needs one more confirming signal. "
            f"{'RSI ' + str(round(rsi)) + ' overbought — wait for pullback to ' + curr + str(round(reclaim, 2)) + '. ' if rsi > 68 else ''}"
            f"{'R/R ' + str(round(rr, 1)) + ':1 below 2:1 — entry here is not efficient. ' if rr < 2.0 else ''}"
            f"Set a price alert and revisit in 1–2 weeks."
        )
    else:
        bullets.append(
            f"🎯 **RECOMMENDED ACTION: AVOID** — Score {score:.0f}/100, R/R {rr:.1f}:1. "
            f"Signals are not aligned enough to justify capital allocation. "
            f"There are better setups available. Revisit after next earnings or "
            f"when price reclaims {curr}{ma50_v:.2f} (MA50)."
        )

    return bullets[:5]


# ══════════════════════════════════════════════════════════════════════════════
# DCF VALUATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_dcf(f):
    """
    Simple 5-year DCF + terminal value using yfinance fundamental data.
    Returns dict with intrinsic_value, upside_pct, verdict — or None if data missing.
    """
    try:
        price  = f.get("price", 0)
        mktcap = _safe_float(f.get("mktcap"))
        if not price or not mktcap or price <= 0 or mktcap <= 0:
            return None

        # FCF from FCF yield × market cap
        fcf_yield = _safe_float(f.get("fcf_yield"))
        if fcf_yield and fcf_yield > 0:
            base_fcf = fcf_yield * mktcap
        else:
            return None   # can't do DCF without FCF

        rev_g  = _safe_float(f.get("rev_growth"))  or 0.10
        earn_g = _safe_float(f.get("earn_growth")) or 0.08
        beta   = _safe_float(f.get("beta"))        or 1.0

        # WACC: risk-free (4.5% US / 7% India) + beta × equity risk premium (5.5%)
        is_india  = str(f.get("exchange","")).upper() in ("NSE","BSE") or \
                    str(f.get("symbol","")).endswith(".NS")
        risk_free = 0.070 if is_india else 0.045
        erp       = 0.055
        wacc      = risk_free + beta * erp
        wacc      = max(0.07, min(0.20, wacc))   # clamp to realistic range

        # Project FCF for 5 years (growth decelerates toward terminal rate)
        growth_years = [
            min(rev_g,        0.40),
            min(rev_g * 0.85, 0.35),
            min(rev_g * 0.70, 0.30),
            min(rev_g * 0.55, 0.25),
            min(rev_g * 0.40, 0.20),
        ]
        pv_fcf   = 0.0
        proj_fcf = base_fcf
        for yr, g in enumerate(growth_years, 1):
            proj_fcf *= (1 + g)
            pv_fcf   += proj_fcf / (1 + wacc) ** yr

        # Terminal value (Gordon Growth, terminal g = 3%)
        terminal_g  = 0.03
        terminal_tv = proj_fcf * (1 + terminal_g) / (wacc - terminal_g)
        pv_terminal = terminal_tv / (1 + wacc) ** 5

        total_equity_value = pv_fcf + pv_terminal
        shares             = mktcap / price
        intrinsic          = total_equity_value / shares

        upside_pct = (intrinsic - price) / price * 100

        if   upside_pct > 30:  verdict = "STRONG BUY"
        elif upside_pct > 10:  verdict = "BUY"
        elif upside_pct > -10: verdict = "HOLD"
        elif upside_pct > -25: verdict = "SELL"
        else:                  verdict = "STRONG SELL"

        return {
            "intrinsic":  round(intrinsic, 2),
            "upside_pct": round(upside_pct, 1),
            "wacc":       round(wacc * 100, 1),
            "base_fcf":   round(base_fcf / 1e9, 2),
            "verdict":    verdict,
        }
    except Exception:
        return None


def compute_relative_valuation(f):
    """
    Compare key multiples to broad sector benchmarks.
    Returns (verdict, details_list).
    """
    signals = []
    score   = 0

    pe     = _safe_float(f.get("pe"))
    fwd_pe = _safe_float(f.get("fwd_pe"))
    pb     = _safe_float(f.get("pb"))
    peg    = _safe_float(f.get("peg"))
    ps     = _safe_float(f.get("priceToSalesTrailingTwelveMonths") or
                         f.get("ev_ebitda"))

    # Broad benchmarks (varies by sector but these are reasonable market medians)
    BENCH = {"pe": 22, "fwd_pe": 18, "pb": 3.5, "peg": 1.5}

    if pe and pe > 0:
        disc = (BENCH["pe"] - pe) / BENCH["pe"] * 100
        if   pe < BENCH["pe"] * 0.7:  score += 2; signals.append(f"P/E {pe:.1f}x — **30%+ below market median** ({BENCH['pe']}x) ✅")
        elif pe < BENCH["pe"]:         score += 1; signals.append(f"P/E {pe:.1f}x — below market median ✅")
        elif pe < BENCH["pe"] * 1.5:  score -= 1; signals.append(f"P/E {pe:.1f}x — above market median ⚠️")
        else:                          score -= 2; signals.append(f"P/E {pe:.1f}x — significantly above market median ❌")

    if peg and peg > 0:
        if   peg < 1.0: score += 2; signals.append(f"PEG {peg:.2f} — cheap relative to growth ✅")
        elif peg < 1.5: score += 1; signals.append(f"PEG {peg:.2f} — fair relative to growth")
        elif peg < 2.5: score -= 1; signals.append(f"PEG {peg:.2f} — expensive vs growth ⚠️")
        else:           score -= 2; signals.append(f"PEG {peg:.2f} — very expensive vs growth ❌")

    if pb and pb > 0:
        if   pb < 1:              score += 2; signals.append(f"P/B {pb:.2f}x — trading below book value ✅")
        elif pb < BENCH["pb"]:    score += 1; signals.append(f"P/B {pb:.2f}x — below median ✅")
        elif pb < BENCH["pb"]*2:  score -= 1; signals.append(f"P/B {pb:.2f}x — above median ⚠️")
        else:                     score -= 2; signals.append(f"P/B {pb:.2f}x — significantly above median ❌")

    max_score = max(len(signals) * 2, 1)
    pct       = score / max_score * 100

    if   pct >= 50:  verdict = "CHEAP"
    elif pct >= 10:  verdict = "FAIR"
    elif pct >= -20: verdict = "FAIRLY PRICED"
    elif pct >= -50: verdict = "EXPENSIVE"
    else:            verdict = "VERY EXPENSIVE"

    return verdict, signals, round(pct, 0)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS SIGNALS  (smart-money tells: IV rank, put/call ratio, skew)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def compute_options_signals(sym, current_price):
    """
    Fetch options chain and compute:
      - ATM implied volatility  (current fear/greed level)
      - IV vs HV ratio          (options expensive or cheap vs realised vol)
      - Put/call volume ratio   (>1 = fear/hedging, <0.7 = greed/complacency)
      - OTM skew                (put IV - call IV: positive = crash protection bid)
      - Unusual activity count  (volume > 3× open interest = fresh positioning)
    Returns dict or None if options unavailable.
    """
    try:
        tk   = yf.Ticker(sym)
        exps = tk.options
        if not exps:
            return None

        # Pick expiry closest to 30 days out
        today  = datetime.date.today()
        target = today + datetime.timedelta(days=30)
        exp    = min(exps, key=lambda e: abs(
            (datetime.datetime.strptime(e, "%Y-%m-%d").date() - target).days
        ))
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        if calls.empty or puts.empty:
            return None

        cp = float(current_price)

        # ATM IV: average of closest call and put strikes to current price
        calls["mon"] = (calls["strike"] - cp).abs()
        puts ["mon"] = (puts ["strike"] - cp).abs()
        atm_c_iv = float(calls.nsmallest(1, "mon")["impliedVolatility"].values[0])
        atm_p_iv = float(puts .nsmallest(1, "mon")["impliedVolatility"].values[0])
        atm_iv   = round((atm_c_iv + atm_p_iv) / 2 * 100, 1)   # in %

        # Put/call volume ratio (all strikes)
        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol  = float(puts ["volume"].fillna(0).sum())
        pc_ratio = round(put_vol / call_vol, 2) if call_vol > 0 else None

        # OTM skew: 5% OTM put IV minus 5% OTM call IV
        otm_put_rows  = puts [puts ["strike"] <= cp * 0.95].nlargest(1, "strike")
        otm_call_rows = calls[calls["strike"] >= cp * 1.05].nsmallest(1, "strike")
        skew = None
        if not otm_put_rows.empty and not otm_call_rows.empty:
            skew = round(
                (float(otm_put_rows ["impliedVolatility"].values[0]) -
                 float(otm_call_rows["impliedVolatility"].values[0])) * 100, 1
            )

        # Unusual activity: volume > 3× open interest (fresh large positioning)
        calls["voi"] = calls["volume"].fillna(0) / (calls["openInterest"].fillna(1) + 1)
        puts ["voi"] = puts ["volume"].fillna(0) / (puts ["openInterest"].fillna(1) + 1)
        unusual_calls = int((calls["voi"] > 3).sum())
        unusual_puts  = int((puts ["voi"] > 3).sum())

        return {
            "atm_iv":        atm_iv,           # current ATM IV in %
            "pc_ratio":      pc_ratio,          # put/call volume ratio
            "skew":          skew,              # OTM put IV - OTM call IV (%)
            "unusual_calls": unusual_calls,     # count of unusual call strikes
            "unusual_puts":  unusual_puts,      # count of unusual put strikes
            "exp_date":      exp,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# JUMP DIFFUSION (Merton model)  — handles earnings gaps that GBM cannot
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_jump_diffusion(f, n_simulations=5000):
    """
    Merton (1976) Jump-Diffusion model.

    dS/S = (μ - λk̄) dt + σ dW + dJ

    Where dJ is a compound Poisson process: each year the stock experiences
    ~3 meaningful jump events (earnings, macro shocks).  Jump sizes are
    log-normal with mean mu_j and std sigma_j.

    Uses the same blended drift as monte_carlo_targets() so both models
    start from the same expected return — the difference is path shape.
    """
    if f is None:
        return None
    try:
        price  = float(f["price"])
        sigma  = float(f.get("hist_std") or 0.015) or 0.015
        mu_base = float(f.get("hist_mean") or 0.0003)

        # Reuse blended drift from standard MC
        fund_daily, _, _ = compute_fundamental_drift_adj(f)
        mom_tilt = sum(
            float(np.clip(f.get(k) or 0, -0.5, 0.5)) * w
            for k, w in [("mom_1m", 0.3), ("mom_3m", 0.4), ("mom_6m", 0.3)]
        )
        mu_adj = mu_base * 0.20 + fund_daily * 0.50 + (mom_tilt / 252) * 0.30
        rsi = float(f.get("rsi") or 50)
        if rsi > 75:   mu_adj -= 0.0003
        elif rsi < 25: mu_adj += 0.0003

        # Jump parameters (calibrated to average S&P500 stock behaviour)
        lambda_j = 3.0     # ~3 jump events per year (earnings + macro)
        mu_j     = -0.015  # mean log-jump: stocks gap down more than up
        sigma_j  = 0.07    # typical earnings gap ±7%
        dt       = 1 / 252

        # Merton drift correction: compensate for expected jump contribution
        k_bar   = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
        mu_corr = mu_adj - lambda_j * k_bar   # so drift is unbiased

        np.random.seed(int(abs(price * 100)) % 99991 + datetime.date.today().toordinal() + 1)

        T = 252
        # Diffusion shocks (n_simulations × T)
        dW = np.random.normal(
            (mu_corr - 0.5 * sigma ** 2) * dt,
            sigma * np.sqrt(dt),
            (n_simulations, T)
        )
        # Poisson jump counts per step
        n_jumps  = np.random.poisson(lambda_j * dt, (n_simulations, T))
        # Jump magnitudes (log-normal)
        j_sizes  = np.random.normal(mu_j, sigma_j, (n_simulations, T))
        jump_log = np.where(n_jumps > 0, n_jumps * j_sizes, 0.0)

        log_ret = np.cumsum(dW + jump_log, axis=1)
        paths   = price * np.exp(log_ret)
        paths   = np.maximum(paths, 0.01)

        results = {}
        for label, days in [("1D", 1), ("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
            fin = paths[:, days - 1]
            p50 = float(np.percentile(fin, 50))
            p90 = float(np.percentile(fin, 90))
            results[label] = {
                "price_base": round(p50, 2),
                "price_p90":  round(p90, 2),
                "ret_base":   round((p50 / price - 1) * 100, 1),
                "ret_p90":    round((p90 / price - 1) * 100, 1),
                "prob_positive": round(float(np.mean(fin > price)) * 100, 1),
            }
        return results
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# KELLY CRITERION  — mathematical position sizing
# ══════════════════════════════════════════════════════════════════════════════

def compute_kelly(mc, f):
    """
    Fractional Kelly Criterion for position sizing.

    Full Kelly:  f* = (p·b - q) / b
      p = probability of a positive outcome (from MC 3M)
      q = 1 - p
      b = expected win / expected loss (risk-reward ratio)

    We use 25% fractional Kelly (quarter-Kelly) to account for model
    uncertainty and real-world slippage.  Capped at 15% of portfolio.
    """
    try:
        p = float(mc["3M"].get("prob_positive", 50)) / 100
        q = 1 - p

        entry  = float(mc.get("entry", f.get("price", 100)))
        stop   = float(mc.get("stop_loss", entry * 0.93))
        target = float(mc["3M"].get("price_p90", entry * 1.10))

        loss_pct = max((entry - stop)  / entry, 0.001)
        win_pct  = max((target - entry) / entry, 0.001)
        b        = win_pct / loss_pct            # odds: win per unit of risk

        kelly_full      = (p * b - q) / b
        kelly_quarter   = kelly_full * 0.25      # quarter-Kelly: safer in practice
        kelly_quarter   = max(0.0, min(kelly_quarter, 0.15))   # hard cap at 15%

        # Interpretation
        if kelly_quarter >= 0.10:
            verdict = "LARGE"
            color   = "#00c853"
        elif kelly_quarter >= 0.05:
            verdict = "MEDIUM"
            color   = "#ffcc00"
        elif kelly_quarter > 0:
            verdict = "SMALL"
            color   = "#ff9800"
        else:
            verdict = "SKIP"
            color   = "#ff5252"

        return {
            "kelly_full":     round(kelly_full * 100, 1),
            "kelly_quarter":  round(kelly_quarter * 100, 1),
            "verdict":        verdict,
            "color":          color,
            "win_prob":       round(p * 100, 1),
            "win_loss_ratio": round(b, 2),
            "loss_pct":       round(loss_pct * 100, 1),
            "win_pct":        round(win_pct * 100, 1),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EARNINGS ESTIMATE REVISIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def compute_earnings_revision(sym):
    """
    Compare current EPS estimates to 30 and 60 days ago.
    Rising estimates = analysts getting more bullish = leading indicator.
    Returns dict or None.
    """
    try:
        tk    = yf.Ticker(sym)
        trend = tk.eps_trend
        if trend is None or trend.empty:
            return None

        results = {}
        for period in trend.columns.tolist():
            row = trend[period]
            curr_est = _safe_float(row.get("current"))
            ago_30   = _safe_float(row.get("30daysAgo"))
            ago_60   = _safe_float(row.get("60daysAgo"))
            if curr_est is None or curr_est == 0:
                continue
            rev_30 = round((curr_est - ago_30) / abs(ago_30) * 100, 1) if ago_30 else None
            rev_60 = round((curr_est - ago_60) / abs(ago_60) * 100, 1) if ago_60 else None
            results[str(period)] = {
                "current": round(curr_est, 2),
                "rev_30d": rev_30,
                "rev_60d": rev_60,
            }
        return results if results else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL DRIFT ENGINE  (the missing piece that bridges MC ↔ AI analysis)
# ══════════════════════════════════════════════════════════════════════════════

def compute_fundamental_drift_adj(f):
    """
    Translate business quality, valuation, growth, macro and analyst consensus
    into an implied annual expected return, then convert to daily drift for MC.

    This is the bridge between what the BUSINESS is worth (fundamentals)
    and what the PRICE has been doing (historical drift).

    Returns:
        daily_adj   : float — daily drift adjustment to add to historical mu
        annual_exp  : float — implied annual return from fundamentals alone
        factors     : dict  — {factor_name: (contribution, description)}
    """
    annual_exp = 0.0
    factors    = {}

    def clip(v, lo, hi): return float(np.clip(v, lo, hi))

    # ── 1. VALUATION — PEG / PE vs fair value ────────────────────────────────
    # Low PEG = paying less per unit of growth = bullish drift
    peg = _safe_float(f.get("peg"))
    pe  = _safe_float(f.get("pe"))
    if peg and peg > 0:
        if   peg < 0.8:  va = 0.18;  lbl = f"PEG {peg:.2f} — deep value, paying 80c per $1 of growth"
        elif peg < 1.2:  va = 0.08;  lbl = f"PEG {peg:.2f} — fair value"
        elif peg < 2.0:  va = 0.0;   lbl = f"PEG {peg:.2f} — slightly expensive"
        elif peg < 3.0:  va = -0.08; lbl = f"PEG {peg:.2f} — expensive"
        else:            va = -0.15; lbl = f"PEG {peg:.2f} — significantly overvalued"
        annual_exp += va
        factors["📐 Valuation (PEG)"] = (va, lbl)
    elif pe and pe > 0:
        if   pe < 12:  va = 0.15;  lbl = f"P/E {pe:.1f}x — deep value (<12x)"
        elif pe < 20:  va = 0.08;  lbl = f"P/E {pe:.1f}x — fair value (12–20x)"
        elif pe < 30:  va = 0.02;  lbl = f"P/E {pe:.1f}x — moderate premium"
        elif pe < 50:  va = -0.06; lbl = f"P/E {pe:.1f}x — elevated (30–50x)"
        else:          va = -0.12; lbl = f"P/E {pe:.1f}x — very expensive (>50x)"
        annual_exp += va
        factors["📐 Valuation (PE)"] = (va, lbl)

    # ── 2. ANALYST CONSENSUS — collective price targets ───────────────────────
    # Analysts model DCF, comparable multiples — their targets encode fundamentals
    au     = _safe_float(f.get("analyst_upside"))
    n_ana  = int(f.get("n_analysts") or 0)
    if au is not None and n_ana >= 3:
        weight = min(0.40, 0.10 + n_ana * 0.02)  # more analysts = higher weight, max 40%
        aa = clip(au * weight, -0.15, 0.20)
        lbl = f"{au*100:.1f}% consensus upside from {n_ana} analysts (weight {weight:.0%})"
        annual_exp += aa
        factors["🔬 Analyst Consensus"] = (aa, lbl)

    # ── 3. REVENUE GROWTH — top-line momentum ────────────────────────────────
    rev_g = _safe_float(f.get("rev_growth"))
    if rev_g is not None:
        # 20% revenue growth historically justifies ~12% incremental return premium
        ga = clip(rev_g * 0.60, -0.12, 0.18)
        lbl = f"Revenue growing {rev_g*100:.1f}% YoY"
        annual_exp += ga
        factors["📈 Revenue Growth"] = (ga, lbl)

    # ── 4. EARNINGS GROWTH — bottom-line execution ───────────────────────────
    earn_g = _safe_float(f.get("earn_growth"))
    if earn_g is not None:
        ea = clip(earn_g * 0.35, -0.10, 0.12)
        lbl = f"Earnings growing {earn_g*100:.1f}% YoY"
        annual_exp += ea
        factors["💰 Earnings Growth"] = (ea, lbl)

    # ── 5. BUSINESS QUALITY — ROE + net margin ───────────────────────────────
    roe = _safe_float(f.get("roe"))
    nm  = _safe_float(f.get("net_margin"))
    if roe is not None:
        if   roe > 0.30: qa = 0.05;  lbl = f"ROE {roe*100:.1f}% — elite capital compounder"
        elif roe > 0.15: qa = 0.02;  lbl = f"ROE {roe*100:.1f}% — quality business"
        elif roe > 0.05: qa = 0.0;   lbl = f"ROE {roe*100:.1f}% — average"
        else:            qa = -0.03; lbl = f"ROE {roe*100:.1f}% — below average"
        annual_exp += qa
        factors["🏦 Business Quality (ROE)"] = (qa, lbl)
    if nm is not None and nm < -0.05:
        annual_exp -= 0.08
        factors["⚠️ Loss-Making"] = (-0.08, f"Net margin {nm*100:.1f}% — company not profitable")

    # ── 6. FREE CASH FLOW YIELD — real cash generation ───────────────────────
    fcf_y = _safe_float(f.get("fcf_yield"))
    if fcf_y and fcf_y > 0:
        fa = clip(fcf_y * 0.8, 0, 0.06)
        lbl = f"FCF yield {fcf_y*100:.1f}% — strong cash generation"
        annual_exp += fa
        factors["💵 FCF Yield"] = (fa, lbl)

    # ── 7. BALANCE SHEET — leverage risk ────────────────────────────────────
    de = _safe_float(f.get("de_ratio"))
    if de is not None:
        if   de > 300: ba = -0.08; lbl = f"D/E {de:.0f} — dangerous leverage"
        elif de > 150: ba = -0.04; lbl = f"D/E {de:.0f} — high leverage"
        elif de > 80:  ba = -0.01; lbl = f"D/E {de:.0f} — elevated but manageable"
        elif de < 30:  ba = 0.02;  lbl = f"D/E {de:.0f} — fortress balance sheet"
        else:          ba = 0.0;   lbl = f"D/E {de:.0f} — normal leverage"
        annual_exp += ba
        factors["🏛️ Balance Sheet"] = (ba, lbl)

    # ── 8. MACRO REGIME — VIX-driven risk premium ────────────────────────────
    try:
        vix_now = fetch_vix()
        regime_name = "normal"
        for rname, rcfg in VIX_REGIMES.items():
            if vix_now <= rcfg["max"]:
                regime_name = rname
                break
        macro_adj = {"calm": 0.04, "normal": 0.0, "fearful": -0.06, "crisis": -0.14}[regime_name]
        lbl = f"VIX {vix_now:.1f} — {regime_name} regime ({'tailwind' if macro_adj > 0 else 'headwind' if macro_adj < 0 else 'neutral'})"
        annual_exp += macro_adj
        factors["🌍 Macro/VIX Regime"] = (macro_adj, lbl)
    except Exception:
        pass

    # Convert annual implied return to daily drift
    daily_adj = annual_exp / 252
    return daily_adj, annual_exp, factors


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO PRICE TARGET ENGINE  (now blends fundamentals into drift)
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_targets(f, n_simulations=10000):
    """
    Student's t Monte Carlo simulation with BLENDED drift:

      Drift = 20% historical price mean
            + 50% fundamental-implied drift  (PE, growth, ROE, analyst targets, macro)
            + 30% momentum signal             (recent price direction)

    This means a falling stock with strong fundamentals (e.g. ServiceNow) will
    show a much better median path than a purely historical model would predict —
    reflecting that the business justifies recovery.
    """
    if f is None:
        return None

    price = f["price"]
    mu    = float(f.get("hist_mean") or 0.0003)   # historical daily mean
    sigma = float(f.get("hist_std")  or 0.015) or 0.015
    kurt  = float(f.get("hist_kurt") or 0.0)

    # ── Component 1: Historical drift (20%) ──────────────────────────────────
    hist_daily = mu * 0.20

    # ── Component 2: Fundamental-implied drift (50%) ─────────────────────────
    fund_daily, fund_annual, drift_factors = compute_fundamental_drift_adj(f)
    fund_daily_weighted = fund_daily * 0.50

    # ── Component 3: Momentum signal (30%) ───────────────────────────────────
    mom_tilt = 0.0
    for key, wt in [("mom_1m", 0.3), ("mom_3m", 0.4), ("mom_6m", 0.3)]:
        v = f.get(key)
        if v is not None:
            mom_tilt += float(np.clip(v, -0.5, 0.5)) * wt
    mom_daily_weighted = (mom_tilt / 252) * 0.30

    # ── Blended drift ─────────────────────────────────────────────────────────
    mu_adj = hist_daily + fund_daily_weighted + mom_daily_weighted

    # ── RSI overbought / oversold micro-adjustment ────────────────────────────
    rsi = float(f.get("rsi") or 50)
    if rsi > 75:   mu_adj -= 0.0003   # overbought → slight negative nudge
    elif rsi < 25: mu_adj += 0.0003   # oversold   → slight positive nudge

    # ── Simulation ───────────────────────────────────────────────────────────
    df_t     = max(3, int(30 - abs(kurt)))
    horizons = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    results  = {}
    # Seed based on price + date so results refresh daily but are reproducible
    np.random.seed(int(abs(price * 100)) % 99991 + datetime.date.today().toordinal())

    for label, days in horizons.items():
        shocks     = stats.t.rvs(df=df_t, size=(n_simulations, days))
        shocks     = shocks * sigma + mu_adj
        cum_return = np.prod(1 + shocks, axis=1)
        fin_prices = price * cum_return
        returns    = cum_return - 1

        def pp(p): return round(float(np.percentile(fin_prices, p)), 2)
        def rp(p): return round(float(np.percentile(returns,    p)) * 100, 1)
        def prob(thr): return round(float(np.mean(returns > thr)) * 100, 1)

        results[label] = {
            "price_bear": pp(10),  "price_p25": pp(25),
            "price_base": pp(50),  "price_p75": pp(75),
            "price_p90":  pp(90),  "price_p95": pp(95),  "price_p99": pp(99),
            "ret_bear":   rp(10),  "ret_p25":   rp(25),
            "ret_base":   rp(50),  "ret_p75":   rp(75),
            "ret_p90":    rp(90),  "ret_p95":   rp(95),  "ret_p99":   rp(99),
            "prob_positive": prob(0.00),
            "prob_5pct":     prob(0.05),  "prob_10pct": prob(0.10),
            "prob_15pct":    prob(0.15),  "prob_20pct": prob(0.20),
            "prob_25pct":    prob(0.25),  "prob_30pct": prob(0.30),
            "prob_40pct":    prob(0.40),  "prob_50pct": prob(0.50),
            "prob_75pct":    prob(0.75),  "prob_100pct": prob(1.00),
            "prob_loss10":   round(float(np.mean(returns < -0.10)) * 100, 1),
            "prob_loss20":   round(float(np.mean(returns < -0.20)) * 100, 1),
        }

    atr_pct   = (f.get("atr_pct", 2.0) or 2.0) / 100
    stop_loss = min(price * (1 - 1.5 * atr_pct), price * 0.93)
    stop_loss = round(stop_loss, 2)
    risk_pct  = round((price - stop_loss) / price * 100, 1)
    target_3m = results["3M"]["price_base"]
    rr        = round((target_3m - price) / max(price - stop_loss, 0.01), 2)

    results["entry"]         = price
    results["stop_loss"]     = stop_loss
    results["risk_pct"]      = risk_pct
    results["rr_3m"]         = rr
    # Drift breakdown — exposed for display in the card
    results["drift"] = {
        "hist_pct":     round(hist_daily * 252 * 100, 2),
        "fund_pct":     round(fund_daily * 252 * 100, 2),
        "fund_annual":  round(fund_annual * 100, 1),
        "mom_pct":      round(mom_daily_weighted / 0.30 * 252 * 100, 2),
        "blended_pct":  round(mu_adj * 252 * 100, 2),
        "factors":      drift_factors,
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def get_xgb_prediction(f, models):
    """Use trained XGBoost models to predict forward returns."""
    if not XGB_AVAILABLE or not models:
        return None
    try:
        row = np.array(
            [float(f.get(col) or 0.0) for col in FEATURE_COLS]
        ).reshape(1, -1)
        return {
            "pred_1m": float(models["1m"].predict(row)[0]),
            "pred_3m": float(models["3m"].predict(row)[0]),
            "pred_1y": float(models["1y"].predict(row)[0]),
        }
    except Exception:
        return None


def blend_predictions(mc_result, xgb_pred, horizon="3M"):
    """
    Blend Monte Carlo P50 with XGBoost prediction.
    60% XGBoost + 40% Monte Carlo when XGB available.
    Returns (blended_price, blended_return_pct).
    """
    if mc_result is None:
        return None, None

    price  = mc_result["entry"]
    mc_ret = mc_result[horizon]["ret_base"] / 100

    if xgb_pred:
        key      = {"1M": "pred_1m", "3M": "pred_3m", "1Y": "pred_1y"}.get(horizon)
        xgb_ret  = xgb_pred.get(key, mc_ret) if key else mc_ret
        blended  = 0.60 * xgb_ret + 0.40 * mc_ret
    else:
        blended  = mc_ret

    return round(price * (1 + blended), 2), round(blended * 100, 1)


@st.cache_data(ttl=86400, show_spinner=False)
def train_xgb_models(universe_sample):
    """Train XGBoost models on walk-forward data. Returns model dict or None."""
    if not XGB_AVAILABLE:
        return None

    paths = {k: MODEL_DIR / f"xgb_{k}.pkl" for k in ["1m", "3m", "1y"]}
    if all(p.exists() for p in paths.values()):
        try:
            loaded = {}
            for k, p in paths.items():
                with open(p, "rb") as fh:
                    loaded[k] = pickle.load(fh)
            return loaded
        except Exception:
            pass

    X, y_1m, y_3m, y_1y = [], [], [], []

    for sym in universe_sample[:40]:
        try:
            df_h = yf.Ticker(sym).history(period="5y")
            if df_h is None or len(df_h) < 300:
                continue
            close = df_h["Close"].dropna()

            for i in range(200, len(close) - 252, 21):
                sub = close.iloc[:i]
                if len(sub) < 100:
                    continue

                def pct_sub(d):
                    return float((sub.iloc[-1] / sub.iloc[-d - 1]) - 1) \
                           if len(sub) > d else 0.0

                dr = sub.pct_change().dropna()
                row_feats = (
                    [pct_sub(5), pct_sub(21), pct_sub(63), pct_sub(126), pct_sub(252),
                     float(dr.mean()), float(dr.std() or 0.01),
                     float(stats.skew(dr)), float(stats.kurtosis(dr))]
                    + [0.0] * (len(FEATURE_COLS) - 9)
                )

                def fwd(d):
                    idx = min(i + d, len(close) - 1)
                    return float((close.iloc[idx] / close.iloc[i]) - 1)

                X.append(row_feats[:len(FEATURE_COLS)])
                y_1m.append(fwd(21))
                y_3m.append(fwd(63))
                y_1y.append(fwd(252))
        except Exception:
            continue

    if len(X) < 100:
        return None

    X_arr = np.array(X)
    split = int(len(X_arr) * 0.80)
    models = {}

    for label, y_arr in [("1m", y_1m), ("3m", y_3m), ("1y", y_1y)]:
        y     = np.array(y_arr)
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        model.fit(X_arr[:split], y[:split],
                  eval_set=[(X_arr[split:], y[split:])],
                  verbose=False)
        models[label] = model
        try:
            with open(paths[label], "wb") as fh:
                pickle.dump(model, fh)
        except Exception:
            pass

    return models


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGGER & VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions():
    """Load predictions from JSON file."""
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE, "r") as fh:
            return json.load(fh)
    except Exception:
        return []


def save_predictions(preds):
    """Save predictions to JSON file."""
    try:
        with open(PREDICTIONS_FILE, "w") as fh:
            json.dump(preds, fh, indent=2, default=str)
    except Exception:
        pass


def log_prediction(f, mc_result, bucket_key, score, percentile_class,
                   jd_result=None, bkdn=None, xgb_pred=None):
    """
    Log a new prediction entry. Returns the entry id.
    Stores predictions for all models so the tracker can compare them daily.
    """
    preds     = load_predictions()
    entry_px  = mc_result["entry"]
    currency  = f["currency"]

    # Daily blended drift (for per-day interpolation in tracker)
    daily_drift = mc_result.get("drift", {}).get("blended_pct", 0.0) / 100 / 252

    # Score-implied annual returns (for Tech/Fund tracker rows)
    tech_annual = (float((bkdn or {}).get("technical",   50) or 50) - 50) / 50 * 0.20
    fund_annual = (float((bkdn or {}).get("fundamental", 50) or 50) - 50) / 50 * 0.30

    def _hor(mc, h):
        if not mc or h not in mc: return None
        return {"price": round(float(mc[h].get("price_base", entry_px)), 2),
                "ret":   round(float(mc[h].get("ret_base",   0)),        2)}

    def _hor90(mc, h):
        if not mc or h not in mc: return None
        return {"price": round(float(mc[h].get("price_p90", entry_px)), 2),
                "ret":   round(float(mc[h].get("ret_p90",   0)),        2)}

    HORIZONS = ["1D", "1W", "1M", "3M", "1Y"]

    entry = {
        "id":          str(uuid.uuid4())[:8],
        "logged_at":   datetime.datetime.now().isoformat(),
        "symbol":      f["symbol"],
        "name":        f.get("name", f["symbol"]),
        "bucket":      bucket_key,
        "score":       score,
        "percentile":  percentile_class,
        "entry_price": entry_px,
        "stop_loss":   mc_result["stop_loss"],
        "currency":    currency,
        "daily_drift": daily_drift,
        "tech_annual": tech_annual,
        "fund_annual": fund_annual,
        # All-horizon predictions per model
        "mc_p50": {h: _hor(mc_result,   h) for h in HORIZONS if _hor(mc_result, h)},
        "mc_p90": {h: _hor90(mc_result, h) for h in HORIZONS if _hor90(mc_result, h)},
        "jd_p50": ({h: {"price": round(float(jd_result[h]["price_base"]), 2),
                         "ret":   round(float(jd_result[h]["ret_base"]),   2)}
                    for h in ["1D","1W","1M","3M","1Y"] if h in jd_result}
                   if jd_result else {}),
        "xgb":    ({h: {"price": round(entry_px * (1 + float(xgb_pred.get(k, 0))), 2),
                         "ret":   round(float(xgb_pred.get(k, 0)) * 100, 2)}
                    for h, k in [("1M","pred_1m"),("3M","pred_3m"),("1Y","pred_1y")]
                    if k in (xgb_pred or {})}
                   if xgb_pred else {}),
        # Legacy fields (backward-compat)
        "targets": {
            "1M": {"price": mc_result["1M"]["price_base"],
                   "return_pct": mc_result["1M"]["ret_base"]},
            "3M": {"price": mc_result["3M"]["price_base"],
                   "return_pct": mc_result["3M"]["ret_base"]},
            "1Y": {"price": mc_result["1Y"]["price_base"],
                   "return_pct": mc_result["1Y"]["ret_base"]},
        },
        "confidence_bands": {
            "3M_P90": {"price": mc_result["3M"]["price_p90"],
                       "return_pct": mc_result["3M"]["ret_p90"]},
            "3M_P95": {"price": mc_result["3M"]["price_p95"],
                       "return_pct": mc_result["3M"]["ret_p95"]},
            "3M_P99": {"price": mc_result["3M"]["price_p99"],
                       "return_pct": mc_result["3M"]["ret_p99"]},
        },
        "signals": {
            "rsi":       f.get("rsi"),
            "mom_3m":    f.get("mom_3m"),
            "macd_hist": f.get("macd_hist"),
            "composite": score,
            "discovery": f.get("discovery_bonus"),
        },
        "status":            "ACTIVE",
        "validation_checks": [],
    }
    preds.append(entry)
    save_predictions(preds)
    return entry["id"]


def validate_predictions():
    """
    Validate all ACTIVE predictions against current prices.
    Returns summary dict.
    """
    preds   = load_predictions()
    updated = False
    summary = {
        "total": 0, "active": 0, "on_track": 0,
        "outperforming": 0, "underperforming": 0,
        "stopped": 0, "missed": 0,
    }

    for pred in preds:
        if pred.get("status") != "ACTIVE":
            continue
        summary["total"]  += 1
        summary["active"] += 1

        try:
            sym         = pred["symbol"]
            entry_price = float(pred["entry_price"])
            curr_price  = yf.Ticker(sym).fast_info.last_price
            if not curr_price:
                continue

            actual_ret   = (curr_price / entry_price) - 1
            logged_dt    = datetime.datetime.fromisoformat(pred["logged_at"])
            days_elapsed = (datetime.datetime.now() - logged_dt).days
            stop         = float(pred["stop_loss"])

            if curr_price <= stop:
                pred["status"] = "STOPPED_OUT"
                summary["stopped"] += 1
                summary["active"]  -= 1
                updated = True
                continue

            checks = pred.setdefault("validation_checks", [])
            for horizon, days_needed in [("1M", 21), ("3M", 63), ("1Y", 252)]:
                if days_elapsed < days_needed:
                    continue
                if any(c.get("horizon") == horizon for c in checks):
                    continue

                target = pred["targets"][horizon]["return_pct"] / 100
                if   actual_ret >= target * 1.1:
                    result = "OUTPERFORMING";  summary["outperforming"] += 1
                elif actual_ret >= target * 0.6:
                    result = "ON_TRACK";       summary["on_track"] += 1
                elif actual_ret >= 0:
                    result = "UNDERPERFORMING"; summary["underperforming"] += 1
                else:
                    result = "MISSED";         summary["missed"] += 1

                checks.append({
                    "horizon":       horizon,
                    "checked_at":    datetime.datetime.now().isoformat(),
                    "actual_ret":    round(actual_ret * 100, 2),
                    "predicted_ret": pred["targets"][horizon]["return_pct"],
                    "result":        result,
                    "current_price": round(curr_price, 2),
                })
                updated = True

        except Exception:
            continue

    if updated:
        save_predictions(preds)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION MOVEMENT TRACKER  — actual price vs predicted at each horizon
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prediction_movements(sym, logged_at_str, entry_price):
    """
    For a logged prediction, compute actual price movement at 1D/1W/1M/3M/6M/1Y
    by fetching historical prices from the logged date onward.
    Returns dict {horizon_label: {"actual_price": x, "actual_pct": y} or None (future)}
    """
    try:
        logged_date = datetime.datetime.fromisoformat(logged_at_str).date()
        today       = datetime.date.today()
        days_since  = (today - logged_date).days

        if days_since < 1:
            return {}

        # Fetch enough data to cover 1 year
        period = "2y" if days_since > 300 else "1y" if days_since > 60 else "6mo"
        df = yf.Ticker(sym).history(period=period, interval="1d")
        if df.empty:
            return {}

        # Align index to dates (strip timezone safely)
        price_by_date = {}
        for ts, row in df.iterrows():
            try:
                d = ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()
            except Exception:
                continue
            price_by_date[d] = float(row["Close"])
        trading_days = sorted(price_by_date.keys())

        # Find the entry row (first trading day on/after logged_date)
        start_days = [d for d in trading_days if d >= logged_date]
        if not start_days:
            return {}
        start_date  = start_days[0]
        start_idx   = trading_days.index(start_date)

        HORIZONS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        results  = {}

        for label, n_trading_days in HORIZONS.items():
            target_idx = start_idx + n_trading_days
            if target_idx >= len(trading_days):
                results[label] = None   # not reached yet
                continue
            actual_date  = trading_days[target_idx]
            actual_price = price_by_date[actual_date]
            actual_pct   = round((actual_price / entry_price - 1) * 100, 2)
            results[label] = {
                "actual_price": round(actual_price, 2),
                "actual_pct":   actual_pct,
                "actual_date":  str(actual_date),
            }

        return results
    except Exception:
        return {}


def _next_n_trading_days(start_date, n):
    """Return the next n weekday dates after start_date (no holiday adjustment)."""
    days, d = [], start_date + datetime.timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:   # Mon–Fri
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_prices(sym, logged_at_str, n_days=10):
    """
    Return {day_index: {date_str, price_or_None}} for n_days trading days
    after logged_at_str.  Always returns an entry for every day (price=None
    if the market date hasn't been reached / data not yet available).
    """
    try:
        logged_date   = datetime.datetime.fromisoformat(logged_at_str).date()
        expected_days = _next_n_trading_days(logged_date, n_days)   # list of date

        # Fetch price history — strip timezone so .date() comparison is safe
        df = yf.Ticker(sym).history(period="3mo", interval="1d")
        price_by_date = {}
        if not df.empty:
            for ts, row in df.iterrows():
                try:
                    d = ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()
                except Exception:
                    continue
                price_by_date[d] = round(float(row["Close"]), 2)

        today  = datetime.date.today()
        result = {}
        for i, d in enumerate(expected_days, 1):
            result[i] = {
                "date":  d.strftime("%b %d"),
                "price": price_by_date.get(d) if d <= today else None,
            }
        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_watchlist():
    if not WATCHLIST_FILE.exists():
        return []
    try:
        return json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return []


def save_watchlist(wl):
    try:
        WATCHLIST_FILE.write_text(json.dumps(wl, indent=2, default=str))
    except Exception:
        pass


def build_watchlist_entry(sym, curr_sym):
    """
    Fetch features + fundamentals + Monte Carlo for a single stock
    and return a watchlist row dict.
    """
    f = fetch_stock_features(sym)
    if f is None:
        return None
    enrich_with_fundamentals(f)
    mc = monte_carlo_targets(f)

    def mp(horizon, key, fallback=0):
        if mc and horizon in mc:
            return mc[horizon].get(key, fallback)
        return fallback

    price = f["price"]
    result = {
        "symbol":    sym,
        "name":      f.get("name") or sym,
        "currency":  curr_sym,
        "added_at":  datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        "price":     price,
        "entry":     mc["entry"]     if mc else price,
        "stop_loss": mc["stop_loss"] if mc else round(price * 0.93, 2),
        "rr":        mc.get("rr_3m", 0) if mc else 0,
        "targets": {
            "1D": {"price": mp("1D","price_base",price), "ret": mp("1D","ret_base",0)},
            "1W": {"price": mp("1W","price_base",price), "ret": mp("1W","ret_base",0)},
            "1M": {"price": mp("1M","price_base",price), "ret": mp("1M","ret_base",0)},
            "3M": {"price": mp("3M","price_base",price), "ret": mp("3M","ret_base",0)},
            "6M": {"price": round(price * (1 + mp("3M","ret_base",0)/100 * 2), 2),
                   "ret":   round(mp("3M","ret_base",0) * 2, 1)},  # approx 6M from 3M
            "1Y": {"price": mp("1Y","price_base",price), "ret": mp("1Y","ret_base",0)},
        },
    }
    track_watchlist_entry_in_history(result)  # record in price history
    return result


def refresh_watchlist_prices(wl):
    """Update current price for every entry in the watchlist."""
    for entry in wl:
        try:
            fi = yf.Ticker(entry["symbol"]).fast_info
            p  = getattr(fi, "last_price", None)
            if p and p > 0:
                entry["price"] = round(float(p), 2)
        except Exception:
            pass
    return wl


# ══════════════════════════════════════════════════════════════════════════════
# PRICE HISTORY · VALIDATION · MODEL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def load_price_history():
    if PRICE_HISTORY_FILE.exists():
        try: return json.loads(PRICE_HISTORY_FILE.read_text())
        except Exception: return {}
    return {}

def save_price_history(ph):
    PRICE_HISTORY_FILE.write_text(json.dumps(ph, indent=2, default=str))

def load_validation_log():
    if VALIDATION_LOG_FILE.exists():
        try: return json.loads(VALIDATION_LOG_FILE.read_text())
        except Exception: return []
    return []

def save_validation_log(vl):
    VALIDATION_LOG_FILE.write_text(json.dumps(vl, indent=2, default=str))

def load_calibration():
    default = {
        "updated": None, "n_validations": 0,
        "mean_error": 0.0, "mean_abs_error": 0.0,
        "direction_accuracy": 0.5, "return_multiplier": 1.0,
        "score_correction": 0.0,
    }
    if CALIBRATION_FILE.exists():
        try:
            d = json.loads(CALIBRATION_FILE.read_text())
            return {**default, **d}
        except Exception: pass
    return default

def save_calibration(cal):
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2, default=str))

def track_watchlist_entry_in_history(entry: dict):
    """
    Called every time a stock is added to watchlist.
    Stores entry price + MC predictions + first daily snapshot.
    """
    ph  = load_price_history()
    sym = entry["symbol"]
    today = datetime.date.today().isoformat()
    ep  = entry.get("price", 0) or entry.get("entry", 0)
    tgts = entry.get("targets", {})
    preds = {
        h: {"target": tgts[h]["price"], "return_pct": tgts[h]["ret"],
            "prob_up": 60}  # default prob; will be enriched by monte_carlo_targets if available
        for h in tgts if h in ("1M","3M","6M","1Y")
    }
    if sym not in ph:
        ph[sym] = {
            "added": today,
            "entry_price": ep,
            "predictions": preds,
            "daily_prices": [],
        }
    # Always record today's price
    existing = {p["date"] for p in ph[sym].get("daily_prices", [])}
    if today not in existing:
        ph[sym].setdefault("daily_prices", []).append({"date": today, "price": ep})
    save_price_history(ph)

def sync_daily_prices(tickers: list) -> dict:
    """Fetch today's closing/live price for each ticker and append to price_history."""
    ph    = load_price_history()
    today = datetime.date.today().isoformat()
    done  = {}
    for sym in tickers:
        try:
            px = float(yf.Ticker(sym).fast_info.last_price or 0)
            if px <= 0:
                continue
            if sym not in ph:
                ph[sym] = {"added": today, "entry_price": px, "predictions": {}, "daily_prices": []}
            snaps = ph[sym].setdefault("daily_prices", [])
            # Update if already today, else append
            found = next((s for s in snaps if s["date"] == today), None)
            if found:
                found["price"] = px
            else:
                snaps.append({"date": today, "price": px})
            done[sym] = px
        except Exception:
            pass
    save_price_history(ph)
    return done

def check_due_predictions(ph: dict, vlog: list) -> list:
    """
    Return list of new validation entries for predictions whose horizon has elapsed.
    Uses the daily_prices snapshots to find the actual price at/after the due date.
    """
    import datetime as _dt
    horizon_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    today = _dt.date.today()
    new_entries = []
    already_ids = {v["id"] for v in vlog}

    for sym, data in ph.items():
        added_str = data.get("added")
        if not added_str:
            continue
        try:
            added_dt = _dt.date.fromisoformat(added_str[:10])
        except Exception:
            continue
        ep = float(data.get("entry_price", 0))
        if ep <= 0:
            continue
        for h, pred in data.get("predictions", {}).items():
            days = horizon_days.get(h)
            if not days:
                continue
            due = added_dt + _dt.timedelta(days=days)
            if today < due:
                continue
            vid = f"{sym}_{h}_{added_str[:10]}"
            if vid in already_ids:
                continue
            # Find actual price on or just after due date
            snaps = sorted(data.get("daily_prices", []), key=lambda x: x["date"])
            actual_snap = next((s for s in snaps if s["date"] >= due.isoformat()), None)
            if not actual_snap:
                continue
            predicted_price  = float(pred.get("target", 0) or pred.get("price", 0))
            predicted_return = float(pred.get("return_pct", 0))
            actual_price     = float(actual_snap["price"])
            actual_return    = (actual_price - ep) / ep * 100 if ep > 0 else 0
            model_error      = actual_return - predicted_return
            direction_ok     = (predicted_return > 0) == (actual_return > 0)
            new_entries.append({
                "id":                    vid,
                "ticker":                sym,
                "horizon":               h,
                "added":                 added_str[:10],
                "validation_date":       actual_snap["date"],
                "entry_price":           round(ep, 4),
                "predicted_price":       round(predicted_price, 4),
                "predicted_return_pct":  round(predicted_return, 2),
                "actual_price":          round(actual_price, 4),
                "actual_return_pct":     round(actual_return, 2),
                "model_error_pct":       round(model_error, 2),
                "deviation_pct":         round(abs(model_error), 2),
                "direction_correct":     direction_ok,
                "delta_explanation":     None,
                "delta_factors":         [],
            })
    return new_entries

def analyze_delta_with_gemini(ticker: str, predicted_ret: float,
                               actual_ret: float, entry_date: str,
                               validation_date: str) -> tuple:
    """Call Gemini to explain why the price deviated. Returns (explanation_str, factors_list)."""
    try:
        from google import genai as _gai
        import os, re as _re, json as _js
        _client = _gai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        direction = "outperformed" if actual_ret > predicted_ret else "underperformed"
        delta = abs(actual_ret - predicted_ret)
        prompt = (
            f"Stock {ticker} was predicted to return {predicted_ret:+.1f}% "
            f"between {entry_date} and {validation_date}. "
            f"Actual return: {actual_ret:+.1f}%. It {direction} the model by {delta:.1f} pp.\n\n"
            "In 2-3 sentences, explain the most likely reasons for this delta. "
            "Consider: earnings surprise, macro events (rates/inflation), sector rotation, "
            "M&A, guidance change, or model limitations (high-beta/momentum stocks overshoot).\n"
            "Also list 2-3 factor tags.\n"
            'Respond in JSON only: {"explanation": "...", "factors": ["tag1","tag2"]}'
        )
        resp = _client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        m = _re.search(r'\{.*\}', resp.text or "", _re.DOTALL)
        if m:
            d = _js.loads(m.group())
            return d.get("explanation",""), d.get("factors",[])
    except Exception:
        pass
    return "Delta analysis unavailable.", []

def recalculate_calibration(vlog: list) -> dict:
    """Recompute model calibration stats from all validation entries."""
    errors     = [v["model_error_pct"]    for v in vlog if "model_error_pct"   in v]
    directions = [v["direction_correct"]  for v in vlog if "direction_correct"  in v]
    if not errors:
        return load_calibration()
    mean_err = sum(errors) / len(errors)
    mae      = sum(abs(e) for e in errors) / len(errors)
    dir_acc  = sum(directions) / len(directions) if directions else 0.5
    # Multiplier: if model consistently under-predicts (mean_err > 0), scale up slightly
    multiplier = max(0.5, min(1.5, 1.0 + mean_err / 200))
    score_corr = round(max(-10, min(10, -mean_err / 10)), 1)
    cal = {
        "updated":           datetime.date.today().isoformat(),
        "n_validations":     len(errors),
        "mean_error":        round(mean_err, 2),
        "mean_abs_error":    round(mae, 2),
        "direction_accuracy": round(dir_acc, 3),
        "return_multiplier": round(multiplier, 3),
        "score_correction":  score_corr,
    }
    save_calibration(cal)
    return cal

def apply_calibration(raw_return_pct: float) -> float:
    """Scale a raw MC return prediction by the learned multiplier."""
    cal = load_calibration()
    return round(raw_return_pct * cal.get("return_multiplier", 1.0), 2)


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA HUNTER — Helper functions (Tab 5)
# ══════════════════════════════════════════════════════════════════════════════

def _load_paper_trades():
    if PAPER_TRADES_FILE.exists():
        try:
            return json.loads(PAPER_TRADES_FILE.read_text())
        except Exception:
            return []
    return []

def _save_paper_trades(trades):
    PAPER_TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str))

@st.cache_data(ttl=600)
def _get_earnings_info(sym):
    """Next earnings date + average historical post-earnings move magnitude."""
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="2y")
        if hist.empty:
            return {"next_earn": None, "days_to_earn": None, "avg_earn_move": 0, "n_samples": 0}

        # Calendar for next earnings
        next_earn = None
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date", [])
                dates  = ed_raw if isinstance(ed_raw, list) else [ed_raw]
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                dates = list(cal.loc["Earnings Date"])
            else:
                dates = []
            future = []
            for d in dates:
                try:
                    ts = pd.Timestamp(d)
                    if ts.tzinfo:
                        ts = ts.tz_localize(None)
                    if ts.date() > datetime.date.today():
                        future.append(ts)
                except Exception:
                    pass
            if future:
                next_earn = min(future)
        except Exception:
            pass

        # Historical post-earnings moves
        moves = []
        try:
            earn_dates_df = tk.get_earnings_dates(limit=8)
            if earn_dates_df is not None and not earn_dates_df.empty:
                hist_notz = hist.copy()
                if hist_notz.index.tzinfo:
                    hist_notz.index = hist_notz.index.tz_localize(None)
                for ed in earn_dates_df.index[:8]:
                    try:
                        ed_ts = pd.Timestamp(ed)
                        if ed_ts.tzinfo:
                            ed_ts = ed_ts.tz_localize(None)
                        idx = hist_notz.index.searchsorted(ed_ts)
                        if 0 < idx < len(hist_notz):
                            day_ret = hist_notz["Close"].iloc[idx] / hist_notz["Close"].iloc[idx - 1] - 1
                            moves.append(abs(float(day_ret)))
                    except Exception:
                        pass
        except Exception:
            pass

        avg_move = float(np.mean(moves)) if moves else 0
        days_to  = None
        if next_earn is not None:
            try:
                days_to = (next_earn.date() - datetime.date.today()).days
            except Exception:
                pass
        return {
            "next_earn": str(next_earn.date()) if next_earn else None,
            "days_to_earn": days_to,
            "avg_earn_move": avg_move,
            "n_samples": len(moves),
        }
    except Exception:
        return {"next_earn": None, "days_to_earn": None, "avg_earn_move": 0, "n_samples": 0}

@st.cache_data(ttl=600)
def _get_breakout_data(sym):
    """Return breakout score + key metrics for a symbol, or None on failure."""
    try:
        hist = yf.Ticker(sym).history(period="1y")
        if hist.empty or len(hist) < 50:
            return None
        close = hist["Close"].dropna()
        vol   = hist["Volume"].dropna()
        if len(close) < 20:
            return None

        curr     = float(close.iloc[-1])
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())

        # RSI-14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

        vol_20     = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
        vol_ratio  = float(vol.iloc[-1]) / vol_20 if vol_20 > 0 else 1.0
        ath_pct    = (curr / high_52w - 1) * 100
        mom_5d     = float(close.pct_change(5).iloc[-1]) * 100

        ma20        = float(close.rolling(20).mean().iloc[-1])
        ma50        = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
        bb_std      = float(close.rolling(20).std().iloc[-1])
        bb_upper    = ma20 + 2 * bb_std
        bb_breakout = curr > bb_upper

        score = 0
        if ath_pct > -3:        score += 30
        if vol_ratio > 2.0:     score += 25
        elif vol_ratio > 1.5:   score += 15
        if 55 <= rsi <= 75:     score += 20
        elif 45 <= rsi < 55:    score += 8
        if curr > ma20:         score += 10
        if curr > ma50:         score += 5
        if bb_breakout:         score += 10
        if mom_5d > 5:          score += 10
        elif mom_5d > 2:        score += 5

        return {
            "sym": sym, "curr": curr, "high_52w": high_52w, "ath_pct": ath_pct,
            "rsi": rsi, "vol_ratio": vol_ratio, "bb_breakout": bb_breakout,
            "mom_5d": mom_5d, "score": score,
            "above_ma20": curr > ma20, "above_ma50": curr > ma50,
        }
    except Exception:
        return None

@st.cache_data(ttl=600)
def _get_squeeze_data(sym):
    """Return short-squeeze score + key metrics, or None on failure."""
    try:
        tk   = yf.Ticker(sym)
        info = tk.info
        hist = tk.history(period="3mo")
        if hist.empty:
            return None
        close = hist["Close"].dropna()
        vol   = hist["Volume"].dropna()
        if len(close) < 10:
            return None

        curr        = float(close.iloc[-1])
        ma20        = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else curr
        mom_1m      = float(close.pct_change(20).iloc[-1]) * 100
        vol_5       = float(vol.tail(5).mean())
        vol_avg20   = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
        vol_ratio   = vol_5 / vol_avg20 if vol_avg20 > 0 else 1.0
        short_float = float(info.get("shortPercentOfFloat") or 0)
        short_ratio = float(info.get("shortRatio") or 0)       # days to cover

        score = 0
        if short_float > 0.25:   score += 35
        elif short_float > 0.15: score += 20
        elif short_float > 0.08: score += 10
        if short_ratio > 10:     score += 25
        elif short_ratio > 5:    score += 15
        if mom_1m > 10:          score += 20
        elif mom_1m > 5:         score += 10
        if vol_ratio > 1.5:      score += 10
        if curr > ma20:          score += 10

        return {
            "sym": sym, "curr": curr, "short_float": short_float,
            "short_ratio": short_ratio, "mom_1m": mom_1m,
            "vol_ratio": vol_ratio, "above_ma20": curr > ma20, "score": score,
        }
    except Exception:
        return None

def _scan_universe_parallel(syms, fn, n_workers=20, timeout=8):
    """Scan a list of symbols with fn() in parallel; return list of non-None results."""
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(fn, s): s for s in syms}
        for fut in as_completed(futs, timeout=timeout * len(syms)):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception:
                pass
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PORTED FROM app.py — Chart, Signals, Gemini AI, Fundamentals helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _get_ta_data(sym):
    """Fetch 2-year history + compute full TA indicator suite (ta library)."""
    df = _yf_history(sym, period="2y")
    if df.empty or not TA_AVAILABLE:
        return df
    try:
        df["RSI"]         = RSIIndicator(df["Close"]).rsi()
        stoch             = StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["Stoch_K"]     = stoch.stoch()
        df["Stoch_D"]     = stoch.stoch_signal()
        macd              = TA_MACD(df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MA50"]        = df["Close"].rolling(50).mean()
        df["MA200"]       = df["Close"].rolling(200).mean()
        df["EMA20"]       = EMAIndicator(df["Close"], window=20).ema_indicator()
        df["EMA50"]       = EMAIndicator(df["Close"], window=50).ema_indicator()
        adx               = ADXIndicator(df["High"], df["Low"], df["Close"])
        df["ADX"]         = adx.adx()
        df["DI_Plus"]     = adx.adx_pos()
        df["DI_Minus"]    = adx.adx_neg()
        bb                = BollingerBands(df["Close"])
        df["BB_High"]     = bb.bollinger_hband()
        df["BB_Low"]      = bb.bollinger_lband()
        df["BB_Mid"]      = bb.bollinger_mavg()
        df["OBV"]         = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    except Exception:
        pass
    return df

@st.cache_data(ttl=3600)
def _get_quarterly_data(sym):
    t = yf.Ticker(sym)
    try:
        qf = t.quarterly_income_stmt
    except Exception:
        qf = getattr(t, "quarterly_financials", None)
    if qf is None or qf.empty:
        return pd.DataFrame()
    return qf.T.sort_index()

@st.cache_data(ttl=900)
def _get_stock_news(sym):
    try:
        return yf.Ticker(sym).news or []
    except Exception:
        return []

def _app_fmt(v, prefix="", suffix="", scale=1, dec=2):
    if v is None:
        return "—"
    try:
        v = float(v) * scale
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    if abs(v) >= 1e12: return f"{prefix}{v/1e12:.{dec}f}T{suffix}"
    if abs(v) >= 1e9:  return f"{prefix}{v/1e9:.{dec}f}B{suffix}"
    if abs(v) >= 1e6:  return f"{prefix}{v/1e6:.{dec}f}M{suffix}"
    return f"{prefix}{v:.{dec}f}{suffix}"

def _app_safe(v):
    if v is None: return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def _app_parse_news_item(item):
    content   = item.get("content", {}) or {}
    title     = content.get("title") or item.get("title", "")
    canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    link      = canonical.get("url") if isinstance(canonical, dict) else ""
    link      = link or item.get("link") or item.get("url") or "#"
    provider  = content.get("provider") or {}
    pub       = (provider.get("displayName") if isinstance(provider, dict) else "") \
                or item.get("publisher", "")
    pub_date  = content.get("pubDate") or content.get("displayTime") or ""
    if pub_date:
        try:
            ts = datetime.datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            age_str = ts.strftime("%d %b %Y, %H:%M")
        except Exception:
            age_str = str(pub_date)[:16]
    else:
        ts_int  = item.get("providerPublishTime")
        age_str = datetime.datetime.fromtimestamp(ts_int).strftime("%d %b %Y, %H:%M") \
                  if isinstance(ts_int, int) else ""
    summary = content.get("summary") or item.get("summary", "")
    return title, link, pub, age_str, summary

def _app_compute_atr(df_price, period=14):
    h = df_price["High"]; l = df_price["Low"]; c = df_price["Close"].shift()
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def _app_signal_to_plain(signal_text):
    LOOKUP = {
        "golden cross":            "Long-term trend is up — price rising consistently for months",
        "death cross":             "Long-term trend is down — price falling for months",
        "oversold + %k crossing":  "Price near a bottom with early recovery signs",
        "overbought + %k crossing":"Price near a peak and starting to turn down",
        "oversold":                "Price has fallen sharply — may bounce back",
        "overbought":              "Price has risen too fast — may pull back soon",
        "bullish above zero":      "Momentum positive and getting stronger",
        "bearish below zero":      "Momentum negative and getting weaker",
        "bullish crossover":       "Momentum just turned positive — early buy signal",
        "bearish crossover":       "Momentum just turned negative — early sell signal",
        "rising obv":              "More money flowing in than leaving — buyers in control",
        "falling obv":             "More money leaving than entering — sellers in control",
        "strong uptrend":          "Price in a clear confirmed upward trend",
        "strong downtrend":        "Price in a clear confirmed downward trend",
        "near lower band":         "Price at historically low level — potential bounce zone",
        "near upper band":         "Price at historically high level — risk of pullback",
        "above mid-band":          "Price above 20-day average — mild upward bias",
        "below mid-band":          "Price below 20-day average — mild downward bias",
        "ema20 above ema50":       "Short-term trend stronger than medium-term — bullish",
        "ema20 below ema50":       "Short-term trend weaker than medium-term — bearish",
        "mildly oversold":         "Price showing some weakness, possible upside",
        "mildly overbought":       "Price a bit stretched, watch for slowdown",
        "neutral":                 "No strong signal in either direction",
        "weak / ranging":          "No clear trend — market moving sideways",
        "moderate trend":          "A trend developing but not yet confirmed",
        "flat obv":                "No strong conviction from volume — wait for clarity",
    }
    lower = signal_text.lower()
    for key, val in LOOKUP.items():
        if key in lower:
            return val
    return signal_text

def _app_build_signals(df):
    if df.empty:
        return []
    row   = df.dropna(subset=["Close"]).iloc[-1]
    close = row["Close"]
    signals = []

    def add(indicator, value, signal_text, action, score):
        signals.append({"Indicator": indicator, "Value": value,
                        "Signal": signal_text, "Plain": _app_signal_to_plain(signal_text),
                        "Action": action, "Score": score})

    v = row.get("RSI")
    if pd.notna(v):
        if   v < 30: add("RSI", f"{v:.1f}", "Oversold",          "BUY",       1.0)
        elif v > 70: add("RSI", f"{v:.1f}", "Overbought",        "SELL",     -1.0)
        elif v < 45: add("RSI", f"{v:.1f}", "Mildly Oversold",   "WEAK BUY",  0.5)
        elif v > 55: add("RSI", f"{v:.1f}", "Mildly Overbought", "WEAK SELL",-0.5)
        else:        add("RSI", f"{v:.1f}", "Neutral",           "NEUTRAL",   0.0)

    k, d = row.get("Stoch_K"), row.get("Stoch_D")
    if pd.notna(k) and pd.notna(d):
        if k < 20 and k > d:  add("Stochastic", f"%K={k:.1f} %D={d:.1f}", "Oversold + %K crossing up",   "BUY",  1.0)
        elif k > 80 and k < d:add("Stochastic", f"%K={k:.1f} %D={d:.1f}", "Overbought + %K crossing down","SELL",-1.0)
        elif k < 20:           add("Stochastic", f"%K={k:.1f} %D={d:.1f}", "Oversold",                    "WEAK BUY",0.5)
        elif k > 80:           add("Stochastic", f"%K={k:.1f} %D={d:.1f}", "Overbought",                  "WEAK SELL",-0.5)

    mac, sig = row.get("MACD"), row.get("MACD_Signal")
    if pd.notna(mac) and pd.notna(sig):
        prev = df.dropna(subset=["MACD","MACD_Signal"]).iloc[-2] if len(df) >= 2 else row
        cross_up   = mac > sig and (prev.get("MACD",0) or 0) <= (prev.get("MACD_Signal",0) or 0)
        cross_down = mac < sig and (prev.get("MACD",0) or 0) >= (prev.get("MACD_Signal",0) or 0)
        if   cross_up:   add("MACD", f"{mac:.3f}", "Bullish crossover",  "BUY",       1.5)
        elif cross_down: add("MACD", f"{mac:.3f}", "Bearish crossover",  "SELL",     -1.5)
        elif mac > 0:    add("MACD", f"{mac:.3f}", "Bullish above zero", "WEAK BUY",  0.5)
        else:            add("MACD", f"{mac:.3f}", "Bearish below zero", "WEAK SELL",-0.5)

    # MA50/MA200 — long-term indicator, minimal weight for short-term (≤1 month) traders
    ma50, ma200 = row.get("MA50"), row.get("MA200")
    if pd.notna(ma50) and pd.notna(ma200):
        prev2 = df.dropna(subset=["MA50","MA200"]).iloc[-2] if len(df) >= 2 else row
        if ma50 > ma200 and (prev2.get("MA50",0) or 0) <= (prev2.get("MA200",0) or 0):
            add("MA50/MA200", f"MA50={ma50:.2f}", "Golden Cross (long-term bullish)", "WEAK BUY",  0.2)
        elif ma50 < ma200 and (prev2.get("MA50",0) or 0) >= (prev2.get("MA200",0) or 0):
            add("MA50/MA200", f"MA50={ma50:.2f}", "Death Cross (long-term bearish)",  "WEAK SELL",-0.2)
        elif ma50 > ma200:
            add("MA50/MA200", f"MA50={ma50:.2f}", "Above MA200 (long-term context)", "NEUTRAL",  0.0)
        else:
            add("MA50/MA200", f"MA50={ma50:.2f}", "Below MA200 (long-term context)", "NEUTRAL",  0.0)

    e20, e50 = row.get("EMA20"), row.get("EMA50")
    if pd.notna(e20) and pd.notna(e50):
        if   e20 > e50: add("EMA20/50", f"EMA20={e20:.2f}", "EMA20 above EMA50", "WEAK BUY",  0.5)
        else:           add("EMA20/50", f"EMA20={e20:.2f}", "EMA20 below EMA50", "WEAK SELL",-0.5)

    adx_v = row.get("ADX"); dip = row.get("DI_Plus"); dim = row.get("DI_Minus")
    if pd.notna(adx_v):
        if   adx_v > 25 and pd.notna(dip) and pd.notna(dim) and dip > dim:
            add("ADX", f"{adx_v:.1f}", "Strong uptrend",     "BUY",       1.0)
        elif adx_v > 25 and pd.notna(dip) and pd.notna(dim) and dim > dip:
            add("ADX", f"{adx_v:.1f}", "Strong downtrend",   "SELL",     -1.0)
        elif adx_v > 20:
            add("ADX", f"{adx_v:.1f}", "Moderate trend",     "NEUTRAL",   0.0)
        else:
            add("ADX", f"{adx_v:.1f}", "Weak / ranging market","NEUTRAL",  0.0)

    bbh, bbl, bbm = row.get("BB_High"), row.get("BB_Low"), row.get("BB_Mid")
    if pd.notna(bbh) and pd.notna(bbl) and pd.notna(bbm) and bbh > bbl:
        pct = (close - bbl) / (bbh - bbl)
        if   pct < 0.15: add("Bollinger", f"{close:.2f}", "Near lower band", "WEAK BUY",  0.5)
        elif pct > 0.85: add("Bollinger", f"{close:.2f}", "Near upper band", "WEAK SELL",-0.5)
        elif pct > 0.5:  add("Bollinger", f"{close:.2f}", "Above mid-band",  "WEAK BUY",  0.3)
        else:            add("Bollinger", f"{close:.2f}", "Below mid-band",  "WEAK SELL",-0.3)

    obv_s = df.get("OBV")
    if obv_s is not None and len(obv_s.dropna()) >= 5:
        recent_obv = obv_s.dropna().iloc[-5:]
        slope = float(np.polyfit(range(len(recent_obv)), recent_obv.values, 1)[0])
        if   slope > 0:  add("OBV", f"{obv_s.dropna().iloc[-1]:,.0f}", "Rising OBV — buyers in control",  "WEAK BUY",  0.5)
        elif slope < 0:  add("OBV", f"{obv_s.dropna().iloc[-1]:,.0f}", "Falling OBV — sellers in control","WEAK SELL",-0.5)
        else:            add("OBV", f"{obv_s.dropna().iloc[-1]:,.0f}", "Flat OBV",    "NEUTRAL",   0.0)

    # Short-term momentum (5d and 10d) — high weight for ≤1 month traders
    closes = df["Close"].dropna()
    if len(closes) >= 10:
        _m5  = float((closes.iloc[-1]-closes.iloc[-5])/closes.iloc[-5]*100)
        _m10 = float((closes.iloc[-1]-closes.iloc[-10])/closes.iloc[-10]*100)
        if   _m5 > 5  and _m10 > 8:  add("Momentum (5d/10d)", f"+{_m5:.1f}%/{_m10:+.1f}%", "Strong short-term uptrend",   "BUY",       1.5)
        elif _m5 > 2  and _m10 > 3:  add("Momentum (5d/10d)", f"+{_m5:.1f}%/{_m10:+.1f}%", "Positive short-term momentum","WEAK BUY",  0.8)
        elif _m5 < -5 and _m10 < -8: add("Momentum (5d/10d)", f"{_m5:.1f}%/{_m10:+.1f}%",  "Strong short-term downtrend", "SELL",     -1.5)
        elif _m5 < -2 and _m10 < -3: add("Momentum (5d/10d)", f"{_m5:.1f}%/{_m10:+.1f}%",  "Negative short-term momentum","WEAK SELL",-0.8)
        else:                          add("Momentum (5d/10d)", f"{_m5:+.1f}%/{_m10:+.1f}%", "Sideways — no clear momentum","NEUTRAL",   0.0)

    return signals

_APP_COLORS = {
    "STRONG BUY":    ("#00ff7f", "#003a1f"),
    "BUY":           ("#00c853", "#002010"),
    "HOLD / NEUTRAL":("#ffcc00", "#332900"),
    "SELL":          ("#ff5252", "#2a0000"),
    "STRONG SELL":   ("#d50000", "#1a0000"),
}
_APP_ACTION_STYLE = {
    "BUY":       ("🟢", "#003a1f", "#00c853"),
    "WEAK BUY":  ("🟡", "#1a2600", "#aacc00"),
    "NEUTRAL":   ("⚪", "#222",    "#888888"),
    "WEAK SELL": ("🟠", "#2a1500", "#ff9800"),
    "SELL":      ("🔴", "#2a0000", "#ff5252"),
}
_APP_VERDICT_COLORS = {
    "strong buy":   ("#00ff7f", "#003a1f"),
    "buy":          ("#00c853", "#002010"),
    "watch":        ("#ffcc00", "#332900"),
    "avoid":        ("#ff5252", "#2a0000"),
    "strong avoid": ("#d50000", "#1a0000"),
}

def _app_score_signals(signals):
    if not signals: return 0, 0
    total = sum(s["Score"] for s in signals)
    maxsc = sum(abs(s["Score"]) for s in signals)
    return total, (total / maxsc * 100) if maxsc else 0

def _app_overall_label(pct):
    if   pct >= 60:  return "STRONG BUY",    "success"
    elif pct >= 20:  return "BUY",            "success"
    elif pct >= -20: return "HOLD / NEUTRAL", "warning"
    elif pct >= -60: return "SELL",           "error"
    else:            return "STRONG SELL",    "error"

def _app_color_action(val):
    clean = str(val).split(" ", 1)[-1].strip()
    _, bg, tx = _APP_ACTION_STYLE.get(clean, ("⚪", "#1a1a1a", "#888"))
    return f"background-color:{bg};color:{tx};font-weight:bold"

def _app_color_weight(val):
    try:
        fv = float(val)
        if fv > 0: return "color:#00c853;font-weight:bold"
        if fv < 0: return "color:#ff5252;font-weight:bold"
        return "color:#888"
    except Exception:
        return ""

def _app_render_verdict_banner(lbl, sc, n):
    fg, bg = _APP_COLORS.get(lbl, ("#888", "#111"))
    st.markdown(
        f"""<div style="background:{bg};border:2px solid {fg};border-radius:10px;
            padding:14px 22px;display:flex;align-items:center;gap:20px;margin-bottom:10px">
            <span style="font-size:2rem;font-weight:900;color:{fg};letter-spacing:2px">{lbl}</span>
            <span style="color:#ccc;font-size:0.95rem">Score:
                <b style="color:{fg}">{sc:+.1f}%</b> &nbsp;|&nbsp; {n} indicators analysed
            </span>
        </div>
        <div style="background:#222;border-radius:6px;height:12px;width:100%;margin-bottom:12px">
            <div style="background:{fg};width:{int((sc+100)/2)}%;height:100%;border-radius:6px"></div>
        </div>""",
        unsafe_allow_html=True,
    )

def _app_compute_macro_context(info):
    bullets = []
    beta = _app_safe(info.get("beta"))
    if beta:
        if   beta > 1.5: bullets.append(f"High-beta stock (β={beta:.2f}): amplifies market moves by {beta:.1f}×")
        elif beta < 0.7: bullets.append(f"Low-beta stock (β={beta:.2f}): relatively defensive vs market swings")
    div_y = _app_safe(info.get("dividendYield"))
    if div_y and div_y > 0.02:
        bullets.append(f"Dividend yield {div_y*100:.1f}% — provides income floor even in flat markets")
    country = info.get("country") or ""
    if country in ("India", "IN"):
        bullets.append("Indian market: sensitive to RBI policy, USD/INR, FII flows, and crude oil prices")
    elif country in ("United States", "US", "USA"):
        bullets.append("US market: Fed rate decisions, USD strength, and tech-sector earnings cycles key risks")
    sf = _app_safe(info.get("shortPercentOfFloat"))
    if sf and sf > 0.10:
        bullets.append(f"Short interest {sf*100:.1f}% of float — elevated short conviction; squeeze potential if positive catalyst")
    return bullets

def _app_compute_comprehensive_score(info, news_items, price):
    sections = {}

    # Valuation
    sc_v, mx_v, bl_v = 0, 45, []
    pe  = _app_safe(info.get("trailingPE"));   peg = _app_safe(info.get("pegRatio"))
    pb  = _app_safe(info.get("priceToBook"));  ps  = _app_safe(info.get("priceToSalesTrailing12Months"))
    if pe:
        if   pe < 15:  sc_v += 15; bl_v.append(f"P/E {pe:.1f} — cheap vs historical averages")
        elif pe < 25:  sc_v += 8;  bl_v.append(f"P/E {pe:.1f} — fair value range")
        elif pe < 40:  sc_v -= 5;  bl_v.append(f"P/E {pe:.1f} — moderately expensive")
        else:          sc_v -= 15; bl_v.append(f"P/E {pe:.1f} — expensive; needs strong growth to justify")
    if peg:
        if   peg < 1:  sc_v += 15; bl_v.append(f"PEG {peg:.2f} — growth-adjusted valuation is attractive")
        elif peg < 2:  sc_v += 5;  bl_v.append(f"PEG {peg:.2f} — reasonable for growth")
        else:          sc_v -= 10; bl_v.append(f"PEG {peg:.2f} — expensive even accounting for growth")
    if pb:
        if pb < 1.5:   sc_v += 10; bl_v.append(f"P/B {pb:.2f} — trading near book value")
        elif pb > 10:  sc_v -= 5;  bl_v.append(f"P/B {pb:.2f} — large premium to book")
    sections["💰 Valuation"] = (sc_v, mx_v, bl_v)

    # Growth
    sc_g, mx_g, bl_g = 0, 40, []
    rg = _app_safe(info.get("revenueGrowth")); eg = _app_safe(info.get("earningsGrowth"))
    if rg:
        if   rg > 0.25: sc_g += 20; bl_g.append(f"Revenue growing {rg*100:.1f}% YoY — strong top-line")
        elif rg > 0.10: sc_g += 10; bl_g.append(f"Revenue growing {rg*100:.1f}% YoY — solid growth")
        elif rg > 0:    sc_g += 3;  bl_g.append(f"Revenue growing {rg*100:.1f}% — slow but positive")
        else:           sc_g -= 15; bl_g.append(f"Revenue declining {rg*100:.1f}% — fundamental headwind")
    if eg:
        if   eg > 0.30: sc_g += 20; bl_g.append(f"Earnings growing {eg*100:.1f}% — exceptional")
        elif eg > 0.10: sc_g += 10; bl_g.append(f"Earnings growing {eg*100:.1f}% — healthy")
        elif eg > 0:    sc_g += 3;  bl_g.append(f"Earnings growing {eg*100:.1f}% — modest")
        else:           sc_g -= 10; bl_g.append(f"Earnings declining {eg*100:.1f}%")
    sections["📈 Growth"] = (sc_g, mx_g, bl_g)

    # Profitability
    sc_p, mx_p, bl_p = 0, 30, []
    nm = _app_safe(info.get("profitMargins")); gm = _app_safe(info.get("grossMargins"))
    roe = _app_safe(info.get("returnOnEquity"))
    if nm:
        if   nm > 0.20: sc_p += 10; bl_p.append(f"Net margin {nm*100:.1f}% — high-quality earnings")
        elif nm > 0.05: sc_p += 5;  bl_p.append(f"Net margin {nm*100:.1f}% — adequate profitability")
        elif nm > 0:    sc_p += 1;  bl_p.append(f"Net margin {nm*100:.1f}% — thin but positive")
        else:           sc_p -= 10; bl_p.append(f"Negative net margin {nm*100:.1f}% — burning cash")
    if roe:
        if   roe > 0.20: sc_p += 10; bl_p.append(f"ROE {roe*100:.1f}% — excellent capital efficiency")
        elif roe > 0.10: sc_p += 5;  bl_p.append(f"ROE {roe*100:.1f}% — decent returns on equity")
        elif roe < 0:    sc_p -= 10; bl_p.append(f"Negative ROE {roe*100:.1f}% — destroying shareholder value")
    sections["💼 Profitability"] = (sc_p, mx_p, bl_p)

    # Balance sheet
    sc_b, mx_b, bl_b = 0, 20, []
    de = _app_safe(info.get("debtToEquity")); cr = _app_safe(info.get("currentRatio"))
    if de:
        if   de < 30:  sc_b += 10; bl_b.append(f"D/E {de:.1f} — very low debt, strong balance sheet")
        elif de < 100: sc_b += 5;  bl_b.append(f"D/E {de:.1f} — manageable debt levels")
        elif de < 200: sc_b -= 5;  bl_b.append(f"D/E {de:.1f} — elevated debt; watch rates sensitivity")
        else:          sc_b -= 10; bl_b.append(f"D/E {de:.1f} — high leverage; default risk in stress")
    if cr:
        if   cr > 2:   sc_b += 10; bl_b.append(f"Current ratio {cr:.2f} — strong liquidity")
        elif cr > 1:   sc_b += 4;  bl_b.append(f"Current ratio {cr:.2f} — adequate liquidity")
        else:          sc_b -= 8;  bl_b.append(f"Current ratio {cr:.2f} — liquidity tight; may need capital")
    sections["🏦 Balance Sheet"] = (sc_b, mx_b, bl_b)

    # Analyst consensus
    sc_a, mx_a, bl_a = 0, 20, []
    rec = (info.get("recommendationKey") or "").lower()
    tgt = _app_safe(info.get("targetMeanPrice"))
    n_ana = info.get("numberOfAnalystOpinions") or 0
    if rec in ("strong_buy", "buy"):
        sc_a += 15; bl_a.append(f"Analyst consensus: {rec.replace('_',' ').title()} ({n_ana} analysts)")
    elif rec == "hold":
        sc_a += 5;  bl_a.append(f"Analyst consensus: Hold ({n_ana} analysts)")
    elif rec in ("sell", "strong_sell", "underperform"):
        sc_a -= 10; bl_a.append(f"Analyst consensus: {rec.title()} ({n_ana} analysts) — professional bearish view")
    if tgt and price > 0:
        upside = (tgt - price) / price * 100
        if   upside > 20: sc_a += 5; bl_a.append(f"Analyst target {upside:+.1f}% upside from current price")
        elif upside < -10: sc_a -= 5; bl_a.append(f"Analyst target {upside:+.1f}% — below current price")
    sections["🔭 Analyst View"] = (sc_a, mx_a, bl_a)

    total = sum(s for s, _, _ in sections.values())
    maxT  = sum(m for _, m, _ in sections.values())
    return (total / maxT * 100) if maxT else 0, sections

@st.cache_data(ttl=900)
def _fetch_daily_price_summary(ticker_sym: str) -> str:
    """Fetch 2 months of daily OHLCV, return a compact text table + computed stats for AI context."""
    try:
        import yfinance as _yf3
        df = _yf3.Ticker(ticker_sym).history(period="2mo", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return "No daily price history available."
        df = df.dropna(subset=["Close"])
        closes = df["Close"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        vols   = df["Volume"].values
        dates  = [str(i.date()) if hasattr(i, "date") else str(i)[:10] for i in df.index]

        current   = float(closes[-1])
        two_m_ago = float(closes[0])
        pct_2m    = (current - two_m_ago) / two_m_ago * 100

        # 2-month high/low
        hi_2m = float(max(highs));  lo_2m = float(min(lows))
        # 20-day high/low
        hi_20 = float(max(highs[-20:]));  lo_20 = float(min(lows[-20:]))
        # Average volume
        avg_vol = float(sum(vols) / len(vols))
        last_vol = float(vols[-1])

        # Trend: count up vs down days last 20
        up_days   = sum(1 for i in range(1, min(20, len(closes))) if closes[-i] > closes[-i-1])
        down_days = 20 - up_days

        # Key support/resistance: swing lows/highs in window
        swings_hi, swings_lo = [], []
        for i in range(2, len(closes)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swings_hi.append(float(highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swings_lo.append(float(lows[i]))
        # Nearest resistance above and support below current price
        res_levels = sorted([h for h in swings_hi if h > current])[:3]
        sup_levels = sorted([l for l in swings_lo if l < current], reverse=True)[:3]

        # Recent 10-day moves (weekly pattern)
        recent_rows = []
        step = max(1, len(df) // 10)
        for i in range(max(0, len(df)-10), len(df)):
            pct = (closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0
            recent_rows.append(f"  {dates[i]}  C:{closes[i]:.2f}  {pct:+.1f}%  Vol:{vols[i]/1e6:.1f}M")

        lines = [
            f"=== 2-MONTH DAILY PRICE HISTORY: {ticker_sym} ===",
            f"Current price: {current:.2f}",
            f"2-month move: {pct_2m:+.1f}%  (from {two_m_ago:.2f} → {current:.2f})",
            f"2-month High: {hi_2m:.2f}  |  2-month Low: {lo_2m:.2f}",
            f"20-day High:  {hi_20:.2f}  |  20-day Low:  {lo_20:.2f}",
            f"Last 20 days: {up_days} up-days / {down_days} down-days",
            f"Last volume: {last_vol/1e6:.1f}M  vs  Avg volume: {avg_vol/1e6:.1f}M  ({last_vol/avg_vol:.1f}x)",
            f"Key resistance above: {', '.join(f'{r:.2f}' for r in res_levels) or 'none identified'}",
            f"Key support below:    {', '.join(f'{s:.2f}' for s in sup_levels) or 'none identified'}",
            "",
            "Recent 10 trading days (Date | Close | % change | Volume):",
        ] + recent_rows
        return "\n".join(lines)
    except Exception as e:
        return f"Price history fetch error: {e}"


@st.cache_data(ttl=3600)
def _get_gemini_analysis_full(ticker_sym, price_str, rsi_v, macd_v,
                               ma50_v, ma200_v, bbh_v, bbl_v,
                               h52, l52, bull_n, bear_n, comp_score, currency,
                               earnings_info="Unknown", price_history_summary=""):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "__NO_KEY__"
    MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=api_key)
        prompt = f"""You are a senior portfolio manager at a $5B long/short hedge fund. You must give an HONEST, UNBIASED assessment — your fund's capital depends on it. You have NO obligation to be bullish. AVOID is the correct answer when the data warrants it.

Stock data:
Ticker: {ticker_sym} | Price: {price_str} | RSI: {rsi_v} | MACD: {macd_v}
MA50: {ma50_v} | MA200: {ma200_v} | BB High: {bbh_v} | BB Low: {bbl_v}
52W High: {h52} | 52W Low: {l52}
Bullish technical signals: {bull_n} | Bearish technical signals: {bear_n}
Composite technical score: {comp_score} (negative = bearish, positive = bullish)
Next Earnings: {earnings_info}

{price_history_summary}

TRADE HORIZON: Maximum 30 days. This is a pure short-term swing trade. Ignore all long-term fundamentals — the only question is: can this stock make a profitable move in the next 1-30 days?

FRAMEWORK:
1. TREND (from the 2-month price history above): Is the stock in an uptrend, downtrend, or sideways? Where are the key support/resistance levels? Are recent days showing accumulation (rising volume on up days) or distribution (rising volume on down days)?
2. TIMING: RSI, MACD, EMA20. Are we near support (good entry) or near resistance (wait or avoid)? Is momentum building or fading?
3. CATALYST (optional): Earnings run-up, sector rotation, news-driven move in the next 2-4 weeks?

VERDICT RULES:
- Uptrend + near support + bullish momentum → BUY (enter now)
- Uptrend + overbought / near resistance → WATCH (wait for pullback to specific price)
- Downtrend or breakdown → AVOID
- Stop loss: anchored to the nearest real support level from the 2-month history, 3-7% below entry
- Targets: based on resistance levels from the 2-month history
- Max hold: 30 days. Exit at target or stop, whichever comes first.

CRITICAL — answer this directly: ENTER NOW or WAIT FOR PULLBACK TO [exact price]?
- ENTER NOW: entry price must be within 1% of current price
- WAIT: give one specific price level (single number) where you'd enter, based on a real support level from the price history. State what candle/volume confirmation triggers entry.
- AVOID: explain why the setup is broken and when it would become tradeable again

Begin your response with EXACTLY this JSON block — ALL values must be single numbers, NO ranges, NO text, NO currency symbols, ALL within 20% of current price:
===JSON_START===
{{"verdict":"[STRONG BUY|BUY|WATCH|AVOID|STRONG AVOID]","conviction":"[HIGH|MEDIUM|LOW]","entry":"[single number]","stop_loss":"[single number, 3-7% below entry]","target_1m":"[single number — 5-10 day target]","target_3m":"[single number — 30-day max target]","target_6m":"0","target_1y":"0","rr_ratio":"[X.X:1]"}}
===JSON_END===

Then write analysis with ONLY these sections (be direct, no padding):

**VERDICT** — One word + one sentence reason.
**PRICE TREND (Last 2 Months)** — Uptrend / Downtrend / Sideways. Key support at [price], resistance at [price]. Accumulation or distribution in recent 5 days?
**ENTRY DECISION** — ENTER NOW at [price] OR WAIT FOR PULLBACK TO [exact price]. For WAIT: what specific candle/volume signal confirms entry? For AVOID: what would need to change?
**TECHNICAL SETUP** — RSI [value]: overbought/neutral/oversold. MACD: bullish/bearish cross. Volume: above/below average. EMA20 vs price. One-line conclusion.
**EARNINGS** — Next: {earnings_info}. Impact on this trade: run-up play / risk to avoid / no impact.
**STOP LOSS** — [exact price] — anchored to [support level name]. Risk: [X]% from entry.
**5-10 DAY TARGET** — [price] (+[X]%). Based on [resistance level].
**30-DAY TARGET** — [price] (+[X]%). Based on [resistance/measured move].
**RISK/REWARD** — [X.X]:1. [acceptable ≥2:1 / unacceptable <2:1].
**EXIT PLAN** — Hit target → sell at [price]. Stop triggered → exit at [price], no averaging down.
**TOP 3 RISKS** — Bullet list, specific to this stock and timeframe.
**TRADE SUMMARY** — One sentence: action, entry, stop, target, days.

Use {currency} for all prices in the text sections."""
        last_err = None
        for model in MODELS:
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    last_err = f"Quota on {model}"
                    continue
                raise
        return f"__ERROR__All models quota-exhausted. Last: {last_err}"
    except Exception as e:
        return f"__ERROR__{e}"


@st.cache_data(ttl=3600)
def _get_groq_analysis(ticker_sym, price_str, rsi_v, macd_v,
                        ma50_v, ma200_v, bbh_v, bbl_v,
                        h52, l52, bull_n, bear_n, comp_score, currency,
                        earnings_info="Unknown", price_history_summary=""):
    """Groq (Llama 3.3 70B) fallback when Gemini quota is exhausted. Free tier, no card needed."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "__NO_GROQ_KEY__"
    try:
        from groq import Groq as _Groq
        client = _Groq(api_key=api_key)
        prompt = f"""You are a senior portfolio manager at a $5B long/short hedge fund. You must give an HONEST, UNBIASED assessment — your fund's capital depends on it. You have NO obligation to be bullish. AVOID is the correct answer when the data warrants it.

Stock data:
Ticker: {ticker_sym} | Price: {price_str} | RSI: {rsi_v} | MACD: {macd_v}
MA50: {ma50_v} | MA200: {ma200_v} | BB High: {bbh_v} | BB Low: {bbl_v}
52W High: {h52} | 52W Low: {l52}
Bullish technical signals: {bull_n} | Bearish technical signals: {bear_n}
Composite technical score: {comp_score} (negative = bearish, positive = bullish)
Next Earnings: {earnings_info}

{price_history_summary}

TRADE HORIZON: Maximum 30 days. This is a pure short-term swing trade. Ignore all long-term fundamentals — the only question is: can this stock make a profitable move in the next 1-30 days?

FRAMEWORK:
1. TREND (from the 2-month price history above): Is the stock in an uptrend, downtrend, or sideways? Where are the key support/resistance levels? Are recent days showing accumulation (rising volume on up days) or distribution (rising volume on down days)?
2. TIMING: RSI, MACD, EMA20. Are we near support (good entry) or near resistance (wait or avoid)? Is momentum building or fading?
3. CATALYST (optional): Earnings run-up, sector rotation, news-driven move in the next 2-4 weeks?

VERDICT RULES:
- Uptrend + near support + bullish momentum → BUY (enter now)
- Uptrend + overbought / near resistance → WATCH (wait for pullback to specific price)
- Downtrend or breakdown → AVOID
- Stop loss: anchored to the nearest real support level from the 2-month history, 3-7% below entry
- Targets: based on resistance levels from the 2-month history
- Max hold: 30 days. Exit at target or stop, whichever comes first.

CRITICAL — answer this directly: ENTER NOW or WAIT FOR PULLBACK TO [exact price]?
- ENTER NOW: entry price must be within 1% of current price
- WAIT: give one specific price level (single number) where you'd enter, based on a real support level from the price history. State what candle/volume confirmation triggers entry.
- AVOID: explain why the setup is broken and when it would become tradeable again

Begin your response with EXACTLY this JSON block — ALL values must be single numbers, NO ranges, NO text, NO currency symbols, ALL within 20% of current price:
===JSON_START===
{{"verdict":"[STRONG BUY|BUY|WATCH|AVOID|STRONG AVOID]","conviction":"[HIGH|MEDIUM|LOW]","entry":"[single number]","stop_loss":"[single number, 3-7% below entry]","target_1m":"[single number — 5-10 day target]","target_3m":"[single number — 30-day max target]","target_6m":"0","target_1y":"0","rr_ratio":"[X.X:1]"}}
===JSON_END===

Then write analysis with ONLY these sections (be direct, no padding):

**VERDICT** — One word + one sentence reason.
**PRICE TREND (Last 2 Months)** — Uptrend / Downtrend / Sideways. Key support at [price], resistance at [price]. Accumulation or distribution in recent 5 days?
**ENTRY DECISION** — ENTER NOW at [price] OR WAIT FOR PULLBACK TO [exact price]. For WAIT: what specific candle/volume signal confirms entry? For AVOID: what would need to change?
**TECHNICAL SETUP** — RSI [value]: overbought/neutral/oversold. MACD: bullish/bearish cross. Volume: above/below average. EMA20 vs price. One-line conclusion.
**EARNINGS** — Next: {earnings_info}. Impact on this trade: run-up play / risk to avoid / no impact.
**STOP LOSS** — [exact price] — anchored to [support level name]. Risk: [X]% from entry.
**5-10 DAY TARGET** — [price] (+[X]%). Based on [resistance level].
**30-DAY TARGET** — [price] (+[X]%). Based on [resistance/measured move].
**RISK/REWARD** — [X.X]:1. [acceptable ≥2:1 / unacceptable <2:1].
**EXIT PLAN** — Hit target → sell at [price]. Stop triggered → exit at [price], no averaging down.
**TOP 3 RISKS** — Bullet list, specific to this stock and timeframe.
**TRADE SUMMARY** — One sentence: action, entry, stop, target, days.

Use {currency} for all prices in the text sections."""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"__ERROR__{e}"


@st.cache_data(ttl=3600)
def _get_claude_analysis(ticker_sym, price_str, rsi_v, macd_v,
                          ma50_v, ma200_v, bbh_v, bbl_v,
                          h52, l52, bull_n, bear_n, comp_score, currency,
                          earnings_info="Unknown", price_history_summary=""):
    """Claude Sonnet — primary AI provider, highest quality analysis."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "__NO_CLAUDE_KEY__"
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a senior portfolio manager at a $5B long/short hedge fund. You must give an HONEST, UNBIASED assessment — your fund's capital depends on it. You have NO obligation to be bullish. AVOID is the correct answer when the data warrants it.

Stock data:
Ticker: {ticker_sym} | Price: {price_str} | RSI: {rsi_v} | MACD: {macd_v}
MA50: {ma50_v} | MA200: {ma200_v} | BB High: {bbh_v} | BB Low: {bbl_v}
52W High: {h52} | 52W Low: {l52}
Bullish technical signals: {bull_n} | Bearish technical signals: {bear_n}
Composite technical score: {comp_score} (negative = bearish, positive = bullish)
Next Earnings: {earnings_info}

{price_history_summary}

TRADE HORIZON: Maximum 30 days. This is a pure short-term swing trade. Ignore all long-term fundamentals — the only question is: can this stock make a profitable move in the next 1-30 days?

FRAMEWORK:
1. TREND (from the 2-month price history above): Is the stock in an uptrend, downtrend, or sideways? Where are the key support/resistance levels? Are recent days showing accumulation (rising volume on up days) or distribution (rising volume on down days)?
2. TIMING: RSI, MACD, EMA20. Are we near support (good entry) or near resistance (wait or avoid)? Is momentum building or fading?
3. CATALYST (optional): Earnings run-up, sector rotation, news-driven move in the next 2-4 weeks?

VERDICT RULES:
- Uptrend + near support + bullish momentum → BUY (enter now)
- Uptrend + overbought / near resistance → WATCH (wait for pullback to specific price)
- Downtrend or breakdown → AVOID
- Stop loss: anchored to the nearest real support level from the 2-month history, 3-7% below entry
- Targets: based on resistance levels from the 2-month history
- Max hold: 30 days. Exit at target or stop, whichever comes first.

CRITICAL — answer this directly: ENTER NOW or WAIT FOR PULLBACK TO [exact price]?
- ENTER NOW: entry price must be within 1% of current price
- WAIT: give one specific price level (single number) where you'd enter, based on a real support level from the price history. State what candle/volume confirmation triggers entry.
- AVOID: explain why the setup is broken and when it would become tradeable again

Begin your response with EXACTLY this JSON block — ALL values must be single numbers, NO ranges, NO text, NO currency symbols, ALL within 20% of current price:
===JSON_START===
{{"verdict":"[STRONG BUY|BUY|WATCH|AVOID|STRONG AVOID]","conviction":"[HIGH|MEDIUM|LOW]","entry":"[single number]","stop_loss":"[single number, 3-7% below entry]","target_1m":"[single number — 5-10 day target]","target_3m":"[single number — 30-day max target]","target_6m":"0","target_1y":"0","rr_ratio":"[X.X:1]"}}
===JSON_END===

Then write analysis with ONLY these sections (be direct, no padding):

**VERDICT** — One word + one sentence reason.
**PRICE TREND (Last 2 Months)** — Uptrend / Downtrend / Sideways. Key support at [price], resistance at [price]. Accumulation or distribution in recent 5 days?
**ENTRY DECISION** — ENTER NOW at [price] OR WAIT FOR PULLBACK TO [exact price]. For WAIT: what specific candle/volume signal confirms entry? For AVOID: what would need to change?
**TECHNICAL SETUP** — RSI [value]: overbought/neutral/oversold. MACD: bullish/bearish cross. Volume: above/below average. EMA20 vs price. One-line conclusion.
**EARNINGS** — Next: {earnings_info}. Impact on this trade: run-up play / risk to avoid / no impact.
**STOP LOSS** — [exact price] — anchored to [support level name]. Risk: [X]% from entry.
**5-10 DAY TARGET** — [price] (+[X]%). Based on [resistance level].
**30-DAY TARGET** — [price] (+[X]%). Based on [resistance/measured move].
**RISK/REWARD** — [X.X]:1. [acceptable ≥2:1 / unacceptable <2:1].
**EXIT PLAN** — Hit target → sell at [price]. Stop triggered → exit at [price], no averaging down.
**TOP 3 RISKS** — Bullet list, specific to this stock and timeframe.
**TRADE SUMMARY** — One sentence: action, entry, stop, target, days.

Use {currency} for all prices in the text sections."""

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        err = str(e)
        if "credit" in err.lower() or "billing" in err.lower() or "quota" in err.lower():
            return f"__ERROR__Claude credits exhausted. Top up at console.anthropic.com/billing. {err[:100]}"
        return f"__ERROR__{e}"


def _app_parse_gemini_response(raw):
    summary = None
    text = raw
    m = re.search(r"===JSON_START===\s*(.*?)\s*===JSON_END===", text, re.DOTALL)
    if m:
        try:
            summary = json.loads(m.group(1))
        except Exception:
            pass
        text = (text[:m.start()] + text[m.end():]).strip()
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)
    text = re.sub(r"(\d)\.([A-Z])", r"\1. \2", text)
    # Escape $ signs so Streamlit doesn't render them as LaTeX math
    text = re.sub(r'\$(?=[\d\s])', r'\\$', text)
    return summary, text


WS_LIVE_FILE    = Path("./ws_live.json")
WS_CONTROL_FILE = Path("./ws_control.json")

def _ws_subscribe(sym: str):
    """Tell the ws_feed process to subscribe to a symbol."""
    try:
        existing = []
        if WS_CONTROL_FILE.exists():
            existing = json.loads(WS_CONTROL_FILE.read_text()).get("symbols", [])
        if sym not in existing:
            existing.append(sym)
        WS_CONTROL_FILE.write_text(json.dumps({"symbols": existing}))
    except Exception:
        pass

def _ws_read(sym: str, max_age_s: int = 5) -> dict | None:
    """
    Read live WebSocket data for a symbol.
    Returns None if feed is not running or data is stale.
    """
    try:
        if not WS_LIVE_FILE.exists():
            return None
        raw  = json.loads(WS_LIVE_FILE.read_text())
        data = raw.get(sym)
        if not data:
            return None
        updated = datetime.datetime.fromisoformat(data["updated"])
        age_s   = (datetime.datetime.utcnow() - updated).total_seconds()
        if age_s > max_age_s:
            return None   # stale — feed may not be running
        return data
    except Exception:
        return None

def _ws_candles_to_df(candles: list) -> "pd.DataFrame | None":
    """Convert ws_feed candle list to a DataFrame compatible with the chart."""
    if not candles:
        return None
    import pandas as _pdw
    rows = []
    for c in candles:
        ts = _pdw.Timestamp(c["t"], unit="ms", tz="UTC")
        rows.append({"Open": c["o"], "High": c["h"], "Low": c["l"],
                     "Close": c["c"], "Volume": c["v"]})
    idx = [_pdw.Timestamp(c["t"], unit="ms", tz="UTC") for c in candles]
    return _pdw.DataFrame(rows, index=idx)

def _ws_is_running() -> bool:
    """Check if ws_feed is actively writing (file updated in last 10s)."""
    try:
        if not WS_LIVE_FILE.exists():
            return False
        mtime = WS_LIVE_FILE.stat().st_mtime
        return (time.time() - mtime) < 10
    except Exception:
        return False


@st.cache_data(ttl=900)
def _compute_market_risk_score() -> dict:
    """Lightweight market risk score for use across all tabs. Returns score 0-100 + level."""
    try:
        import yfinance as _yf3
        def _ql(sym):
            df = _yf3.download(sym, period="65d", progress=False, auto_adjust=True)
            if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
            if df is None or df.empty: return None, None, None
            close = df["Close"].dropna()
            last  = float(close.iloc[-1])
            chg20 = float((close.iloc[-1]-close.iloc[-20])/close.iloc[-20]*100) if len(close)>=20 else None
            chg60 = float((close.iloc[-1]-close.iloc[-60])/close.iloc[-60]*100) if len(close)>=60 else None
            return last, chg20, chg60
        _vix_l, _vix_c20, _ = _ql("^VIX")
        _t10_l, _, _        = _ql("^TNX")
        _t3m_l, _, _        = _ql("^IRX")
        _, _sp_c20, _sp_c60 = _ql("^GSPC")
        _, _hyg_c20, _      = _ql("HYG")
        _, _gold_c20, _     = _ql("GLD")
        pts, maxp = 0, 0
        def _add(p, m, cond): nonlocal pts, maxp; pts += p if cond else 0; maxp += m
        if _vix_l:
            _add(20, 20, _vix_l > 40); _add(12, 0, _vix_l > 30 and _vix_l <= 40); _add(5, 0, _vix_l > 20 and _vix_l <= 30)
        if _t10_l and _t3m_l:
            _add(18, 18, (_t10_l - _t3m_l/10) < 0)
        if _sp_c60:
            _add(15, 15, _sp_c60 < -5); _add(7, 0, -5 <= _sp_c60 < 0)
        if _hyg_c20:
            _add(15, 15, _hyg_c20 < -3); _add(6, 0, -3 <= _hyg_c20 < -1)
        if _gold_c20:
            _add(10, 10, _gold_c20 > 5); _add(4, 0, 2 < _gold_c20 <= 5)
        maxp  = max(maxp, 78)
        score = int(pts / maxp * 100)
        level = "EXTREME" if score>=70 else "HIGH" if score>=50 else "MODERATE" if score>=30 else "LOW"
        return {"score": score, "level": level, "vix": _vix_l, "sp_3m": _sp_c60, "hyg_20": _hyg_c20}
    except Exception:
        return {"score": 20, "level": "LOW", "vix": None, "sp_3m": None, "hyg_20": None}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔮 Stock Predictor Model")

main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "📈 Analysis & Prediction",
    "🔍 Find Stocks",
    "💼 My Portfolio",
    "🎯 Live Trade Signals",
    "🚨 Market Risk Monitor",
    "📓 Prediction Tracker",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Full 4-layer analysis (same engine as Tab 2 screener)
# ══════════════════════════════════════════════════════════════════════════════
with main_tab1:
    for _tab1_once in [True]:
        # ── Load price data (needed for stat chart in expander) ───────────────────
        with st.spinner(f"Loading {ticker}…"):
            df = load_price_data(ticker)

        if df.empty:
            st.error(f"No data for **{ticker}**. Check the ticker and try again.")
            break

        current_price = df["Close"].dropna().iloc[-1]

        # ── Market regime banner ──────────────────────────────────────────────────
        _mk = "IN" if market_label.startswith("🇮🇳") else "US"
        _rd = detect_market_regime(_mk)
        st.markdown(
            f"""<div style="background:{_rd['bg']};border:1px solid {_rd['color']};
            border-radius:8px;padding:10px 18px;margin-bottom:14px;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
              <div>
                <span style="color:{_rd['color']};font-weight:700;font-size:1rem">
                {_rd['emoji']} {_rd['regime']} MARKET REGIME</span>
                <span style="color:#aaa;font-size:0.83rem;margin-left:14px">{_rd['desc']}</span>
              </div>
              <div style="font-size:0.78rem;color:#888;text-align:right">
                VIX {_rd['vix']} &nbsp;·&nbsp; Index vs 200MA {_rd['vs_200ma']:+.1f}%
                &nbsp;·&nbsp; 3M mom {_rd['mom_3m']:+.1f}%
                &nbsp;·&nbsp; {'☀️ Golden cross' if _rd['golden'] else '💀 Death cross'}
              </div>
            </div>
            <div style="background:#111;border-radius:6px;padding:6px 14px;
            margin-bottom:16px;font-size:0.82rem;color:{_rd['color']}">
              📌 Regime action: {_rd['action']}
              &nbsp;&nbsp;<span style="color:#666">Kelly multiplier: {_rd['kelly_mult']:.0%} of full size</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Run full 4-layer analysis (cached in session_state per ticker) ────────
        cache_key = f"tab1_{ticker}"
        if st.session_state.get("_tab1_key") != cache_key:
            with st.spinner(f"Running full analysis for {ticker} — fetching technicals, "
                            f"fundamentals & running Monte Carlo…"):
                _f = fetch_stock_features(ticker)
                if _f:
                    enrich_with_fundamentals(_f)
                    _mc     = monte_carlo_targets(_f)
                    _ranked = normalise_universe({ticker: _f})
                    _bkt    = classify_bucket(_f, _ranked)
                    _score, _bkdn = score_stock(_f, _ranked, _bkt)
                else:
                    _f = _mc = _ranked = _bkdn = None
                    _score = 0; _bkt = "growth"

            st.session_state["_tab1_key"]   = cache_key
            st.session_state["_tab1_f"]     = _f
            st.session_state["_tab1_mc"]    = _mc
            st.session_state["_tab1_score"] = _score
            st.session_state["_tab1_bkdn"]  = _bkdn
            st.session_state["_tab1_bkt"]   = _bkt

        f     = st.session_state.get("_tab1_f")
        mc    = st.session_state.get("_tab1_mc")
        score = st.session_state.get("_tab1_score", 0)
        bkdn  = st.session_state.get("_tab1_bkdn", {})
        bkt   = st.session_state.get("_tab1_bkt", "growth")

        re_run = st.button("🔁 Re-run Analysis", key="tab1_rerun")
        if re_run:
            for k in ["_tab1_key","_tab1_f","_tab1_mc","_tab1_score","_tab1_bkdn","_tab1_bkt"]:
                st.session_state.pop(k, None)
            st.rerun()

        if f is None:
            st.error("Could not compute features for this ticker. Try again or select a different stock.")
            break

        # ── Stale-cache detection — auto-invalidate if price moved >3% ────────────
        try:
            live_price = _yf_fast_price(ticker)
            cached_price = f.get("price", 0)
            if cached_price > 0 and live_price > 0:
                move_pct = (live_price - cached_price) / cached_price * 100
                if abs(move_pct) >= 3.0:
                    move_dir = "risen" if move_pct > 0 else "fallen"
                    st.warning(
                        f"⚠️ **Price has {move_dir} {abs(move_pct):.1f}% since this analysis was last run** "
                        f"(analysis: ${cached_price:.2f} → live: ${live_price:.2f}). "
                        f"Click **🔁 Re-run Analysis** above to refresh the prediction.",
                        icon="🔄",
                    )
        except Exception:
            pass

        # ── Summary Strip — always visible above sub-tabs ─────────────────────────
        _mc3m_ret  = (mc["3M"].get("ret_base",      0)             if mc and "3M" in mc else 0)
        _mc3m_prob = (mc["3M"].get("prob_positive",  50)            if mc and "3M" in mc else 50)
        _mc3m_px   = (mc["3M"].get("price_base",     current_price) if mc and "3M" in mc else current_price)
        _mc1m_ret  = (mc["1M"].get("ret_base",       0)             if mc and "1M" in mc else 0)
        _mc1m_px   = (mc["1M"].get("price_base",     current_price) if mc and "1M" in mc else current_price)
        _mc1y_ret  = (mc["1Y"].get("ret_base",       0)             if mc and "1Y" in mc else 0)
        _mc1y_px   = (mc["1Y"].get("price_base",     current_price) if mc and "1Y" in mc else current_price)

        # Verdict: score AND Monte Carlo must agree
        _mc_bullish = _mc3m_prob >= 60 and _mc3m_ret > 0
        _mc_bearish = _mc3m_prob < 45 or _mc3m_ret < -5

        if score >= 75 and _mc_bullish:       _verdict_lbl = "🟢 STRONG BUY"
        elif score >= 65 and _mc_bullish:     _verdict_lbl = "🟢 BUY"
        elif score >= 65 and not _mc_bearish: _verdict_lbl = "🟡 WATCH"
        elif score >= 50 and _mc_bullish:     _verdict_lbl = "🟡 WATCH"
        elif _mc_bearish or score < 45:       _verdict_lbl = "🔴 AVOID"
        else:                                  _verdict_lbl = "⚪ HOLD"

        _name_display = (f.get("name") or ticker)[:40] if f else ticker

        # Row 1 — ticker header
        st.markdown(f"### {ticker} &nbsp; <span style='font-size:1rem;color:#888;font-weight:400'>{_name_display}</span>", unsafe_allow_html=True)

        # Row 2 — key metrics as native st.metric tiles
        _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
        with _m1:
            st.metric("Score", f"{score:.0f}/100",
                      delta=_verdict_lbl,
                      delta_color="normal" if "BUY" in _verdict_lbl else
                                  "off"    if "HOLD" in _verdict_lbl or "WATCH" in _verdict_lbl else "inverse")
        with _m2:
            st.metric("Live Price", f"{curr}{current_price:.2f}")
        with _m3:
            st.metric("1M Target (MC)",  f"{curr}{_mc1m_px:.2f}",  delta=f"{_mc1m_ret:+.1f}%",
                      delta_color="normal" if _mc1m_ret >= 0 else "inverse")
        with _m4:
            st.metric("3M Target (MC)",  f"{curr}{_mc3m_px:.2f}",  delta=f"{_mc3m_ret:+.1f}%",
                      delta_color="normal" if _mc3m_ret >= 0 else "inverse")
        with _m5:
            st.metric("1Y Target (MC)",  f"{curr}{_mc1y_px:.2f}",  delta=f"{_mc1y_ret:+.1f}%",
                      delta_color="normal" if _mc1y_ret >= 0 else "inverse")
        with _m6:
            st.metric("Prob Up (3M)",    f"{_mc3m_prob:.0f}%",
                      delta_color="normal" if _mc3m_prob >= 60 else
                                  "off"    if _mc3m_prob >= 45 else "inverse")

        # Row 3 — action buttons
        _b1, _b2, _b3, _b4 = st.columns([1.2, 1.2, 1.2, 3])
        with _b1:
            _top_add = st.button("➕ Add to Watchlist", key="top_add_watch", use_container_width=True)
        with _b2:
            _top_ref = st.button("🔄 Refresh Watchlist", key="top_ref_watch", use_container_width=True)
        with _b3:
            _top_rerun = st.button("🔁 Re-run Analysis", key="top_rerun_btn", use_container_width=True)

        if _top_add:
            with st.spinner(f"Building MC targets for {ticker}…"):
                _top_entry = build_watchlist_entry(ticker, curr)
            if _top_entry:
                _top_wl = load_watchlist()
                _top_wl = [_e for _e in _top_wl if _e["symbol"] != ticker]
                _top_wl.append(_top_entry)
                save_watchlist(_top_wl)
                st.success(f"✅ {ticker} added to watchlist with price targets!")
            else:
                st.error(f"Could not fetch data for {ticker}.")
        if _top_ref:
            with st.spinner("Refreshing prices…"):
                save_watchlist(refresh_watchlist_prices(load_watchlist()))
            st.success("Watchlist prices updated.")
        if _top_rerun:
            for _k in ["_tab1_key","_tab1_f","_tab1_mc","_tab1_score","_tab1_bkdn","_tab1_bkt"]:
                st.session_state.pop(_k, None)
            st.rerun()

        st.divider()

        # ── Command Center — unified single-scroll layout ─────────────────────────
        # Kept as named objects so any code below that still references them works.
        t1_sub1 = t1_sub2 = t1_sub3 = t1_sub4 = None  # placeholders (not used as tabs anymore)

        # ── SUB-TAB 1: Chart & Technical Signals ──────────────────────────────────
        # ── SECTION 1: AI Fund Manager — headline decision ────────────────────────
        # (rendered first; data fetched below and injected here via session_state)

        _cc_ai_placeholder = st.empty()   # filled after AI runs below

        st.markdown("---")

        # ── SECTION 2: Chart + Technicals ────────────────────────────────────────
        st.markdown("#### 📊 Chart & Technical Signals")
        if True:  # scope block (replaces with t1_sub1:)
            # Period selector
            _PERIOD_MAP = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
            _t1_period  = st.select_slider(
                "Chart Period",
                options=list(_PERIOD_MAP.keys()),
                value="3M",
                key="t1_period_slider",
            )
            _chart_days = _PERIOD_MAP[_t1_period]

            # Load TA data
            with st.spinner("Loading chart data…"):
                _ta_df = _get_ta_data(ticker)

            if _ta_df.empty:
                st.error("No price data available for this ticker.")
            else:
                _ta_close   = _ta_df["Close"].dropna()
                _ta_curr    = float(_ta_close.iloc[-1])
                _ta_rsi     = float(_ta_df["RSI"].dropna().iloc[-1]) if "RSI" in _ta_df else 0
                _ta_end     = _ta_df.index[-1]
                _ta_start   = _ta_end - pd.Timedelta(days=_chart_days)
                _ta_df_view = _ta_df[_ta_df.index >= _ta_start]

                # Key metrics row
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("Current Price",  f"{curr}{_ta_curr:.2f}")
                _m2.metric("RSI (14)",        f"{_ta_rsi:.1f}",
                           delta="Overbought" if _ta_rsi > 70 else ("Oversold" if _ta_rsi < 30 else "Neutral"))
                _m3.metric("52W High",        f"{curr}{_ta_close.max():.2f}")
                _m4.metric("52W Low",         f"{curr}{_ta_close.min():.2f}")

                # 4-pane chart
                _cfig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True,
                    row_heights=[0.50, 0.15, 0.20, 0.15],
                    vertical_spacing=0.03,
                    subplot_titles=("Price & Indicators", "Volume", "MACD", "RSI"),
                )
                # Candlestick
                _cfig.add_trace(go.Candlestick(
                    x=_ta_df_view.index,
                    open=_ta_df_view["Open"], high=_ta_df_view["High"],
                    low=_ta_df_view["Low"],   close=_ta_df_view["Close"],
                    name="Price", increasing_line_color="#00c853", decreasing_line_color="#ff5252",
                ), row=1, col=1)
                if "MA50" in _ta_df_view:
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["MA50"],
                        name="MA50", line=dict(color="#ff9100", width=1.2)), row=1, col=1)
                if "MA200" in _ta_df_view:
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["MA200"],
                        name="MA200", line=dict(color="#2979ff", width=1.2)), row=1, col=1)
                if "BB_High" in _ta_df_view and "BB_Low" in _ta_df_view:
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["BB_High"],
                        name="BB High", line=dict(color="rgba(180,180,180,0.4)", dash="dash", width=1)), row=1, col=1)
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["BB_Low"],
                        name="BB Low", fill="tonexty", fillcolor="rgba(180,180,180,0.06)",
                        line=dict(color="rgba(180,180,180,0.4)", dash="dash", width=1)), row=1, col=1)
                # Volume
                _vol_colors = ["#00c853" if c >= o else "#ff5252"
                               for c, o in zip(_ta_df_view["Close"], _ta_df_view["Open"])]
                _cfig.add_trace(go.Bar(x=_ta_df_view.index, y=_ta_df_view["Volume"],
                    name="Volume", marker_color=_vol_colors, opacity=0.7), row=2, col=1)
                # MACD
                if "MACD" in _ta_df_view and "MACD_Signal" in _ta_df_view:
                    _macd_hist = _ta_df_view["MACD"] - _ta_df_view["MACD_Signal"]
                    _cfig.add_trace(go.Bar(x=_ta_df_view.index, y=_macd_hist,
                        name="MACD Hist",
                        marker_color=["#00c853" if v >= 0 else "#ff5252" for v in _macd_hist.fillna(0)],
                        opacity=0.7), row=3, col=1)
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["MACD"],
                        name="MACD", line=dict(color="#00bcd4", width=1.2)), row=3, col=1)
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["MACD_Signal"],
                        name="Signal", line=dict(color="#ff9100", width=1.2)), row=3, col=1)
                # RSI
                if "RSI" in _ta_df_view:
                    _cfig.add_trace(go.Scatter(x=_ta_df_view.index, y=_ta_df_view["RSI"],
                        name="RSI", line=dict(color="#ffcc00", width=1.5)), row=4, col=1)
                    _cfig.add_hline(y=70, line=dict(color="#ff5252", width=1, dash="dash"), row=4, col=1)
                    _cfig.add_hline(y=30, line=dict(color="#00c853", width=1, dash="dash"), row=4, col=1)
                    _cfig.add_hrect(y0=70, y1=100, fillcolor="#ff5252", opacity=0.05, row=4, col=1)
                    _cfig.add_hrect(y0=0,  y1=30,  fillcolor="#00c853", opacity=0.05, row=4, col=1)
                    _cfig.update_yaxes(range=[0, 100], row=4, col=1)

                _cfig.update_layout(
                    height=820, template="plotly_dark", showlegend=True,
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=40, b=20),
                    legend=dict(orientation="h", y=1.02, x=0),
                )
                st.plotly_chart(_cfig, use_container_width=True)

                # Technical signal breakdown
                st.markdown("---")
                st.markdown("#### 📋 Technical Signal Breakdown")
                _ta_signals = _app_build_signals(_ta_df)
                if _ta_signals:
                    _, _ta_score_pct = _app_score_signals(_ta_signals)
                    _ta_buy   = [s for s in _ta_signals if s["Score"] >  0]
                    _ta_sell  = [s for s in _ta_signals if s["Score"] <  0]
                    _ta_neut  = [s for s in _ta_signals if s["Score"] == 0]
                    _ta_label, _ = _app_overall_label(_ta_score_pct)

                    _app_render_verdict_banner(_ta_label, _ta_score_pct, len(_ta_signals))

                    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                    _sc1.metric("🟢 Bullish",   len(_ta_buy))
                    _sc2.metric("🔴 Bearish",   len(_ta_sell))
                    _sc3.metric("⚪ Neutral",   len(_ta_neut))
                    _sc4.metric("Composite",    f"{_ta_score_pct:+.1f}%")

                    _sig_rows = []
                    for _s in _ta_signals:
                        _emoji, _, _ = _APP_ACTION_STYLE.get(_s["Action"], ("⚪","#222","#888"))
                        _sig_rows.append({
                            "Indicator": _s["Indicator"],
                            "Value":     _s["Value"],
                            "Signal":    _s["Signal"],
                            "What it means": _s["Plain"],
                            "Action":    f"{_emoji} {_s['Action']}",
                            "Weight":    f"{_s['Score']:+.1f}",
                        })
                    _sig_df = pd.DataFrame(_sig_rows)
                    st.dataframe(
                        _sig_df.style
                            .map(_app_color_action, subset=["Action"])
                            .map(_app_color_weight, subset=["Weight"]),
                        use_container_width=True, hide_index=True,
                    )

                    # Entry / stop guidance
                    st.markdown("---")
                    st.markdown("#### 🎯 Entry & Stop Guidance")
                    _ta_atr = _app_compute_atr(_ta_df)
                    _ta_last = _ta_df.dropna(subset=["Close"]).iloc[-1]
                    _ma50n = _app_safe(_ta_last.get("MA50"))
                    _bbln  = _app_safe(_ta_last.get("BB_Low"))
                    _bbmn  = _app_safe(_ta_last.get("BB_Mid"))

                    # ── Gather checklist data ─────────────────────────────────────
                    _closes = _ta_df["Close"].dropna()
                    try:
                        _mom_5d  = float((_closes.iloc[-1]-_closes.iloc[max(0,len(_closes)-5)])/_closes.iloc[max(0,len(_closes)-5)]*100)
                        _mom_20d = float((_closes.iloc[-1]-_closes.iloc[max(0,len(_closes)-20)])/_closes.iloc[max(0,len(_closes)-20)]*100)
                    except Exception:
                        _mom_5d = _mom_20d = 0

                    _52w_hi  = float(f.get("high_52w") or _closes.max())
                    _52w_lo  = float(f.get("low_52w")  or _closes.min())
                    _52w_pos = int((_ta_curr-_52w_lo)/(_52w_hi-_52w_lo)*100) if _52w_hi > _52w_lo else 50

                    _short_pct = float(f.get("short_percent") or f.get("shortPercentOfFloat") or 0)
                    if _short_pct < 1: _short_pct *= 100   # convert 0.xx → %

                    _vol_now = float(_ta_df["Volume"].iloc[-1]) if "Volume" in _ta_df else 0
                    _vol_avg = float(_ta_df["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in _ta_df else 1
                    _vol_ratio = _vol_now / (_vol_avg or 1)

                    # Earnings proximity
                    _days_to_earn = 999
                    try:
                        import yfinance as _yft
                        _earn_cal = _yft.Ticker(ticker).calendar
                        if _earn_cal is not None and not _earn_cal.empty:
                            _earn_dates = [v for v in _earn_cal.values.flatten() if hasattr(v, 'date')]
                            if _earn_dates:
                                _days_to_earn = max(0, (_earn_dates[0].date() - datetime.date.today()).days)
                    except Exception:
                        pass

                    # SPY relative strength
                    _spy_5d = 0
                    try:
                        _spy_df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
                        if not _spy_df.empty:
                            if hasattr(_spy_df.columns, "levels"):
                                _spy_df.columns = _spy_df.columns.get_level_values(0)
                            _spy_5d = float((_spy_df["Close"].iloc[-1]-_spy_df["Close"].iloc[0])/_spy_df["Close"].iloc[0]*100)
                    except Exception:
                        pass

                    # ── Build checklist ───────────────────────────────────────────
                    # Short-term focused checklist (max 1-month hold)
                    _macd_ok   = bool(_ta_last.get("MACD") and (_ta_last.get("MACD") or 0) > (_ta_last.get("MACD_Signal") or 0))
                    _ema_ok    = bool(_ta_last.get("EMA20") and _ta_last.get("EMA50") and
                                      (_ta_last.get("EMA20") or 0) > (_ta_last.get("EMA50") or 0))
                    _stoch_ok  = bool(_ta_last.get("Stoch_K") and (_ta_last.get("Stoch_K") or 100) < 80)

                    _chk = [
                        ("RSI not overbought (<70)",   _ta_rsi < 70,        f"RSI {_ta_rsi:.0f}" + (" — stretched, high chase risk" if _ta_rsi>=70 else " — ok")),
                        ("MACD bullish",               _macd_ok,            "MACD above signal line — momentum up" if _macd_ok else "MACD below signal — momentum down"),
                        ("Short-term trend (EMA20>50)", _ema_ok,            "EMA20 above EMA50 — short-term uptrend" if _ema_ok else "EMA20 below EMA50 — short-term downtrend"),
                        ("Stochastic not overbought",  _stoch_ok,           f"Stoch {_ta_last.get('Stoch_K',0):.0f}" + (" — overbought, possible pullback" if not _stoch_ok else " — ok")),
                        ("Volume confirmed",           _vol_ratio >= 1.2,   f"{_vol_ratio:.1f}× avg vol" + (" — conviction" if _vol_ratio>=1.2 else " — low volume, weak signal")),
                        ("Outperforming market",       _mom_5d > _spy_5d,   f"Stock {_mom_5d:+.1f}% vs SPY {_spy_5d:+.1f}% (5d)"),
                        ("Not chasing (10d move <15%)", _mom_5d < 15,       f"{_mom_5d:+.1f}% in 5 days" + (" — extended, consider waiting for pullback" if _mom_5d>=15 else "")),
                        ("Safe from earnings (>14d)",  _days_to_earn > 14,  f"{_days_to_earn}d to earnings" if _days_to_earn < 999 else "Earnings date unknown — verify manually"),
                        ("Room to run (52W pos 30–85%)", 30 <= _52w_pos <= 85,
                            f"At {_52w_pos}% of 52W range ({curr}{_52w_lo:.0f}→{curr}{_52w_hi:.0f})" +
                            (" — near 52W high, overhead resistance, minimal upside" if _52w_pos > 85 else
                             " — too beaten down, falling knife risk" if _52w_pos < 30 else
                             " — good room to run")),
                    ]
                    _pass = sum(1 for _, v, _ in _chk if v)
                    _total = len(_chk)

                    # ── Display checklist ─────────────────────────────────────────
                    st.markdown("**Pre-buy checklist:**")
                    for _cn, _cv, _cd in _chk:
                        _ico = "✅" if _cv else "❌"
                        st.markdown(f"{_ico} **{_cn}** — {_cd}")

                    st.markdown(f"**Score: {_pass}/{_total} checks passed**")
                    st.markdown("---")

                    # ── Final verdict ─────────────────────────────────────────────
                    _entry = round(_ta_curr, 2)
                    _stop  = round(max(_ta_curr - 1.5*_ta_atr, _ta_curr*0.94), 2)
                    _tgt1  = round(_entry + 2*(_entry-_stop), 2)
                    _tgt2  = round(_entry + 3*(_entry-_stop), 2)

                    if _pass >= 7:
                        # Good to enter
                        _sup = [_ta_curr]
                        if _ma50n and _ma50n <= _ta_curr: _sup.append(_ma50n)
                        if _bbmn  and _bbmn  <= _ta_curr: _sup.append(_bbmn)
                        _entry = round(min(_sup), 2)
                        _stop  = round(max(_entry - 1.5*_ta_atr, _entry*0.94), 2)
                        _tgt1  = round(_entry + 2*(_entry-_stop), 2)
                        _tgt2  = round(_entry + 3*(_entry-_stop), 2)
                        st.success(f"GOOD TO BUY — {_pass}/{_total} checks pass. Enter near {curr}{_entry:.2f} | Stop: {curr}{_stop:.2f} ({(_stop/_entry-1)*100:.1f}%) | Target 1: {curr}{_tgt1:.2f} | Target 2: {curr}{_tgt2:.2f}")
                    elif _pass >= 4:
                        st.warning(f"PROCEED WITH CAUTION — Only {_pass}/{_total} checks pass. Keep position small. Stop: {curr}{_stop:.2f} | Target: {curr}{_tgt1:.2f}")
                    else:
                        _wait = round(_ta_curr * 0.95, 2)
                        _wait_stop = round(_wait * 0.94, 2)
                        _wait_tgt  = round(_wait + 2*(_wait-_wait_stop), 2)
                        st.error(f"DO NOT BUY NOW — Only {_pass}/{_total} checks pass. Too many risks. Watch {curr}{_wait:.2f} for a safer entry. If entered there: Stop {curr}{_wait_stop:.2f} | Target {curr}{_wait_tgt:.2f}")

        st.markdown("---")
        # ── SECTION 3: Monte Carlo Price Prediction ───────────────────────────────
        st.markdown("#### 🔮 Monte Carlo Price Prediction")
        with st.expander("Show Price Prediction Detail", expanded=False):
          if True:  # scope block
            # ── Layout helpers (shared with Tab 2) ────────────────────────────────────
            PCLASS_COLORS = {
                "P99":   ("#ffd700", "#1a1400"),
                "P95":   ("#00c853", "#002010"),
                "P90":   ("#2979ff", "#001233"),
                "P80":   ("#888888", "#1a1a1a"),
                "WATCH": ("#ff9100", "#1a0d00"),
            }

            def t1_score_bar(val, color):
                try:
                    w = int(np.clip(float(val or 0), 0, 100))
                except Exception:
                    w = 0
                return (f'<div style="background:#222;border-radius:4px;height:8px;margin:2px 0">'
                        f'<div style="background:{color};width:{w}%;height:100%;border-radius:4px">'
                        f'</div></div>')

            def t1_mc(horizon, key, fallback=0):
                if mc and horizon in mc:
                    return mc[horizon].get(key, fallback)
                return fallback

            price      = f["price"]
            name_str   = f.get("name") or ticker
            sector     = f.get("sector") or "—"
            exchange   = f.get("exchange") or "—"
            disc_lbl   = f.get("discovery_label", "quality")
            disc_cfg   = DISCOVERY_LABELS.get(disc_lbl, DISCOVERY_LABELS["quality"])
            n_ana      = int(f.get("n_analysts") or 0)
            stop       = mc["stop_loss"] if mc else price * 0.93
            rr         = mc.get("rr_3m", 0) if mc else 0
            risk_p     = mc.get("risk_pct", 3.0) if mc else 3.0
            pclass     = assign_percentile(score, [score])   # single stock → P80
            pc_fg, pc_bg = PCLASS_COLORS.get(pclass, ("#888", "#111"))

            tgt_3m_p, tgt_3m_r = blend_predictions(mc, None, "3M")
            tgt_1m_p, tgt_1m_r = blend_predictions(mc, None, "1M")
            tgt_1y_p, tgt_1y_r = blend_predictions(mc, None, "1Y")

            bkt_color  = BUCKET_CONFIG.get(bkt, BUCKET_CONFIG["growth"])["color"]
            bkt_border = BUCKET_CONFIG.get(bkt, BUCKET_CONFIG["growth"])["border"]
            bkt_name   = BUCKET_CONFIG.get(bkt, BUCKET_CONFIG["growth"])["name"]

            # Probability stats from 3M MC
            prob_pos  = t1_mc("3M", "prob_positive", 50)
            prob_40   = t1_mc("3M", "prob_40pct", 0)
            prob_loss = t1_mc("3M", "prob_loss10", 0)

            # Signal pills
            pills = []
            if (f.get("mom_3m") or 0) > 0.10:   pills.append(("🟢 Momentum Strong",    "#002010", "#00c853"))
            if f.get("golden_cross") == 1:       pills.append(("🟢 Golden Cross",        "#002010", "#00c853"))
            rsi_v = float(f.get("rsi") or 50)
            if   rsi_v < 40:                     pills.append(("🟢 Oversold",            "#002010", "#00c853"))
            elif rsi_v > 65:                     pills.append(("🔴 Overbought",           "#2a0000", "#ff5252"))
            else:                                pills.append(("🟡 RSI Neutral",          "#332900", "#ffcc00"))
            if (f.get("inst_holding") or 0) > 0.50: pills.append(("🟢 Inst. Backing",   "#002010", "#00c853"))
            if (f.get("obv_slope") or 0) > 0.05:    pills.append(("🟢 OBV Rising",       "#002010", "#00c853"))
            if (f.get("macd_hist") or 0) > 0:       pills.append(("🟢 MACD Bullish",     "#002010", "#00c853"))
            else:                                    pills.append(("🔴 MACD Bearish",      "#2a0000", "#ff5252"))
            if f.get("above_ma200") == 1:        pills.append(("🟢 Above MA200",         "#002010", "#00c853"))
            else:                                pills.append(("🔴 Below MA200",          "#2a0000", "#ff5252"))
            if n_ana == 0:                       pills.append(("🔭 Zero Coverage",        "#1a0033", "#9c27b0"))
            elif n_ana <= 3:                     pills.append(("💎 Low Coverage",         "#001a1f", "#00bcd4"))

            pills_html = " ".join([
                f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
                f'border-radius:12px;padding:3px 12px;font-size:0.78rem;margin-right:4px">{lbl}</span>'
                for lbl, bg, fg in pills
            ])

            # Score breakdown bars
            breakdown_html = "".join([
                f'<div><div style="color:#888;font-size:0.72rem">{lbl}</div>'
                f'{t1_score_bar(val, clr)}'
                f'<div style="color:#aaa;font-size:0.72rem">{float(val or 0):.0f}/100</div></div>'
                for lbl, val, clr in [
                    ("Technical",   float(bkdn.get("technical",   50) or 50), "#2979ff"),
                    ("Fundamental", float(bkdn.get("fundamental", 50) or 50), "#00c853"),
                    ("Risk/Macro",  float(bkdn.get("risk",        50) or 50), "#ff9100"),
                    ("Sentiment",   float(bkdn.get("sentiment",   50) or 50), "#9c27b0"),
                ]
            ])

            # Horizon tiles
            horizons_html = (
                '<span style="color:#555;font-size:0.72rem;margin-right:6px">Median path (P50) →</span>'
                + " ".join([
                    f'<span style="background:#111;border:1px solid #222;border-radius:6px;'
                    f'padding:4px 10px;font-size:0.78rem">'
                    f'<span style="color:#888">{hl}:</span> '
                    f'<span style="color:#fff;font-weight:600">{curr}{t1_mc(hl,"price_base",price):.2f}</span> '
                    f'<span style="color:{"#00c853" if t1_mc(hl,"ret_base",0)>0 else "#ff5252"}">'
                    f'({t1_mc(hl,"ret_base",0):+.1f}%)</span></span>'
                    for hl in ["1D", "1W", "1M", "3M", "1Y"]
                ])
            )

            # ── Main analysis card ────────────────────────────────────────────────────
            st.markdown(f"""
        <div style="background:#0d0d0d;border:2px solid {bkt_border};
        border-radius:12px;padding:20px 24px;margin-bottom:16px">

          <!-- HEADER ROW -->
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap">
            <span style="background:{pc_bg};color:{pc_fg};border:1px solid {pc_fg};
            border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:700">{pclass}</span>
            <span style="color:{disc_cfg[1]};background:{disc_cfg[2]};border:1px solid {disc_cfg[1]};
            border-radius:6px;padding:3px 10px;font-size:0.78rem;font-weight:600">{disc_cfg[0]}</span>
            <span style="color:#fff;font-size:1.3rem;font-weight:800">{ticker}</span>
            <span style="color:#aaa;font-size:1rem">{name_str}</span>
            <span style="color:#666">·</span>
            <span style="color:#888;font-size:0.85rem">{sector}</span>
            <span style="color:#666">·</span>
            <span style="color:#666;font-size:0.85rem">{exchange}</span>
            <span style="margin-left:auto;font-size:0.9rem;color:#888">
              Bucket: <b style="color:{bkt_color}">{bkt_name}</b>
              &nbsp;·&nbsp; Score: <b style="color:{bkt_color}">{score:.1f}/100</b>
            </span>
          </div>

          <!-- PRICE / TARGET ROW -->
          <div style="display:flex;gap:32px;margin-bottom:14px;flex-wrap:wrap;align-items:flex-end">
            <div>
              <div style="color:#888;font-size:0.75rem">Current Price</div>
              <div style="color:#fff;font-size:1.5rem;font-weight:800">{curr}{price:.2f}</div>
            </div>
            <div style="color:#555;font-size:2rem;align-self:center">→</div>
            <div>
              <div style="color:#888;font-size:0.75rem">Base Target (3M · Monte Carlo P50)</div>
              <div style="color:{bkt_color};font-size:1.5rem;font-weight:800">
              {curr}{tgt_3m_p or 0:.2f}
              <span style="font-size:1rem">({tgt_3m_r or 0:+.1f}%)</span></div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">Entry</div>
              <div style="color:#fff;font-size:1.1rem;font-weight:700">{curr}{price:.2f}</div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">Stop Loss</div>
              <div style="color:#ff5252;font-size:1.1rem;font-weight:700">
              {curr}{stop:.2f}
              <span style="color:#888;font-size:0.85rem"> (-{risk_p:.1f}%)</span></div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">R/R Ratio</div>
              <div style="color:{"#00c853" if rr >= 2 else "#ff9100"};font-size:1.1rem;font-weight:700">
              {rr:.1f}:1 {"✅" if rr >= 2 else "⚠️"}</div>
            </div>
          </div>

          <!-- CONFIDENCE BANDS TABLE -->
          <div style="background:#111;border-radius:8px;padding:14px;margin-bottom:14px">
            <div style="color:#888;font-size:0.75rem;margin-bottom:8px;font-weight:600">
            CONFIDENCE BANDS (3M) · 10,000 Monte Carlo Paths</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
              <tr style="color:#666">
                <td style="padding:4px 8px">Scenario</td>
                <td style="padding:4px 8px">Price Target</td>
                <td style="padding:4px 8px">Return %</td>
                <td style="padding:4px 8px">Probability</td>
              </tr>
              <tr style="color:#ff5252">
                <td style="padding:4px 8px">Bear (P10)</td>
                <td style="padding:4px 8px">{curr}{t1_mc("3M","price_bear",price):.2f}</td>
                <td style="padding:4px 8px">{t1_mc("3M","ret_bear",0):+.1f}%</td>
                <td style="padding:4px 8px">10%</td>
              </tr>
              <tr style="color:#aaa">
                <td style="padding:4px 8px">Base (P50)</td>
                <td style="padding:4px 8px">{curr}{t1_mc("3M","price_base",price):.2f}</td>
                <td style="padding:4px 8px">{t1_mc("3M","ret_base",0):+.1f}%</td>
                <td style="padding:4px 8px">50%</td>
              </tr>
              <tr style="color:#2979ff">
                <td style="padding:4px 8px">P90 Target</td>
                <td style="padding:4px 8px;font-weight:700">{curr}{t1_mc("3M","price_p90",price):.2f}</td>
                <td style="padding:4px 8px;font-weight:700">{t1_mc("3M","ret_p90",0):+.1f}%</td>
                <td style="padding:4px 8px">{prob_pos:.1f}% positive</td>
              </tr>
              <tr style="color:#00c853">
                <td style="padding:4px 8px">P95 Target</td>
                <td style="padding:4px 8px;font-weight:700">{curr}{t1_mc("3M","price_p95",price):.2f}</td>
                <td style="padding:4px 8px;font-weight:700">{t1_mc("3M","ret_p95",0):+.1f}%</td>
                <td style="padding:4px 8px">{prob_40:.1f}% chance >40%</td>
              </tr>
              <tr style="color:#ffd700">
                <td style="padding:4px 8px">P99 Stretch</td>
                <td style="padding:4px 8px;font-weight:700">{curr}{t1_mc("3M","price_p99",price):.2f}</td>
                <td style="padding:4px 8px;font-weight:700">{t1_mc("3M","ret_p99",0):+.1f}%</td>
                <td style="padding:4px 8px;color:#666">⚠️ Loss>10%: {prob_loss:.1f}%</td>
              </tr>
            </table>
          </div>

          <!-- ALL HORIZONS -->
          <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;align-items:center">
            {horizons_html}
          </div>

          <!-- SIGNAL PILLS -->
          <div style="margin-bottom:14px">{pills_html}</div>

          <!-- SCORE BREAKDOWN BARS -->
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:14px">
            {breakdown_html}
          </div>

        </div>""", unsafe_allow_html=True)

            # ── Multi-model price prediction comparison ───────────────────────────────
            st.markdown("### 📐 Price Prediction Comparison — All Models")
            st.caption(
                "Same stock, same horizons, different models. "
                "When multiple models agree on direction — conviction is higher. "
                "When they disagree — the outcome is genuinely uncertain."
            )

            HORIZONS_MAP = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

            # Pre-compute DCF and Relative Valuation for both the table and analysis bullets
            t1_dcf = compute_dcf(f)
            _rv_verdict, _rv_signals, _rv_score_pct = compute_relative_valuation(f)

            # P/E-implied fair price (normalised to market median P/E = 22)
            _pe = _safe_float(f.get("pe"))
            _rv_fair_price = round(current_price * (22.0 / _pe), 2) if (_pe and _pe > 0 and current_price > 0) else None

            # Score-implied annual returns (centred at 50 → ±20% tech, ±30% fund)
            _tech_score  = float(bkdn.get("technical",   50) or 50)
            _fund_score  = float(bkdn.get("fundamental", 50) or 50)
            _tech_annual = (_tech_score  - 50) / 50 * 0.20
            _fund_annual = (_fund_score  - 50) / 50 * 0.30

            def _score_price(annual_ret, days):
                daily = annual_ret / 252
                pv    = max(1.0, current_price * (1 + daily * days))
                return round(pv, 2), round((pv / current_price - 1) * 100, 1)

            # XGBoost predictions (if available)
            xgb_models_t1 = st.session_state.get("xgb_models")
            xgb_pred_t1   = get_xgb_prediction(f, xgb_models_t1) if xgb_models_t1 else None

            # Jump Diffusion (Merton model) — pre-compute alongside regular MC
            jd_mc = monte_carlo_jump_diffusion(f, n_simulations=5000)

            # Build comparison table
            def _cell(price_val, pct_val, is_mc=False):
                clr  = "#00c853" if pct_val > 0 else ("#ff5252" if pct_val < 0 else "#888")
                bold = "font-weight:700;" if is_mc else ""
                return (
                    f"<td style='padding:6px 10px;text-align:right'>"
                    f"<span style='color:#fff;{bold}'>{curr}{price_val:.2f}</span> "
                    f"<span style='color:{clr};font-size:0.78rem'>({pct_val:+.1f}%)</span></td>"
                )

            def _dash():
                return "<td style='padding:6px 10px;color:#444;text-align:right'>—</td>"

            hl_headers = "".join(
                f"<th style='padding:6px 10px;color:#666;font-weight:500;text-align:right'>{hl}</th>"
                for hl in HORIZONS_MAP
            )

            table_rows = []
            TECH_VALID = {"1D", "1W", "1M", "3M"}
            FUND_VALID = {"3M", "6M", "1Y"}
            LONG_VALID = {"6M", "1Y"}
            XGB_KEY    = {"1M": "pred_1m", "3M": "pred_3m", "1Y": "pred_1y"}

            # Row 1: Monte Carlo P50 (all horizons)
            mc_row = "<td style='padding:6px 10px;color:#aaa;font-weight:700'>📊 Monte Carlo (P50 Median)</td>"
            for hl in HORIZONS_MAP:
                pv  = t1_mc(hl, "price_base", current_price)
                pct = (pv / current_price - 1) * 100 if current_price else 0
                mc_row += _cell(pv, pct, is_mc=True)
            table_rows.append(f"<tr style='background:#0d0d0d;border-bottom:1px solid #1a1a1a'>{mc_row}</tr>")

            # Row 2: Monte Carlo P90 (all horizons)
            mc90_row = "<td style='padding:6px 10px;color:#aaa'>📊 Monte Carlo (P90 Bull)</td>"
            for hl in HORIZONS_MAP:
                pv  = t1_mc(hl, "price_p90", current_price)
                pct = (pv / current_price - 1) * 100 if current_price else 0
                mc90_row += _cell(pv, pct)
            table_rows.append(f"<tr style='border-bottom:1px solid #1a1a1a'>{mc90_row}</tr>")

            # Row 3: Jump Diffusion MC (Merton model) — all horizons
            jd_row = "<td style='padding:6px 10px;color:#aaa'>⚡ Jump Diffusion MC (Merton)</td>"
            for hl in HORIZONS_MAP:
                if jd_mc and hl in jd_mc:
                    pv  = jd_mc[hl]["price_base"]
                    pct = jd_mc[hl]["ret_base"]
                    jd_row += _cell(pv, pct)
                else:
                    jd_row += _dash()
            table_rows.append(f"<tr style='border-bottom:1px solid #1a1a1a'>{jd_row}</tr>")

            # Row 4: Technical Score → short-term price target (1D–3M only)
            tech_lbl = f"⚡ Technical Score ({_tech_score:.0f}/100)"
            tech_row = f"<td style='padding:6px 10px;color:#aaa'>{tech_lbl}</td>"
            for hl, days in HORIZONS_MAP.items():
                if hl in TECH_VALID:
                    pv, pct = _score_price(_tech_annual, days)
                    tech_row += _cell(pv, pct)
                else:
                    tech_row += _dash()
            table_rows.append(f"<tr style='border-bottom:1px solid #111'>{tech_row}</tr>")

            # Row 4: Fundamental Score → medium/long-term price target (3M–1Y only)
            fund_lbl = f"💰 Fundamental Score ({_fund_score:.0f}/100)"
            fund_row = f"<td style='padding:6px 10px;color:#aaa'>{fund_lbl}</td>"
            for hl, days in HORIZONS_MAP.items():
                if hl in FUND_VALID:
                    pv, pct = _score_price(_fund_annual, days)
                    fund_row += _cell(pv, pct)
                else:
                    fund_row += _dash()
            table_rows.append(f"<tr style='border-bottom:1px solid #111'>{fund_row}</tr>")

            # Row 5: DCF Valuation → partial convergence toward intrinsic value (6M, 1Y)
            dcf_row = "<td style='padding:6px 10px;color:#aaa'>📋 DCF Valuation</td>"
            for hl, days in HORIZONS_MAP.items():
                if hl in LONG_VALID and t1_dcf:
                    conv = 0.20 if hl == "6M" else 0.35
                    pv   = max(1.0, round(current_price + (t1_dcf["intrinsic"] - current_price) * conv, 2))
                    pct  = round((pv / current_price - 1) * 100, 1)
                    dcf_row += _cell(pv, pct)
                else:
                    dcf_row += _dash()
            table_rows.append(f"<tr style='border-bottom:1px solid #111'>{dcf_row}</tr>")

            # Row 6: Relative Valuation → P/E-normalised fair price (6M, 1Y)
            rv_lbl  = f"📐 Relative Valuation ({_rv_verdict})"
            rv_row  = f"<td style='padding:6px 10px;color:#aaa'>{rv_lbl}</td>"
            for hl, days in HORIZONS_MAP.items():
                if hl in LONG_VALID and _rv_fair_price:
                    conv = 0.20 if hl == "6M" else 0.35
                    pv   = max(1.0, round(current_price + (_rv_fair_price - current_price) * conv, 2))
                    pct  = round((pv / current_price - 1) * 100, 1)
                    rv_row += _cell(pv, pct)
                else:
                    rv_row += _dash()
            table_rows.append(f"<tr style='border-bottom:1px solid #111'>{rv_row}</tr>")

            # Row 7: XGBoost (1M, 3M, 1Y only)
            if xgb_pred_t1:
                xgb_row = "<td style='padding:6px 10px;color:#aaa'>🤖 XGBoost (ML)</td>"
                for hl in HORIZONS_MAP:
                    key = XGB_KEY.get(hl)
                    if key and key in xgb_pred_t1:
                        ret = float(xgb_pred_t1[key])
                        pv  = round(current_price * (1 + ret), 2)
                        xgb_row += _cell(pv, round(ret * 100, 1))
                    else:
                        xgb_row += _dash()
                table_rows.append(f"<tr style='border-bottom:1px solid #111'>{xgb_row}</tr>")

            # Consensus row — weighted average of all available predictions at each horizon
            consensus_row = "<td style='padding:6px 10px;color:#ffcc00;font-weight:700'>⚖️ Model Consensus</td>"
            for hl, days in HORIZONS_MAP.items():
                pvs = [t1_mc(hl, "price_base", current_price)]   # MC P50 always included
                if hl in TECH_VALID:
                    pvs.append(_score_price(_tech_annual, days)[0])
                if hl in FUND_VALID:
                    pvs.append(_score_price(_fund_annual, days)[0])
                if hl in LONG_VALID and t1_dcf:
                    conv = 0.20 if hl == "6M" else 0.35
                    pvs.append(max(1.0, current_price + (t1_dcf["intrinsic"] - current_price) * conv))
                if hl in LONG_VALID and _rv_fair_price:
                    conv = 0.20 if hl == "6M" else 0.35
                    pvs.append(max(1.0, current_price + (_rv_fair_price - current_price) * conv))
                # Jump Diffusion in consensus
                if jd_mc and hl in jd_mc:
                    pvs.append(jd_mc[hl]["price_base"])
                xk = XGB_KEY.get(hl)
                if xk and xgb_pred_t1 and xk in xgb_pred_t1:
                    pvs.append(current_price * (1 + float(xgb_pred_t1[xk])))
                avg_pv  = round(float(np.mean(pvs)), 2)
                avg_pct = round((avg_pv / current_price - 1) * 100, 1)
                consensus_row += _cell(avg_pv, avg_pct, is_mc=True)
            table_rows.append(f"<tr style='background:#1a1a00;border-top:2px solid #444'>{consensus_row}</tr>")

            st.markdown(
                f"""<div style="overflow-x:auto">
                <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
                  <thead>
                    <tr style="border-bottom:2px solid #333">
                      <th style="padding:6px 10px;color:#666;font-weight:500;text-align:left">Model</th>
                      {hl_headers}
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(table_rows)}
                  </tbody>
                </table></div>""",
                unsafe_allow_html=True,
            )

            st.caption(
                "💡 **How to read:** Green = model predicts price rise, Red = price fall. "
                "⚡ Jump Diffusion MC adds earnings gap risk to standard MC paths. "
                "Technical Score drives short-term targets (1D–3M). "
                "💰 Fundamental Score drives medium-term targets (3M–1Y). "
                "📋 DCF and 📐 Relative Valuation show partial convergence toward fair value (6M–1Y only). "
                "🤖 XGBoost predicts 1M, 3M, 1Y. "
                "⚖️ Consensus = average of all available models at each horizon."
            )

            st.divider()

            # ── Smart Money Signals ───────────────────────────────────────────────────
            st.markdown("### 🎯 Smart Money Signals")
            sm_col1, sm_col2 = st.columns(2)

            with sm_col1:
                # Options signals
                with st.spinner("Fetching options chain…"):
                    opts = compute_options_signals(ticker, current_price)

                st.markdown("#### 📊 Options Market Signals")
                if opts:
                    oc1, oc2, oc3 = st.columns(3)
                    oc1.metric("ATM Implied Vol", f"{opts['atm_iv']:.1f}%",
                               help="Current market expectation of future volatility priced into options")
                    pc = opts.get("pc_ratio")
                    pc_delta = "🔴 Fear / hedging" if (pc and pc > 1) else ("🟢 Greed / bullish" if (pc and pc < 0.7) else "⚪ Neutral")
                    oc2.metric("Put/Call Ratio", f"{pc:.2f}" if pc else "—",
                               delta=pc_delta,
                               help=">1 = more puts than calls = market hedging; <0.7 = complacency")
                    sk = opts.get("skew")
                    sk_label = "🔴 Crash protection bid" if (sk and sk > 5) else ("🟢 Call demand" if (sk and sk < -2) else "⚪ Balanced")
                    oc3.metric("OTM Skew (put−call)", f"{sk:+.1f}%" if sk else "—",
                               delta=sk_label,
                               help="Positive = OTM puts more expensive than OTM calls = market fears downside")
                    ua_c = opts.get("unusual_calls", 0)
                    ua_p = opts.get("unusual_puts", 0)
                    if ua_c > 0 or ua_p > 0:
                        if ua_c > ua_p:
                            st.success(f"⚡ Unusual call activity: {ua_c} strike(s) with volume >3× open interest — potential bullish positioning")
                        elif ua_p > ua_c:
                            st.warning(f"⚡ Unusual put activity: {ua_p} strike(s) with volume >3× open interest — potential hedging or bearish bet")
                        else:
                            st.info(f"⚡ Unusual activity on both calls ({ua_c}) and puts ({ua_p}) — ambiguous, possible straddle play")
                    st.caption(f"Options expiry used: {opts['exp_date']}")
                else:
                    st.info("Options data unavailable for this ticker (common for NSE/ETFs).")

                # Kelly Criterion — regime-adjusted
                st.markdown("#### 🎲 Kelly Criterion — Regime-Adjusted Position Size")
                kelly = compute_kelly(mc, f)
                if kelly:
                    regime_mult   = _rd.get("kelly_mult", 1.0)
                    adj_size      = round(kelly["kelly_quarter"] * regime_mult, 1)
                    kc1, kc2, kc3, kc4 = st.columns(4)
                    kc1.metric("Win Probability", f"{kelly['win_prob']:.1f}%",
                               help="% of MC simulations ending above entry at 3M")
                    kc2.metric("Win / Loss Ratio", f"{kelly['win_loss_ratio']:.2f}x",
                               help="P90 upside ÷ stop-loss distance")
                    kc3.metric("Quarter-Kelly (base)", f"{kelly['kelly_quarter']:.1f}%",
                               help="25% of full Kelly — model-based size before regime adjustment")
                    kc4.metric(f"Regime-Adjusted Size ({_rd['regime']})",
                               f"{adj_size:.1f}%",
                               delta=f"×{regime_mult:.0%} regime multiplier",
                               delta_color="normal",
                               help="Base Kelly × regime multiplier. In BEAR/CRISIS, cut size automatically.")
                    verdict_color = kelly["color"] if adj_size > 0 else "#888"
                    verdict_label = kelly["verdict"] if adj_size > 0 else "SKIP"
                    st.markdown(
                        f"<div style='padding:10px 14px;border-radius:6px;background:#111;"
                        f"border-left:4px solid {verdict_color};margin-top:6px'>"
                        f"<span style='color:{verdict_color};font-weight:700;font-size:1rem'>"
                        f"{verdict_label} POSITION</span>"
                        f"<span style='color:#888;font-size:0.82rem;margin-left:12px'>"
                        f"Full Kelly {kelly['kelly_full']:.1f}% → ¼-Kelly {kelly['kelly_quarter']:.1f}%"
                        f" → Regime-adjusted <b style='color:{_rd['color']}'>{adj_size:.1f}%</b> of portfolio"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Kelly sizes positions by edge × odds. Regime multiplier ({regime_mult:.0%} in {_rd['regime']}) "
                        "scales down automatically in bear/volatile markets. SKIP = negative expected value at this price."
                    )
                else:
                    st.info("Kelly sizing unavailable — missing MC or price data.")

            with sm_col2:
                # Short interest signal
                st.markdown("#### 📉 Short Interest")
                short_ratio   = _safe_float(f.get("short_ratio"))
                short_float   = _safe_float(f.get("short_float"))
                if short_ratio or short_float:
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Days to Cover", f"{short_ratio:.1f}d" if short_ratio else "—",
                               help="Short interest ÷ average daily volume. >5 = hard to cover fast (short squeeze potential)")
                    sc2.metric("Short % of Float", f"{short_float*100:.1f}%" if short_float else "—",
                               help=">20% = heavily shorted; if price rises, forced covering accelerates the move")
                    if short_ratio and short_ratio > 5:
                        st.warning(f"⚠️ Days-to-cover = {short_ratio:.1f} — high short ratio. Any positive catalyst could trigger a short squeeze.")
                    elif short_float and short_float > 0.20:
                        st.warning(f"⚠️ {short_float*100:.0f}% of float is short — crowded short. Watch for squeeze on positive news.")
                    elif short_ratio and short_ratio < 2:
                        st.success(f"✅ Low short interest ({short_ratio:.1f} days to cover) — market not positioned against this stock.")
                else:
                    st.info("Short interest data unavailable (common for NSE stocks).")

                # Earnings estimate revisions
                st.markdown("#### 📈 Earnings Estimate Revisions")
                st.caption("Rising estimates = analysts getting more bullish = leading price indicator")
                with st.spinner("Fetching EPS trends…"):
                    revisions = compute_earnings_revision(ticker)
                if revisions:
                    rev_rows = []
                    for period, data in revisions.items():
                        rev_30 = data.get("rev_30d")
                        rev_60 = data.get("rev_60d")
                        curr_e = data.get("current")
                        color_30 = "#00c853" if (rev_30 and rev_30 > 0) else ("#ff5252" if (rev_30 and rev_30 < 0) else "#888")
                        color_60 = "#00c853" if (rev_60 and rev_60 > 0) else ("#ff5252" if (rev_60 and rev_60 < 0) else "#888")
                        rev_rows.append(
                            f"<tr style='border-bottom:1px solid #1a1a1a'>"
                            f"<td style='padding:5px 8px;color:#aaa'>{period}</td>"
                            f"<td style='padding:5px 8px;text-align:right;color:#fff'>{curr}{curr_e:.2f}</td>"
                            f"<td style='padding:5px 8px;text-align:right;color:{color_30}'>"
                            f"{'—' if rev_30 is None else f'{rev_30:+.1f}%'}</td>"
                            f"<td style='padding:5px 8px;text-align:right;color:{color_60}'>"
                            f"{'—' if rev_60 is None else f'{rev_60:+.1f}%'}</td>"
                            f"</tr>"
                        )
                    if rev_rows:
                        st.markdown(
                            f"""<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:0.82rem">
                            <thead><tr style="border-bottom:2px solid #333">
                            <th style="padding:5px 8px;color:#666">Period</th>
                            <th style="padding:5px 8px;color:#666;text-align:right">Current EPS</th>
                            <th style="padding:5px 8px;color:#666;text-align:right">vs 30d ago</th>
                            <th style="padding:5px 8px;color:#666;text-align:right">vs 60d ago</th>
                            </tr></thead><tbody>{''.join(rev_rows)}</tbody></table></div>""",
                            unsafe_allow_html=True,
                        )
                        # Signal interpretation
                        all_revs = [d.get("rev_30d") for d in revisions.values() if d.get("rev_30d") is not None]
                        if all_revs:
                            avg_rev = sum(all_revs) / len(all_revs)
                            if avg_rev > 2:
                                st.success(f"✅ Analysts have revised EPS estimates UP {avg_rev:+.1f}% on average — bullish leading signal.")
                            elif avg_rev < -2:
                                st.warning(f"⚠️ Analysts have cut EPS estimates {avg_rev:+.1f}% on average — earnings risk ahead.")
                            else:
                                st.info(f"⚪ EPS estimates roughly stable ({avg_rev:+.1f}% avg revision) — no strong analyst signal.")
                else:
                    st.info("Earnings estimate data unavailable for this ticker.")

            st.divider()

            # ── Analysis bullets (Tab 1) ─────────────────────────────────────────────
            st.markdown("### 💡 Analysis & Recommendation")
            t1_bullets = generate_analysis_bullets(f, mc, score, bkdn, dcf=t1_dcf, curr=curr)

            for i, bullet in enumerate(t1_bullets, 1):
                st.markdown(f"{bullet}")

            st.divider()

            # ── Key fundamentals row ──────────────────────────────────────────────────
            st.markdown("#### 📊 Key Fundamentals")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            def fv(key, suffix="", scale=1, dec=1):
                v = f.get(key)
                if v is None: return "—"
                try:
                    flt = float(v) * scale
                    return f"{flt:.{dec}f}{suffix}" if flt == flt else "—"
                except Exception:
                    return "—"
            k1.metric("P/E",         fv("pe", "x", dec=1))
            k2.metric("ROE",         fv("roe", "%", 100, 1))
            k3.metric("Net Margin",  fv("net_margin", "%", 100, 1))
            k4.metric("Rev Growth",  fv("rev_growth", "%", 100, 1))
            k5.metric("D/E Ratio",   fv("de_ratio", dec=1))
            k6.metric("Beta",        fv("beta", dec=2))

            st.caption(
                f"Analysts: {n_ana} covering | "
                f"Analyst upside: {fv('analyst_upside', '%', 100, 1)} | "
                f"Insider: {fv('insider', '%', 100, 1)} | "
                f"Institutional: {fv('inst_holding', '%', 100, 1)}"
            )

            # ── Drift breakdown (what's driving the Monte Carlo prediction) ──────────
            drift = mc.get("drift", {}) if mc else {}
            if drift:
                st.markdown("#### 🔬 What's Driving the Monte Carlo Prediction")
                st.caption(
                    "The prediction blends 3 inputs: recent price behaviour (20%), "
                    "fundamental-implied return (50%), and momentum signals (30%). "
                    "This is why a falling stock with strong fundamentals shows a better "
                    "median path than pure price extrapolation would give."
                )

                # Drift components
                d1, d2, d3, d4 = st.columns(4)
                def drift_color(v):
                    return "#00c853" if v > 0 else ("#ff5252" if v < 0 else "#888")

                blended = drift.get("blended_pct", 0)
                h_pct  = drift.get("hist_pct",  0)
                fn_pct = drift.get("fund_annual", 0)
                m_pct  = drift.get("mom_pct",   0)
                h_c    = drift_color(h_pct)
                fn_c   = drift_color(fn_pct)
                m_c    = drift_color(m_pct)

                d1.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='color:#888;font-size:0.75rem'>Historical drift (20%)</div>"
                    f"<div style='color:{h_c};font-size:1.2rem;font-weight:700'>{h_pct:+.1f}%/yr</div>"
                    f"<div style='color:#666;font-size:0.72rem'>What price has been doing</div></div>",
                    unsafe_allow_html=True,
                )
                d2.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='color:#888;font-size:0.75rem'>Fundamental drift (50%)</div>"
                    f"<div style='color:{fn_c};font-size:1.2rem;font-weight:700'>{fn_pct:+.1f}%/yr</div>"
                    f"<div style='color:#666;font-size:0.72rem'>What the business justifies</div></div>",
                    unsafe_allow_html=True,
                )
                d3.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='color:#888;font-size:0.75rem'>Momentum (30%)</div>"
                    f"<div style='color:{m_c};font-size:1.2rem;font-weight:700'>{m_pct:+.1f}%/yr</div>"
                    f"<div style='color:#666;font-size:0.72rem'>Current price direction</div></div>",
                    unsafe_allow_html=True,
                )
                d4.markdown(
                    f"<div style='text-align:center;background:#111;border-radius:8px;padding:8px'>"
                    f"<div style='color:#888;font-size:0.75rem'>Blended drift</div>"
                    f"<div style='color:{drift_color(blended)};font-size:1.4rem;font-weight:800'>"
                    f"{blended:+.1f}%/yr</div>"
                    f"<div style='color:#aaa;font-size:0.72rem'>Used in simulation</div></div>",
                    unsafe_allow_html=True,
                )

                # Factor breakdown table
                factors = drift.get("factors", {})
                if factors:
                    with st.expander("📋 See all factors driving the fundamental drift", expanded=False):
                        frows = []
                        for fname, (contribution, description) in factors.items():
                            frows.append({
                                "Factor":      fname,
                                "Impact":      f"{contribution*100:+.1f}%/yr",
                                "Description": description,
                            })
                        fdf = pd.DataFrame(frows)

                        def color_impact(val):
                            try:
                                v = float(val.replace("%/yr",""))
                                if v > 0:  return "color:#00c853;font-weight:bold"
                                if v < 0:  return "color:#ff5252;font-weight:bold"
                                return "color:#888"
                            except Exception:
                                return ""

                        st.dataframe(
                            fdf.style.map(color_impact, subset=["Impact"]),
                            use_container_width=True, hide_index=True,
                        )

            st.divider()

            # ── Model Consensus Panel ────────────────────────────────────────────────
            st.markdown("### 🧭 Model Consensus — Which One Should You Trust?")
            st.caption(
                "Each model uses different data and is optimised for a different time horizon. "
                "Reading them together gives a more complete picture than any single model alone."
            )

            # Holding-period selector
            horizon_sel = st.radio(
                "Your holding period:",
                ["< 1 Month (Trader)", "1–6 Months (Swing)", "> 6 Months (Investor)"],
                horizontal=True, index=1, key="horizon_sel_tab1",
            )

            # Compute DCF and relative valuation
            dcf_result  = compute_dcf(f)
            rel_verdict, rel_signals, rel_pct = compute_relative_valuation(f)

            # Monte Carlo verdict
            mc3m_ret = t1_mc("3M", "ret_base", 0)
            mc1y_ret = t1_mc("1Y", "ret_base", 0)
            mc3_prob = t1_mc("3M", "prob_positive", 50)

            if   mc3m_ret > 10:  mc_verdict_3m = "BUY";       mc_c3 = "#00c853"
            elif mc3m_ret > 0:   mc_verdict_3m = "HOLD";      mc_c3 = "#ffcc00"
            elif mc3m_ret > -10: mc_verdict_3m = "HOLD";      mc_c3 = "#ffcc00"
            else:                mc_verdict_3m = "AVOID";     mc_c3 = "#ff5252"

            if   mc1y_ret > 15:  mc_verdict_1y = "BUY";       mc_c1 = "#00c853"
            elif mc1y_ret > 0:   mc_verdict_1y = "HOLD";      mc_c1 = "#ffcc00"
            else:                mc_verdict_1y = "AVOID";     mc_c1 = "#ff5252"

            # Technical verdict from score
            if   score >= 75: tech_verdict = "STRONG BUY"; tech_c = "#00ff7f"
            elif score >= 55: tech_verdict = "BUY";        tech_c = "#00c853"
            elif score >= 40: tech_verdict = "HOLD";       tech_c = "#ffcc00"
            else:             tech_verdict = "SELL";        tech_c = "#ff5252"

            # Fundamental model verdict from score
            fund_sc = float(bkdn.get("fundamental", 50) or 50)
            if   fund_sc >= 65: fund_verdict = "STRONG BUY"; fund_c = "#00c853"
            elif fund_sc >= 50: fund_verdict = "BUY";        fund_c = "#00c853"
            elif fund_sc >= 35: fund_verdict = "HOLD";       fund_c = "#ffcc00"
            else:               fund_verdict = "SELL";       fund_c = "#ff5252"

            # DCF verdict
            dcf_v = dcf_result["verdict"] if dcf_result else "No data"
            dcf_c = {"STRONG BUY":"#00c853","BUY":"#00c853","HOLD":"#ffcc00",
                      "SELL":"#ff5252","STRONG SELL":"#ff5252","No data":"#555"}.get(dcf_v, "#888")

            # Relative valuation verdict
            rel_c = {"CHEAP":"#00c853","FAIR":"#00c853","FAIRLY PRICED":"#ffcc00",
                     "EXPENSIVE":"#ff9100","VERY EXPENSIVE":"#ff5252"}.get(rel_verdict, "#888")

            VERDICT_WEIGHT = {
                "< 1 Month (Trader)":   {"mc_3m": 0.45, "technical": 0.35, "dcf": 0.05, "fundamental": 0.10, "relative": 0.05},
                "1–6 Months (Swing)":   {"mc_3m": 0.25, "technical": 0.25, "dcf": 0.20, "fundamental": 0.20, "relative": 0.10},
                "> 6 Months (Investor)": {"mc_3m": 0.05, "technical": 0.10, "dcf": 0.35, "fundamental": 0.30, "relative": 0.20},
            }
            weights = VERDICT_WEIGHT[horizon_sel]

            SCORE_MAP = {
                "STRONG BUY": 2, "BUY": 1, "HOLD": 0,
                "FAIRLY PRICED": 0, "FAIR": 1, "CHEAP": 2,
                "SELL": -1, "STRONG SELL": -2,
                "AVOID": -1, "EXPENSIVE": -1, "VERY EXPENSIVE": -2,
                "No data": 0,
            }
            weighted_score = (
                SCORE_MAP.get(mc_verdict_3m,  0) * weights["mc_3m"] +
                SCORE_MAP.get(tech_verdict,   0) * weights["technical"] +
                SCORE_MAP.get(dcf_v,          0) * weights["dcf"] +
                SCORE_MAP.get(fund_verdict,   0) * weights["fundamental"] +
                SCORE_MAP.get(rel_verdict,    0) * weights["relative"]
            )

            if   weighted_score >= 1.2:  final_v = "STRONG BUY";  final_c = "#00ff7f"; final_bg = "#003a1f"
            elif weighted_score >= 0.5:  final_v = "BUY";         final_c = "#00c853"; final_bg = "#002010"
            elif weighted_score >= -0.3: final_v = "HOLD";        final_c = "#ffcc00"; final_bg = "#332900"
            elif weighted_score >= -1.0: final_v = "SELL";        final_c = "#ff5252"; final_bg = "#2a0000"
            else:                        final_v = "STRONG SELL"; final_c = "#d50000"; final_bg = "#1a0000"

            # ── Model table ────────────────────────────────────────────────────────────
            ROWS = [
                ("📊 Monte Carlo (3M)",    mc_verdict_3m,  mc_c3,  f"{mc3m_ret:+.1f}% median · {mc3_prob:.0f}% chance positive",      "< 3 months", "best for: short-term traders, timing entries"),
                ("📊 Monte Carlo (1Y)",    mc_verdict_1y,  mc_c1,  f"{mc1y_ret:+.1f}% median annual drift",                            "1 year",     "based on: price history + fundamental drift blend"),
                ("⚡ Technical Score",     tech_verdict,   tech_c, f"{score:.1f}/100 — RSI, MACD, MA crossovers, volume",              "1–3 months", "best for: momentum and entry timing"),
                ("💰 Fundamental Score",   fund_verdict,   fund_c, f"{fund_sc:.0f}/100 — margins, ROE, growth, debt",                  "6–18 months","best for: business quality assessment"),
                ("🏗️ DCF Valuation",
                    dcf_v, dcf_c,
                    (f"Intrinsic {curr}{dcf_result['intrinsic']:.2f} → {dcf_result['upside_pct']:+.1f}% upside (WACC {dcf_result['wacc']:.1f}%)")
                     if dcf_result else "Insufficient FCF data to compute",
                    "1–5 years", "best for: long-term investors, intrinsic value"),
                ("📐 Relative Valuation",  rel_verdict,    rel_c,  rel_signals[0] if rel_signals else "—",                             "Any",        "best for: is the stock cheap vs market?"),
            ]

            table_html = """
        <table style="width:100%;border-collapse:collapse;font-size:0.83rem;margin:8px 0">
          <tr style="color:#555;font-size:0.75rem;border-bottom:1px solid #222">
            <td style="padding:6px 8px">Model</td>
            <td style="padding:6px 8px">Verdict</td>
            <td style="padding:6px 8px">Signal</td>
            <td style="padding:6px 8px">Best horizon</td>
            <td style="padding:6px 8px">Weight ({h})</td>
          </tr>
        """.format(h=horizon_sel.split("(")[1].rstrip(")"))

            wkeys = ["mc_3m","mc_3m","technical","fundamental","dcf","relative"]
            for i,(model, verd, col, sig, hor, note) in enumerate(ROWS):
                wk  = wkeys[i]
                wt  = weights.get(wk, 0)
                bar = int(wt * 500)   # 20% → 100px
                table_html += f"""
          <tr style="border-bottom:1px solid #1a1a1a">
            <td style="padding:6px 8px;color:#aaa">{model}</td>
            <td style="padding:6px 8px">
              <span style="background:{col}22;color:{col};border:1px solid {col};
              border-radius:4px;padding:2px 8px;font-weight:700;font-size:0.78rem">{verd}</span>
            </td>
            <td style="padding:6px 8px;color:#888;font-size:0.78rem">{sig}</td>
            <td style="padding:6px 8px;color:#666;font-size:0.75rem">{hor}</td>
            <td style="padding:6px 8px">
              <div style="background:#333;border-radius:3px;height:6px;width:120px">
                <div style="background:{col};width:{bar}px;height:100%;border-radius:3px"></div>
              </div>
              <div style="color:#666;font-size:0.72rem">{wt:.0%}</div>
            </td>
          </tr>"""

            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            # ── Final reconciled verdict ───────────────────────────────────────────────
            st.markdown(
                f"""<div style="background:{final_bg};border:2px solid {final_c};
                border-radius:10px;padding:16px 22px;margin-top:8px;
                display:flex;align-items:center;gap:20px">
                <div>
                  <div style="color:#888;font-size:0.78rem;margin-bottom:2px">
                  RECONCILED VERDICT · {horizon_sel}</div>
                  <div style="font-size:2rem;font-weight:900;color:{final_c};letter-spacing:2px">
                  {final_v}</div>
                </div>
                <div style="color:#aaa;font-size:0.85rem;max-width:600px">
                  Weighted consensus of {len(ROWS)} models for a <b style="color:#fff">
                  {horizon_sel.split("(")[0].strip()}</b> holding period.
                  {'Models broadly agree — higher confidence.' if abs(weighted_score) > 1.0
                   else 'Models are mixed — lower confidence, smaller position size recommended.' if abs(weighted_score) > 0.3
                   else 'Models are split — wait for one more confirming signal before entering.'}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Explanation of why models differ
            with st.expander("❓ Why do models disagree? How to read this.", expanded=False):
                st.markdown("""
        **Each model is built for a different purpose — disagreement is normal and informative:**

        | Model | What it measures | Most reliable for | Least reliable for |
        |---|---|---|---|
        | **Monte Carlo (3M)** | Statistical distribution of short-term price paths | Entry timing, short-term risk | Predicting specific events like earnings |
        | **Monte Carlo (1Y)** | Long-run drift blending price history + fundamentals | Macro regime assessment | Individual stock catalysts |
        | **Technical Score** | Current momentum, trend alignment, volume | Timing entries and exits | Telling you if the business is good |
        | **Fundamental Score** | Business quality: margins, ROE, growth, debt | Screening for quality companies | Short-term price direction |
        | **DCF Valuation** | Intrinsic value based on discounted cash flows | Long-term value investors | Stocks with no/negative FCF, high-growth pre-profit |
        | **Relative Valuation** | Cheap vs expensive vs market peers | Sector rotation decisions | Disruptors that deserve a premium |

        **Rule of thumb:**
        - If **≥ 4 of 6 models agree** → High conviction, full position size
        - If **3 of 6 models agree** → Medium conviction, half position
        - If **< 3 models agree** → Low conviction, wait for alignment or skip
                """)

            st.divider()

            # ── Statistical trend chart (collapsible) ─────────────────────────────────
            st.divider()
            with st.expander("📈 Statistical Trend Prediction (Linear / Polynomial / EMA)", expanded=False):
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    model_type = st.selectbox(
                        "Model",
                        ["Linear Regression", "Polynomial (Degree 2)", "Polynomial (Degree 3)", "EMA Projection"],
                    )
                with pc2:
                    train_window = st.selectbox(
                        "Training Window", [63, 126, 252, 504], index=2,
                        format_func=lambda x: f"{x} days (~{x//21}mo)",
                    )
                with pc3:
                    pred_horizon = st.selectbox(
                        "Predict Ahead", [7, 21, 63, 126, 252], index=2,
                        format_func=lambda x: {7:"1W",21:"1M",63:"3M",126:"6M",252:"1Y"}[x],
                    )

                train_df = df.dropna(subset=["Close"]).tail(train_window)
                prices   = train_df["Close"].values.astype(float)
                x_train  = np.arange(len(prices), dtype=float)
                x_future = np.arange(len(prices), len(prices) + pred_horizon, dtype=float)

                if model_type == "Linear Regression":
                    coeffs = np.polyfit(x_train, prices, 1); poly = np.poly1d(coeffs)
                    fit = poly(x_train); pred = poly(x_future)
                elif model_type.startswith("Polynomial (Degree 2"):
                    coeffs = np.polyfit(x_train, prices, 2); poly = np.poly1d(coeffs)
                    fit = poly(x_train); pred = poly(x_future)
                elif model_type.startswith("Polynomial (Degree 3"):
                    coeffs = np.polyfit(x_train, prices, 3); poly = np.poly1d(coeffs)
                    fit = poly(x_train); pred = poly(x_future)
                else:
                    ema_s = pd.Series(prices).ewm(span=20, adjust=False).mean()
                    fit   = ema_s.values
                    slope = (fit[-1] - fit[-min(20, len(fit))]) / min(20, len(fit))
                    pred  = np.array([fit[-1] + slope * i for i in range(1, pred_horizon + 1)])

                residuals  = prices - fit
                sigma      = np.std(residuals)
                time_scale = np.sqrt(np.arange(1, pred_horizon + 1))
                upper = pred + 1.96 * sigma * time_scale
                lower = pred - 1.96 * sigma * time_scale
                last_date    = train_df.index[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_horizon)

                pfig = go.Figure()
                pfig.add_trace(go.Scatter(x=train_df.index, y=prices, name="Historical",
                                          line=dict(color="white", width=1.5)))
                pfig.add_trace(go.Scatter(x=train_df.index, y=fit, name="Model Fit",
                                          line=dict(color="yellow", width=1, dash="dot"), opacity=0.7))
                pfig.add_trace(go.Scatter(x=future_dates, y=upper, name="Upper 95%",
                                          line=dict(color="rgba(0,220,120,0.3)", width=1)))
                pfig.add_trace(go.Scatter(x=future_dates, y=lower, name="Lower 95%",
                                          fill="tonexty", fillcolor="rgba(0,220,120,0.07)",
                                          line=dict(color="rgba(0,220,120,0.3)", width=1)))
                pfig.add_trace(go.Scatter(x=future_dates, y=pred, name="Forecast",
                                          line=dict(color="cyan", width=2.5)))
                pfig.add_vline(x=last_date.timestamp() * 1000,
                               line=dict(color="gray", width=1, dash="dash"),
                               annotation_text="Forecast starts", annotation_position="top left")
                pfig.update_layout(template="plotly_dark", height=420, showlegend=True,
                                   title=f"{ticker} · {model_type} · {pred_horizon}d forecast",
                                   yaxis_title=f"Price ({curr})", margin=dict(t=40, b=20))
                st.plotly_chart(pfig, use_container_width=True)

                rmse = np.sqrt(np.mean(residuals**2))
                mape = np.mean(np.abs(residuals / prices)) * 100
                r2   = 1 - np.sum(residuals**2) / np.sum((prices - np.mean(prices))**2)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RMSE", f"{curr}{rmse:.2f}")
                mc2.metric("MAPE", f"{mape:.2f}%")
                mc3.metric("R²",   f"{r2:.3f}")
                st.caption("⚠️ Statistical models extrapolate price patterns only — no fundamentals.")

            # ── Backtest Validator ────────────────────────────────────────────────────
            st.divider()
            st.markdown("### 🧪 Backtest Validator")
            st.caption(
                "Run the model on historical data up to a cutoff date and compare "
                "its predictions against what actually happened. Reveals what the model "
                "can and cannot predict."
            )

            bt_col1, bt_col2 = st.columns([2, 1])
            with bt_col1:
                bt_cutoff = st.date_input(
                    "Cutoff date (model sees data UP TO this date, predicts AFTER it)",
                    value=datetime.date(2026, 5, 15),
                    max_value=datetime.date.today() - datetime.timedelta(days=2),
                    key="bt_cutoff",
                )
            with bt_col2:
                run_bt = st.button("▶ Run Backtest", type="primary",
                                   key="run_backtest_btn", use_container_width=True)

            if run_bt:
                with st.spinner(f"Running model on data up to {bt_cutoff}…"):
                    try:
                        cutoff_ts = pd.Timestamp(bt_cutoff)
                        start_ts  = cutoff_ts - pd.Timedelta(days=420)

                        # Fetch pre-cutoff price data
                        bt_df = yf.Ticker(ticker).history(
                            start=start_ts,
                            end=cutoff_ts + pd.Timedelta(days=1)
                        )

                        if bt_df.empty or len(bt_df) < 60:
                            st.error("Not enough data before the cutoff date. Choose an earlier date.")
                        else:
                            bt_close = bt_df["Close"].dropna().astype(float)
                            bt_high  = bt_df["High"].dropna().astype(float)
                            bt_low   = bt_df["Low"].dropna().astype(float)
                            bt_vol   = bt_df["Volume"].dropna().astype(float)
                            bt_price = float(bt_close.iloc[-1])

                            # ── Compute indicators on pre-cutoff data ──────────────────
                            bt_daily = bt_close.pct_change().dropna()
                            bt_mu    = float(bt_daily.mean())
                            bt_sigma = float(bt_daily.std()) or 0.015
                            bt_kurt  = float(stats.kurtosis(bt_daily))

                            bt_delta = bt_close.diff()
                            bt_gain  = bt_delta.clip(lower=0).rolling(14).mean()
                            bt_loss  = (-bt_delta.clip(upper=0)).rolling(14).mean()
                            bt_rsi   = float((100 - 100/(1 + bt_gain/bt_loss.replace(0,np.nan))).iloc[-1])

                            bt_macd_line = bt_close.ewm(span=12).mean() - bt_close.ewm(span=26).mean()
                            bt_macd_sig  = bt_macd_line.ewm(span=9).mean()
                            bt_macd_hist = float((bt_macd_line - bt_macd_sig).iloc[-1])

                            bt_ma50  = float(bt_close.rolling(50).mean().iloc[-1])
                            bt_ma200 = float(bt_close.rolling(min(200,len(bt_close))).mean().iloc[-1])

                            bt_mom1m = float((bt_close.iloc[-1]/bt_close.iloc[-22]-1)) if len(bt_close)>=22 else 0
                            bt_mom3m = float((bt_close.iloc[-1]/bt_close.iloc[-63]-1)) if len(bt_close)>=63 else 0

                            h_l = bt_high - bt_low
                            h_pc = (bt_high - bt_close.shift()).abs()
                            l_pc = (bt_low  - bt_close.shift()).abs()
                            bt_atr = float(pd.concat([h_l,h_pc,l_pc],axis=1).max(axis=1).rolling(14).mean().iloc[-1])

                            # ── Approximate fundamental drift using current info ────────
                            try:
                                bt_info = _yf_info(ticker)
                                bt_au   = (bt_info.get("targetMeanPrice",0) or 0) / bt_price - 1
                                bt_rg   = bt_info.get("revenueGrowth",0.10) or 0.10
                                bt_roe  = bt_info.get("returnOnEquity",0.15) or 0.15
                                bt_peg  = bt_info.get("pegRatio",2.0) or 2.0
                            except Exception:
                                bt_au=0.20; bt_rg=0.10; bt_roe=0.15; bt_peg=2.0

                            bt_fund = 0
                            if bt_peg < 1.2: bt_fund += 0.08
                            elif bt_peg > 2.0: bt_fund -= 0.08
                            bt_fund += float(np.clip(bt_au * 0.40, -0.15, 0.20))
                            bt_fund += float(np.clip(bt_rg * 0.60, -0.12, 0.18))
                            if bt_roe > 0.15: bt_fund += 0.02

                            bt_mu_adj = (bt_mu*0.20 + (bt_fund/252)*0.50 +
                                         ((bt_mom1m*0.3 + bt_mom3m*0.4)/252)*0.30)
                            if bt_rsi > 75: bt_mu_adj -= 0.0003
                            elif bt_rsi < 25: bt_mu_adj += 0.0003

                            # ── Run MC simulations ─────────────────────────────────────
                            np.random.seed(int(abs(bt_price*100)) % 99991 + bt_cutoff.toordinal())
                            bt_df_t = max(3, int(30 - abs(bt_kurt)))

                            HORIZONS = {"1 Day": 1, "1 Week": 5, "1 Month": 21}
                            bt_preds = {}
                            for label, days in HORIZONS.items():
                                sh = stats.t.rvs(df=bt_df_t, size=(10000, days)) * bt_sigma + bt_mu_adj
                                pr = bt_price * np.prod(1 + sh, axis=1)
                                bt_preds[label] = {
                                    "p10":  float(np.percentile(pr, 10)),
                                    "p50":  float(np.percentile(pr, 50)),
                                    "p90":  float(np.percentile(pr, 90)),
                                    "p99":  float(np.percentile(pr, 99)),
                                    "prob_pos": float(np.mean(pr > bt_price) * 100),
                                    "prices": pr,
                                }

                            # ── Fetch actual prices after cutoff ───────────────────────
                            actual_df = yf.Ticker(ticker).history(
                                start=cutoff_ts + pd.Timedelta(days=1),
                                end=cutoff_ts + pd.Timedelta(days=35)
                            )

                            # ── Display results ────────────────────────────────────────
                            st.markdown(f"#### Results: {ticker} · Model cutoff {bt_cutoff}")

                            # Pre-cutoff snapshot
                            snap = st.columns(5)
                            snap[0].metric("Base Price (cutoff)", f"{curr}{bt_price:.2f}")
                            snap[1].metric("RSI at cutoff", f"{bt_rsi:.1f}")
                            snap[2].metric("MACD histogram", f"{bt_macd_hist:+.3f}")
                            snap[3].metric("vs MA50",  f"{'↑' if bt_price>bt_ma50  else '↓'} {curr}{bt_ma50:.2f}")
                            snap[4].metric("vs MA200", f"{'↑' if bt_price>bt_ma200 else '↓'} {curr}{bt_ma200:.2f}")

                            st.caption(
                                f"Blended drift: historical {bt_mu*0.20*252*100:+.1f}%/yr + "
                                f"fundamental {(bt_fund/252)*0.50*252*100:+.1f}%/yr + "
                                f"momentum {((bt_mom1m*0.3+bt_mom3m*0.4)/252)*0.30*252*100:+.1f}%/yr "
                                f"= **{bt_mu_adj*252*100:+.1f}%/yr blended**"
                            )

                            # Prediction vs Actual table
                            bt_rows = []
                            for label, days in HORIZONS.items():
                                pred = bt_preds[label]

                                # Find actual price at this horizon
                                target_date = cutoff_ts + pd.Timedelta(days=days + 2)
                                actual_slice = actual_df[actual_df.index <= target_date]
                                if not actual_slice.empty:
                                    actual_price = float(actual_slice["Close"].iloc[-1])
                                    actual_ret   = (actual_price - bt_price) / bt_price * 100
                                    pctile       = float(np.mean(pred["prices"] <= actual_price) * 100)
                                    if actual_price < pred["p10"]:
                                        outcome = "🔴 BELOW BEAR (tail event)"
                                    elif actual_price > pred["p99"]:
                                        outcome = "🌟 ABOVE P99 (extreme tail)"
                                    elif actual_price > pred["p90"]:
                                        outcome = "🔵 ABOVE P90 (strong bull)"
                                    elif actual_price > pred["p50"]:
                                        outcome = "🟢 ABOVE MEDIAN"
                                    elif actual_price >= pred["p10"]:
                                        outcome = "🟡 WITHIN RANGE"
                                    else:
                                        outcome = "🔴 BELOW RANGE"
                                    actual_str  = f"{curr}{actual_price:.2f} ({actual_ret:+.1f}%)"
                                    outcome_str = f"{outcome} · {pctile:.0f}th %ile"
                                else:
                                    actual_str = "Not yet available"
                                    outcome_str = "—"

                                bt_rows.append({
                                    "Horizon":    label,
                                    f"P10 Bear ({curr})":  f"{curr}{pred['p10']:.2f} ({(pred['p10']/bt_price-1)*100:+.1f}%)",
                                    f"P50 Base ({curr})":  f"{curr}{pred['p50']:.2f} ({(pred['p50']/bt_price-1)*100:+.1f}%)",
                                    f"P90 Bull ({curr})":  f"{curr}{pred['p90']:.2f} ({(pred['p90']/bt_price-1)*100:+.1f}%)",
                                    "Prob Positive": f"{pred['prob_pos']:.1f}%",
                                    "ACTUAL":        actual_str,
                                    "Outcome":       outcome_str,
                                })

                            bt_df_display = pd.DataFrame(bt_rows)
                            st.dataframe(bt_df_display, use_container_width=True, hide_index=True)

                            # Interpretation
                            st.info(
                                "**How to read this:** Each row shows the model's 10,000-path distribution "
                                "vs what actually happened. If the actual price falls inside P10–P90, "
                                "the model correctly bounded the outcome. Outside that range = a tail event "
                                "the model couldn't predict (earnings surprise, M&A, macro shock, etc.).",
                                icon="📊"
                            )

                    except Exception as e:
                        st.error(f"Backtest error: {e}")

            # ── Add to Watchlist ──────────────────────────────────────────────────────
            st.divider()
            st.markdown("### 📌 Watchlist")

            wl_col1, wl_col2, wl_col3 = st.columns([2, 2, 1])
            with wl_col1:
                st.markdown(f"**Analysing:** `{ticker}`")
                st.caption("Click below to add this stock to your watchlist with Monte Carlo price targets.")
            with wl_col2:
                add_btn = st.button("➕ Add to Watchlist", key="add_watchlist_btn", use_container_width=True)
            with wl_col3:
                refresh_btn = st.button("🔄 Refresh Prices", key="refresh_watchlist_btn", use_container_width=True)

            if add_btn:
                with st.spinner(f"Building price targets for {ticker}…"):
                    entry = build_watchlist_entry(ticker, curr)
                if entry:
                    wl = load_watchlist()
                    # Avoid duplicates — replace if already present
                    wl = [e for e in wl if e["symbol"] != ticker]
                    wl.append(entry)
                    save_watchlist(wl)
                    st.success(f"✅ {ticker} added to watchlist!")
                else:
                    st.error(f"Could not fetch data for {ticker}. Check ticker and try again.")

            if refresh_btn:
                with st.spinner("Refreshing live prices…"):
                    wl = refresh_watchlist_prices(load_watchlist())
                    save_watchlist(wl)
                st.success("Prices updated.")

            # ── Render Watchlist Table ────────────────────────────────────────────────
            wl = load_watchlist()

            if not wl:
                st.info("Your watchlist is empty. Search a stock above and click **➕ Add to Watchlist**.")
            else:
                # Build display dataframe
                rows = []
                for e in wl:
                    c   = e.get("currency", "$")
                    p   = e.get("price", 0)
                    tgts = e.get("targets", {})

                    def tp(h):
                        d = tgts.get(h, {})
                        pr = d.get("price", p)
                        rt = d.get("ret", 0)
                        clr = "color:green" if rt >= 0 else "color:red"
                        return f"{c}{pr:.2f} ({rt:+.1f}%)"

                    rr = e.get("rr", 0)
                    rows.append({
                        "Symbol":   e["symbol"],
                        "Name":     (e.get("name") or e["symbol"])[:22],
                        "Price":    f"{c}{p:.2f}",
                        "1D":       tp("1D"),
                        "1W":       tp("1W"),
                        "1M":       tp("1M"),
                        "3M":       tp("3M"),
                        "6M":       tp("6M"),
                        "1Y":       tp("1Y"),
                        "Entry":    f"{c}{e.get('entry', p):.2f}",
                        "Stop Loss":f"{c}{e.get('stop_loss', 0):.2f}",
                        "R/R":      f"{rr:.1f}:1",
                        "Added":    e.get("added_at", "")[:10],
                    })

                wl_df = pd.DataFrame(rows)

                def color_rr(val):
                    try:
                        r = float(val.replace(":1", ""))
                        if r >= 2:   return "color:#00c853;font-weight:bold"
                        if r >= 1:   return "color:#ffcc00"
                        return "color:#ff5252"
                    except Exception:
                        return ""

                st.dataframe(
                    wl_df.style.map(color_rr, subset=["R/R"]),
                    use_container_width=True,
                    hide_index=True,
                    height=min(60 + len(rows) * 38, 500),
                )

                # Per-row remove buttons
                st.markdown("**Remove from watchlist:**")
                rm_cols = st.columns(min(len(wl), 6))
                for i, e in enumerate(wl):
                    col = rm_cols[i % len(rm_cols)]
                    if col.button(f"✕ {e['symbol']}", key=f"rm_{e['symbol']}_{i}"):
                        updated = [x for x in load_watchlist() if x["symbol"] != e["symbol"]]
                        save_watchlist(updated)
                        st.rerun()


        # ══════════════════════════════════════════════════════════════════════════════

        st.markdown("---")
        # ── SECTION 4: Fundamentals ───────────────────────────────────────────────
        st.markdown("#### 📈 Fundamental Snapshot")
        if True:  # scope block (replaces with t1_sub3:)
            _t3_info = _yf_info(ticker)
            _t3_curr = curr

            _company  = _t3_info.get("shortName") or _t3_info.get("longName") or ticker
            _sector   = _t3_info.get("sector") or _t3_info.get("industry") or "—"
            _exchange = _t3_info.get("fullExchangeName") or _t3_info.get("exchange") or "—"
            st.markdown(f"**{_company}** &nbsp;·&nbsp; {_sector} &nbsp;·&nbsp; {_exchange}")
            if _t3_info.get("longBusinessSummary"):
                with st.expander("Company Overview", expanded=False):
                    st.write(_t3_info["longBusinessSummary"])

            _fund_metrics = [
                ("Market Cap",     _app_fmt(_t3_info.get("marketCap"),          prefix=_t3_curr)),
                ("P/E (TTM)",      _app_fmt(_t3_info.get("trailingPE"),         dec=1)),
                ("Forward P/E",    _app_fmt(_t3_info.get("forwardPE"),          dec=1)),
                ("PEG Ratio",      _app_fmt(_t3_info.get("pegRatio"),           dec=2)),
                ("P/B Ratio",      _app_fmt(_t3_info.get("priceToBook"),        dec=2)),
                ("EPS (TTM)",      _app_fmt(_t3_info.get("trailingEps"),        prefix=_t3_curr, dec=2)),
                ("Revenue (TTM)",  _app_fmt(_t3_info.get("totalRevenue"),       prefix=_t3_curr)),
                ("Rev Growth YoY", _app_fmt(_t3_info.get("revenueGrowth"),      suffix="%", scale=100, dec=1)),
                ("Gross Margin",   _app_fmt(_t3_info.get("grossMargins"),       suffix="%", scale=100, dec=1)),
                ("Net Margin",     _app_fmt(_t3_info.get("profitMargins"),      suffix="%", scale=100, dec=1)),
                ("ROE",            _app_fmt(_t3_info.get("returnOnEquity"),     suffix="%", scale=100, dec=1)),
                ("Debt / Equity",  _app_fmt(_t3_info.get("debtToEquity"),       dec=2)),
                ("Current Ratio",  _app_fmt(_t3_info.get("currentRatio"),       dec=2)),
                ("Beta",           _app_fmt(_t3_info.get("beta"),               dec=2)),
                ("52W High",       _app_fmt(_t3_info.get("fiftyTwoWeekHigh"),   prefix=_t3_curr)),
                ("52W Low",        _app_fmt(_t3_info.get("fiftyTwoWeekLow"),    prefix=_t3_curr)),
            ]
            _fm_cols = st.columns(4)
            for _fi, (_lbl, _val) in enumerate(_fund_metrics):
                _fm_cols[_fi % 4].metric(_lbl, _val)

            # Quarterly financials chart
            _t3_qf = _get_quarterly_data(ticker)
            if not _t3_qf.empty:
                st.markdown("---")
                st.markdown("#### 📊 Quarterly Financials")
                _rev_c = next((c for c in _t3_qf.columns if c == "Total Revenue"), None) or                      next((c for c in _t3_qf.columns if c == "Operating Revenue"), None)
                _ni_c  = next((c for c in _t3_qf.columns if c == "Net Income"), None) or                      next((c for c in _t3_qf.columns if "Net Income Common" in str(c)), None)
                _gp_c  = next((c for c in _t3_qf.columns if c == "Gross Profit"), None)
                _q_dates = [d.strftime("%b '%y") for d in _t3_qf.index[-8:]]
                _qfig = go.Figure()
                if _rev_c: _qfig.add_trace(go.Bar(name="Revenue",      x=_q_dates, y=_t3_qf[_rev_c].iloc[-8:], marker_color="steelblue",      opacity=0.85))
                if _gp_c:  _qfig.add_trace(go.Bar(name="Gross Profit", x=_q_dates, y=_t3_qf[_gp_c].iloc[-8:],  marker_color="mediumseagreen", opacity=0.85))
                if _ni_c:  _qfig.add_trace(go.Bar(name="Net Income",   x=_q_dates, y=_t3_qf[_ni_c].iloc[-8:],  marker_color="gold",           opacity=0.85))
                _qfig.update_layout(barmode="group", template="plotly_dark", height=320,
                                    legend=dict(orientation="h", y=1.1),
                                    yaxis_title=f"Amount ({_t3_curr})", margin=dict(t=30, b=20))
                st.plotly_chart(_qfig, use_container_width=True)
                if _rev_c and len(_t3_qf) >= 4:
                    _ann_rev = _t3_qf[_rev_c].iloc[-4:].sum()
                    _prv_rev = _t3_qf[_rev_c].iloc[-8:-4].sum() if len(_t3_qf) >= 8 else None
                    _qa1, _qa2, _qa3 = st.columns(3)
                    _qa1.metric("TTM Revenue", _app_fmt(_ann_rev, prefix=_t3_curr))
                    if _prv_rev and _prv_rev != 0:
                        _yoy = (_ann_rev - _prv_rev) / abs(_prv_rev) * 100
                        _qa2.metric("YoY Revenue", f"{_yoy:+.1f}%", delta=f"{_yoy:+.1f}%")
                    if _ni_c:
                        _qa3.metric("TTM Net Income", _app_fmt(_t3_qf[_ni_c].iloc[-4:].sum(), prefix=_t3_curr))

            # Comprehensive score
            st.markdown("---")
            st.markdown("#### 🔬 Comprehensive Fundamental Score")
            _t3_news = _get_stock_news(ticker)
            _t3_price = _yf_fast_price(ticker)
            _comp_pct, _comp_secs = _app_compute_comprehensive_score(_t3_info, _t3_news, _t3_price)
            _comp_label, _ = _app_overall_label(_comp_pct)
            _app_render_verdict_banner(_comp_label, _comp_pct, len(_comp_secs))
            st.caption("Based on fundamentals, analyst consensus, ownership, and macro context.")
            _cs_names = list(_comp_secs.keys())
            _cs_cols = st.columns(len(_cs_names))
            for _ci, _cn in enumerate(_cs_names):
                _csc, _cmx, _ = _comp_secs[_cn]
                _cs_cols[_ci].metric(_cn, f"{_csc/_cmx*100:+.0f}%" if _cmx else "—")
            st.markdown("---")
            for _sn, (_csc, _cmx, _cbl) in _comp_secs.items():
                _cp = _csc / _cmx * 100 if _cmx else 0
                _ic = "✅" if _cp > 20 else ("🔴" if _cp < -20 else "⚪")
                with st.expander(f"{_ic} {_sn} — {_cp:+.0f}%", expanded=False):
                    for _b in _cbl:
                        st.markdown(f"- {_b}")
            _t3_macro = _app_compute_macro_context(_t3_info)
            if _t3_macro:
                st.markdown("#### 🌍 Macro Context")
                for _b in _t3_macro:
                    st.markdown(f"- {_b}")

        st.markdown("---")
        # ── SECTION 5: AI Fund Manager ────────────────────────────────────────────
        if True:  # scope block (replaces with t1_sub4:)
            _t4_hdr, _t4_clr_col = st.columns([4, 1])
            with _t4_hdr:
                st.markdown("#### 🧠 AI Fund Manager Analysis")
                st.caption("Powered by Claude Sonnet → Groq (Llama 3.3) → Gemini fallback — technicals + fundamentals + macro + news. Cached per ticker.")
            with _t4_clr_col:
                if st.button("🗑️ Clear AI Cache", key="t4_clear_cache", use_container_width=True):
                    _get_claude_analysis.clear()
                    _get_gemini_analysis_full.clear()
                    _get_groq_analysis.clear()
                    st.success("Cache cleared. Re-run Analysis to get a fresh response.")
                    st.rerun()

            _t4_df = _get_ta_data(ticker)
            _t4_info = _yf_info(ticker)
            _t4_price = _yf_fast_price(ticker)

            if not _t4_df.empty:
                _t4_last = _t4_df.dropna(subset=["Close"]).iloc[-1]
                _t4_rsi  = f"{_t4_last.get('RSI'):.1f}"  if pd.notna(_t4_last.get("RSI"))  else "N/A"
                _t4_macd = f"{_t4_last.get('MACD'):.3f}" if pd.notna(_t4_last.get("MACD")) else "N/A"
                _t4_ma50 = f"{curr}{_t4_last.get('MA50'):.2f}" if pd.notna(_t4_last.get("MA50")) else "N/A"
                _t4_ma200= f"{curr}{_t4_last.get('MA200'):.2f}" if pd.notna(_t4_last.get("MA200")) else "N/A"
                _t4_bbh  = f"{curr}{_t4_last.get('BB_High'):.2f}" if pd.notna(_t4_last.get("BB_High")) else "N/A"
                _t4_bbl  = f"{curr}{_t4_last.get('BB_Low'):.2f}"  if pd.notna(_t4_last.get("BB_Low"))  else "N/A"
                _t4_h52  = f"{curr}{_t4_df['High'].max():.2f}"
                _t4_l52  = f"{curr}{_t4_df['Low'].min():.2f}"
            else:
                _t4_rsi = _t4_macd = _t4_ma50 = _t4_ma200 = _t4_bbh = _t4_bbl = "N/A"
                _t4_h52 = _t4_l52 = "N/A"

            _t4_news_raw = _get_stock_news(ticker)
            _t4_signals  = _app_build_signals(_t4_df) if not _t4_df.empty else []
            _, _t4_comp  = _app_compute_comprehensive_score(_t4_info, _t4_news_raw, _t4_price)
            _t4_comp_pct, _ = _app_compute_comprehensive_score(_t4_info, _t4_news_raw, _t4_price)[0], None
            # Re-compute cleanly
            _t4_comp_score_pct, _ = _app_compute_comprehensive_score(_t4_info, _t4_news_raw, _t4_price)
            _t4_buy_n   = sum(1 for s in _t4_signals if s["Score"] > 0)
            _t4_sell_n  = sum(1 for s in _t4_signals if s["Score"] < 0)

            # Fetch next earnings date
            _t4_earn_str   = "Not available"
            _t4_earn_days  = 999
            _t4_earn_date  = None
            _manual_earn_key = f"manual_earn_{ticker}"

            # Apply manual override FIRST (before yfinance attempt)
            if st.session_state.get(_manual_earn_key):
                _ov = st.session_state[_manual_earn_key]
                _t4_earn_days = max(0, (_ov - datetime.date.today()).days)
                _t4_earn_str  = f"{_ov.strftime('%d %b %Y')} ({_t4_earn_days} days away) ✏️ manual"
            else:
                try:
                    _earn_ticker = yf.Ticker(ticker)
                    _earn_cal    = _earn_ticker.calendar
                    if _earn_cal is not None and not getattr(_earn_cal, 'empty', True):
                        _earn_vals = [v for v in _earn_cal.values.flatten()
                                      if hasattr(v, 'date') or (isinstance(v, str) and '-' in v)]
                        if _earn_vals:
                            _ed = _earn_vals[0]
                            if hasattr(_ed, 'date'): _ed = _ed.date()
                            elif isinstance(_ed, str): _ed = datetime.date.fromisoformat(_ed[:10])
                            _t4_earn_date = _ed
                            _t4_earn_days = max(0, (_ed - datetime.date.today()).days)
                            _t4_earn_str  = f"{_ed.strftime('%d %b %Y')} ({_t4_earn_days} days away)"
                except Exception:
                    pass

            # Fetch 2-month daily price history for AI context
            with st.spinner("Loading 2-month price history…"):
                _t4_price_hist = _fetch_daily_price_summary(ticker)

            _t4_args = (
                ticker, f"{curr}{_t4_price:.2f}",
                _t4_rsi, _t4_macd, _t4_ma50, _t4_ma200, _t4_bbh, _t4_bbl,
                _t4_h52, _t4_l52, _t4_buy_n, _t4_sell_n,
                f"{_t4_comp_score_pct:+.1f}%", curr, _t4_earn_str, _t4_price_hist,
            )

            # ── Earnings Date Card ────────────────────────────────────────────────
            st.markdown("#### 📅 Earnings Date & Strategy")
            if _t4_earn_days == 0:
                st.error(
                    f"🔴 **Earnings TODAY or JUST REPORTED** — Do not enter a new position. "
                    f"Wait 1-2 days for the dust to settle, then evaluate the post-earnings gap for a continuation or reversal trade."
                )
            elif _t4_earn_days <= 2:
                st.error(
                    f"🔴 **Earnings in {_t4_earn_days} day(s) ({_t4_earn_str})** — Too late for a pre-earnings run-up trade. "
                    f"Gap risk is too high to enter now. WAIT for post-earnings reaction instead."
                )
            elif _t4_earn_days <= 7:
                st.success(
                    f"🚀 **PRIME PRE-EARNINGS WINDOW — {_t4_earn_days} days to results ({_t4_earn_str})**\n\n"
                    f"This is the strongest period for a pre-earnings run-up trade. Stocks typically rally 3-7% "
                    f"in the week before results as traders position for a beat.\n\n"
                    f"**Strategy:** Enter now with momentum → **Exit the day BEFORE earnings** — do NOT hold through results. "
                    f"You are trading the anticipation, not the outcome. Set a tight stop 2% below entry."
                )
            elif _t4_earn_days <= 14:
                st.warning(
                    f"🟡 **Pre-earnings setup building — {_t4_earn_days} days to results ({_t4_earn_str})**\n\n"
                    f"Run-up typically accelerates in the final 5-7 days. You can enter now for a longer run, "
                    f"but plan your exit: **sell before earnings, not after**. "
                    f"If you miss the exit, you're gambling on the outcome."
                )
            elif _t4_earn_days < 999:
                st.info(
                    f"✅ **Safe window — Next earnings: {_t4_earn_str} ({_t4_earn_days} days away)**\n\n"
                    f"No earnings risk this week. Trade purely on technicals and momentum. "
                    f"Start watching again when you're within 14 days of results for a possible pre-earnings play."
                )
            else:
                st.warning(f"📅 **Earnings date not found automatically for {ticker}.**")

            # Manual earnings date override panel
            with st.expander("📅 Set earnings date manually (or search online)", expanded=(_t4_earn_days == 999)):
                _mc1, _mc2 = st.columns([2, 1])
                with _mc1:
                    _manual_date = st.date_input(
                        "Select next earnings date",
                        value=st.session_state.get(_manual_earn_key),
                        min_value=datetime.date.today(),
                        key=f"earn_datepick_{ticker}",
                        help="Pick the date from the company's investor relations page or earningswhispers.com",
                    )
                with _mc2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("✅ Apply this date", key=f"apply_earn_{ticker}", use_container_width=True):
                        st.session_state[_manual_earn_key] = _manual_date
                        st.rerun()

                # Online search for earnings date via AI
                if st.button(f"🔍 Ask AI to find {ticker} earnings date", key=f"search_earn_{ticker}", use_container_width=True):
                    with st.spinner("Searching for earnings date…"):
                        try:
                            _sk = os.getenv("GROQ_API_KEY","").strip() or os.getenv("GEMINI_API_KEY","").strip()
                            if _sk and os.getenv("GROQ_API_KEY","").strip():
                                from groq import Groq as _GQ
                                _gq = _GQ(api_key=os.getenv("GROQ_API_KEY"))
                                _er = _gq.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role":"user","content":
                                        f"What is the next quarterly earnings date for {ticker} stock? "
                                        f"Use your training data. Reply with ONLY the date in format DD MMM YYYY "
                                        f"(e.g. 15 Aug 2025), or 'Unknown' if not sure."}],
                                    temperature=0, max_tokens=50,
                                )
                                _earn_ai = _er.choices[0].message.content.strip()
                                st.info(f"AI says next earnings: **{_earn_ai}**. If correct, enter it in the date picker above and apply.")
                            else:
                                st.warning("No AI key available. Add GROQ_API_KEY to .env.")
                        except Exception as _se:
                            st.error(f"Search failed: {_se}")

            # Force-reload .env so new keys are always current
            try:
                from dotenv import load_dotenv as _ldenv
                _ldenv(override=True)
            except Exception:
                pass

            _claude_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            _groq_key   = os.getenv("GROQ_API_KEY", "").strip()
            _gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            _ai_provider = "Claude Sonnet"
            _ai_raw      = None
            _ai_handled  = False
            _ai_text     = ""

            def _is_error(r):
                return not r or r.startswith("__ERROR__") or r in (
                    "__NO_KEY__", "__NO_GROQ_KEY__", "__NO_CLAUDE_KEY__")

            def _is_quota_error(r):
                if not r or not r.startswith("__ERROR__"): return False
                _rl = r.lower()
                return any(k in _rl for k in ("quota", "429", "exhausted", "resource_exhausted",
                                               "rate_limit", "rate limit", "daily", "tokens per day",
                                               "requests per day", "limit reached", "credit",
                                               "billing", "overloaded_error"))

            def _is_overload_error(r):
                if not r or not r.startswith("__ERROR__"): return False
                _rl = r.lower()
                if _is_quota_error(r): return False
                return any(k in _rl for k in ("503", "unavailable", "high demand", "overloaded", "502", "529"))

            # ── Step 1: Try Claude first (best quality) ───────────────────────────
            if _claude_key:
                with st.spinner("Analysing with Claude Sonnet (best quality)…"):
                    _ai_raw = _get_claude_analysis(*_t4_args)
                if _ai_raw and not _ai_raw.startswith(("__", "===", "*", "#")) and len(_ai_raw) < 80:
                    _get_claude_analysis.clear()
                    _ai_raw = "__ERROR__stale"

            # ── Step 2: Groq fallback ─────────────────────────────────────────────
            if _is_error(_ai_raw):
                if _groq_key:
                    _ai_provider = "Groq / Llama 3.3"
                    with st.spinner("Falling back to Groq (Llama 3.3 70B)…" if _claude_key else "Analysing with Groq…"):
                        _ai_raw = _get_groq_analysis(*_t4_args)
                    if _ai_raw and not _ai_raw.startswith(("__", "===", "*", "#")) and len(_ai_raw) < 80:
                        _get_groq_analysis.clear()
                        _ai_raw = "__ERROR__stale"

            # ── Step 3: Gemini last resort ────────────────────────────────────────
            if _is_error(_ai_raw):
                if _gemini_key:
                    _ai_provider = "Gemini"
                    with st.spinner("Falling back to Gemini…"):
                        _ai_raw_gemini = _get_gemini_analysis_full(*_t4_args)
                    if _ai_raw_gemini and not _ai_raw_gemini.startswith(("__", "===", "*", "#")) and len(_ai_raw_gemini) < 80:
                        _get_gemini_analysis_full.clear()
                        _ai_raw_gemini = "__ERROR__stale"
                    _ai_raw = _ai_raw_gemini

            # ── Display errors ────────────────────────────────────────────────────
            _ai_summary = None  # ensure always defined
            if not _claude_key and not _groq_key and not _gemini_key:
                st.warning("No AI keys configured. Add `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` to `.env`.")
                _ai_handled = True
            elif _is_quota_error(_ai_raw):
                _quota_who = "Claude" if "claude" in (_ai_raw or "").lower() or "credit" in (_ai_raw or "").lower() else "Groq/Gemini"
                st.warning(f"**Daily quota reached for {_quota_who}.** Groq free tier = 100 requests/day, Gemini free = 1500/day. Try again tomorrow or upgrade to a paid plan.")
                _ai_handled = True
            elif _is_overload_error(_ai_raw):
                st.warning(
                    "**Gemini is overloaded (503).** Groq cache may also be stale. "
                    "Click **🔁 Retry** below — it clears both caches and retries Groq first."
                )
                if st.button("🔁 Retry AI Analysis", key="t4_retry_overload"):
                    _get_claude_analysis.clear()
                    _get_gemini_analysis_full.clear()
                    _get_groq_analysis.clear()
                    st.rerun()
                _ai_handled = True
            elif _ai_raw and _ai_raw.startswith("__ERROR__"):
                _err_msg = _ai_raw[9:]
                if "stale" in _err_msg:
                    st.info("AI cache was stale — click **🔁 Re-run Analysis** to refresh.")
                else:
                    with st.expander("⚠️ AI error details (click to see)", expanded=True):
                        st.warning(f"**Provider:** {_ai_provider}")
                        st.code(_err_msg[:400], language=None)
                        st.caption("If you see '429' or 'quota' → daily limit hit. If '503' → retry in a minute. If 'stale' → clear cache.")
                _ai_handled = True
            elif not _ai_raw or _ai_raw in ("__NO_KEY__", "__NO_GROQ_KEY__"):
                st.warning("No AI keys configured. Add `GROQ_API_KEY` or `GEMINI_API_KEY` to `.env`.")
                _ai_handled = True
            else:
                _ai_summary, _ai_text = _app_parse_gemini_response(_ai_raw)
                if not _ai_summary and (len(_ai_raw) < 200 or not any(
                    kw in _ai_raw for kw in ("VERDICT", "TECHNICAL", "TARGET", "===JSON")
                )):
                    # Stale/unrecognised cached string — clear and prompt refresh
                    _get_gemini_analysis_full.clear()
                    st.info("AI cache was stale and has been cleared. Click **🔁 Re-run Analysis** above.")
                    _ai_summary = None
                def _fmt_ai_val(v):
                    """Return '—' for zero/placeholder values from AI JSON."""
                    if not v or v in ("0", "0.00", "$0.00", "₹0.00", "N/A", "n/a",
                                      "[exact price]", "Not applicable", "—"):
                        return "—"
                    return str(v)

                if _ai_summary:
                    _ai_v_raw = _ai_summary.get("verdict", "").lower()
                    _ai_fg, _ai_bg = _APP_VERDICT_COLORS.get(_ai_v_raw, ("#ffcc00","#332900"))
                    _is_avoid = _ai_v_raw in ("avoid", "strong avoid")
                    _is_buy   = _ai_v_raw in ("buy", "strong buy")

                    # ── Fill the top placeholder with the Command Center header ───
                    with _cc_ai_placeholder.container():
                        st.markdown("#### 🧠 AI Fund Manager — Decision")
                        _rr_val = _ai_summary.get("rr_ratio","")
                        try: _rr_num = float(_rr_val)
                        except Exception: _rr_num = 0
                        _rr_clr = "#00c853" if _rr_num >= 2 else "#ff9800" if _rr_num >= 1 else "#ff5252"
                        _entry_v  = _fmt_ai_val(_ai_summary.get("entry"))
                        _stop_v   = _fmt_ai_val(_ai_summary.get("stop_loss"))
                        _t1_v     = _fmt_ai_val(_ai_summary.get("target_1m") or _ai_summary.get("target_3d"))
                        _t3_v     = _fmt_ai_val(_ai_summary.get("target_3m") or _ai_summary.get("target_5d"))
                        # Fallback: compute from live price when AI returned blanks
                        _cc_live  = _yf_fast_price(ticker) or current_price
                        if _entry_v == "—" and _cc_live > 0:
                            _entry_v = f"{_cc_live:.2f}"
                        if _stop_v == "—" and _cc_live > 0:
                            _stop_v  = f"{_cc_live * (0.95 if not _is_avoid else 0.97):.2f}"
                        if _t1_v == "—" and _cc_live > 0:
                            _t1_v    = f"{_cc_live * (1.07 if _is_buy else 0.95 if _is_avoid else 1.03):.2f}"
                        if _t3_v == "—" and _cc_live > 0:
                            _t3_v    = f"{_cc_live * (1.15 if _is_buy else 0.90 if _is_avoid else 1.06):.2f}"
                        st.markdown(
                            f"""<div style="background:{_ai_bg};border:2px solid {_ai_fg};border-radius:12px;
                                padding:18px 24px;margin-bottom:12px">
                              <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px">
                                <span style="font-size:2.2rem;font-weight:900;color:{_ai_fg};letter-spacing:3px">
                                  {_ai_summary.get("verdict","—").upper()}</span>
                                <span style="color:#ccc;font-size:1rem">Conviction:
                                  <b style="color:{_ai_fg}">{_ai_summary.get("conviction","—")}</b></span>
                                <span style="margin-left:auto;font-size:0.85rem;color:{_rr_clr};font-weight:700">
                                  R/R {_rr_val} {"✅" if _rr_num >= 2 else "⚠️"}</span>
                              </div>
                              <div style="display:flex;gap:32px;flex-wrap:wrap">
                                <div><span style="color:#888;font-size:0.75rem">{"AVOID ENTRY" if _is_avoid else "ENTRY"}</span>
                                  <div style="color:#fff;font-size:1.1rem;font-weight:700">{curr if _entry_v != "—" else ""}{_entry_v}</div></div>
                                <div><span style="color:#888;font-size:0.75rem">STOP LOSS</span>
                                  <div style="color:#ff5252;font-size:1.1rem;font-weight:700">{curr if _stop_v != "—" else ""}{_stop_v}</div></div>
                                <div><span style="color:#888;font-size:0.75rem">{"DOWNSIDE 5-10D" if _is_avoid else "TARGET 5-10D"}</span>
                                  <div style="color:{"#ff9800" if _is_avoid else "#00c853"};font-size:1.1rem;font-weight:700">{curr if _t1_v != "—" else ""}{_t1_v}</div></div>
                                <div><span style="color:#888;font-size:0.75rem">{"DOWNSIDE 30D" if _is_avoid else "TARGET 30D"}</span>
                                  <div style="color:{"#ff5252" if _is_avoid else "#00c853"};font-size:1.1rem;font-weight:700">{curr if _t3_v != "—" else ""}{_t3_v}</div></div>
                              </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        # Checklist-style summary from technicals (if available)
                        _cc_tech_col, _cc_fund_col = st.columns(2)
                        with _cc_tech_col:
                            st.markdown("**Technicals at a glance:**")
                            try:
                                _cc_ta = _get_ta_data(ticker)
                                if not _cc_ta.empty:
                                    _cc_last = _cc_ta.dropna(subset=["Close"]).iloc[-1]
                                    _cc_rsi  = float(_cc_ta["RSI"].dropna().iloc[-1]) if "RSI" in _cc_ta else None
                                    _cc_macd = _app_safe(_cc_last.get("MACD"))
                                    _cc_msig = _app_safe(_cc_last.get("MACD_Signal"))
                                    _cc_e20  = _app_safe(_cc_last.get("EMA20"))
                                    _cc_e50  = _app_safe(_cc_last.get("EMA50"))
                                    _cc_px   = float(_cc_last["Close"])
                                    if _cc_rsi:  st.markdown(f"{'✅' if _cc_rsi < 70 else '⚠️'} RSI: **{_cc_rsi:.0f}** {'(OK)' if _cc_rsi<70 else '(Overbought)'}")
                                    if _cc_macd and _cc_msig: st.markdown(f"{'✅' if _cc_macd>_cc_msig else '❌'} MACD: **{'Bullish' if _cc_macd>_cc_msig else 'Bearish'}**")
                                    if _cc_e20 and _cc_e50:   st.markdown(f"{'✅' if _cc_e20>_cc_e50 else '❌'} EMA20 vs EMA50: **{'Uptrend' if _cc_e20>_cc_e50 else 'Downtrend'}**")
                                    if _cc_e20:               st.markdown(f"{'✅' if _cc_px>_cc_e20 else '❌'} Price vs EMA20: **{'Above' if _cc_px>_cc_e20 else 'Below'}**")
                            except Exception:
                                st.caption("Technicals unavailable")
                        with _cc_fund_col:
                            st.markdown("**Fundamentals at a glance:**")
                            try:
                                _cc_info = _yf_info(ticker)
                                _cc_pe   = _app_safe(_cc_info.get("trailingPE"))
                                _cc_rg   = _app_safe(_cc_info.get("revenueGrowth"))
                                _cc_mg   = _app_safe(_cc_info.get("profitMargins"))
                                _cc_beta = _app_safe(_cc_info.get("beta"))
                                if _cc_pe:   st.markdown(f"{'✅' if _cc_pe<35 else '⚠️'} P/E: **{_cc_pe:.1f}** {'(reasonable)' if _cc_pe<35 else '(stretched)'}")
                                if _cc_rg:   st.markdown(f"{'✅' if _cc_rg>0.1 else '⚠️'} Revenue Growth: **{_cc_rg*100:.1f}%** YoY")
                                if _cc_mg:   st.markdown(f"{'✅' if _cc_mg>0.1 else '⚠️'} Net Margin: **{_cc_mg*100:.1f}%**")
                                if _cc_beta: st.markdown(f"{'✅' if _cc_beta<1.5 else '⚠️'} Beta: **{_cc_beta:.2f}** {'(moderate risk)' if _cc_beta<1.5 else '(high volatility)'}")
                            except Exception:
                                st.caption("Fundamentals unavailable")

                    # Full AI analysis text in expander
                    with st.expander("📄 Full AI Analysis", expanded=False):
                        st.markdown(_ai_text)
                else:
                    with _cc_ai_placeholder.container():
                        st.info("Run analysis to see the AI Fund Manager decision.")
                    st.markdown(_ai_text)

            # ── Market-Adjusted Decision ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🎯 Market-Adjusted Decision")
            st.caption("Combines this stock's AI verdict with live market risk conditions.")

            with st.spinner("Computing market risk overlay…"):
                _mrd = _compute_market_risk_score()

            _mr_score = _mrd.get("score", 20)
            _mr_level = _mrd.get("level", "LOW")
            _mr_vix   = _mrd.get("vix")
            _mr_sp3m  = _mrd.get("sp_3m")

            # Market risk colour
            _mr_col = {"LOW":"#00c853","MODERATE":"#ff9800","HIGH":"#ff5252","EXTREME":"#d50000"}.get(_mr_level,"#888")

            # Get AI verdict from the analysis that ran above
            _stock_verdict = ""
            if _ai_summary:
                _stock_verdict = _ai_summary.get("verdict","").lower()
            elif _ai_text:
                import re as _rvre
                _vm = _rvre.search(r"VERDICT[^:]*:\s*([A-Za-z ]+)", _ai_text)
                if _vm: _stock_verdict = _vm.group(1).strip().lower()

            # Determine adjusted recommendation
            _is_buy   = any(k in _stock_verdict for k in ("strong buy","buy"))
            _is_watch = "watch" in _stock_verdict
            _is_avoid = any(k in _stock_verdict for k in ("avoid","strong avoid"))

            # Position size matrix
            _pos_matrix = {
                # (verdict_bucket, market_level): (size_pct, action, stop_adj, entry_adj)
                ("buy",   "LOW"):      (100, "BUY FULL POSITION",    1.0, 0.0),
                ("buy",   "MODERATE"): (65,  "BUY — REDUCED SIZE",   0.8, 0.02),
                ("buy",   "HIGH"):     (35,  "SMALL POSITION ONLY",  0.6, 0.04),
                ("buy",   "EXTREME"):  (0,   "HOLD OFF — MARKET RISK TOO HIGH", 0.0, 0.06),
                ("watch", "LOW"):      (50,  "PARTIAL ENTRY — WAIT FOR SETUP",  1.0, 0.01),
                ("watch", "MODERATE"): (25,  "WAIT FOR BETTER ENTRY", 0.8, 0.03),
                ("watch", "HIGH"):     (0,   "WAIT — BOTH STOCK AND MARKET WEAK", 0.0, 0.05),
                ("watch", "EXTREME"):  (0,   "AVOID — REDUCE EXISTING POSITIONS", 0.0, 0.07),
                ("avoid", "LOW"):      (0,   "SKIP THIS STOCK", 0.0, 0.0),
                ("avoid", "MODERATE"): (0,   "SKIP THIS STOCK", 0.0, 0.0),
                ("avoid", "HIGH"):     (0,   "SKIP — REVIEW EXISTING HOLDINGS", 0.0, 0.0),
                ("avoid", "EXTREME"):  (0,   "AVOID ALL — CONSIDER CASH", 0.0, 0.0),
            }

            _vbkt = "buy" if _is_buy else "watch" if _is_watch else "avoid"
            _pos_size, _adj_action, _stop_mult, _entry_delay = _pos_matrix.get(
                (_vbkt, _mr_level), (0, "WAIT FOR CLARITY", 0.0, 0.03))

            # Entry price adjustment
            def _parse_price_field(val, fallback):
                """Parse AI price field — handles ranges like '2050-2080' by taking the midpoint."""
                try:
                    s = str(val).replace(curr, "").replace(",", "").strip()
                    if "-" in s:
                        parts = [p.strip() for p in s.split("-") if p.strip()]
                        nums = []
                        for p in parts:
                            try: nums.append(float(p))
                            except ValueError: pass
                        return sum(nums) / len(nums) if nums else fallback
                    return float(s) if s else fallback
                except (ValueError, TypeError):
                    return fallback
            _base_entry = _parse_price_field(_ai_summary.get("entry", "0"), _t4_price) if _ai_summary else _t4_price
            if _base_entry <= 0: _base_entry = _t4_price
            _adj_entry = round(_base_entry * (1 - _entry_delay), 2)

            # Stop loss adjustment
            _base_stop = _parse_price_field(_ai_summary.get("stop_loss", "0"), 0) if _ai_summary else 0
            if _base_stop <= 0: _base_stop = round(_base_entry * 0.96, 2)
            _stop_gap  = _base_entry - _base_stop
            _adj_stop  = round(_adj_entry - _stop_gap * (2 - _stop_mult) if _stop_mult > 0 else _adj_entry * 0.97, 2)

            # Display
            _adj_color = "#00c853" if _pos_size >= 65 else "#ff9800" if _pos_size >= 25 else "#ff5252"

            # Row 1: market context + verdict
            _dc1, _dc2, _dc3 = st.columns(3)
            with _dc1:
                with st.container(border=True):
                    st.markdown(f"**Market Risk**")
                    st.markdown(f"<h2 style='color:{_mr_col};margin:0'>{_mr_level}</h2>", unsafe_allow_html=True)
                    st.caption(f"Score {_mr_score}/100 · VIX {f'{_mr_vix:.0f}' if _mr_vix else '—'} · S&P {f'{_mr_sp3m:+.1f}' if _mr_sp3m else '—'}%")
            with _dc2:
                with st.container(border=True):
                    st.markdown(f"**Stock Signal**")
                    st.markdown(f"<h2 style='margin:0'>{(_stock_verdict or 'pending').upper()}</h2>", unsafe_allow_html=True)
                    st.caption(f"AI verdict for {ticker}")
            with _dc3:
                with st.container(border=True):
                    st.markdown(f"**Position Size**")
                    st.markdown(f"<h2 style='color:{_adj_color};margin:0'>{_pos_size}%</h2>", unsafe_allow_html=True)
                    st.caption("of your planned allocation")

            # Row 2: adjusted prices + position sizing
            _acct   = st.session_state.get("acct_size", 10000)
            _riskp  = st.session_state.get("risk_pct", 1.0)
            _max_loss = _acct * _riskp / 100
            _risk_per_share = max(_adj_entry - _adj_stop, 0.01)
            _raw_shares = _max_loss / _risk_per_share
            _pos_shares = int(_raw_shares * (_pos_size / 100)) if _pos_size > 0 else 0
            _pos_value  = round(_pos_shares * _adj_entry, 2)
            _pct_of_acct = _pos_value / _acct * 100 if _acct > 0 else 0

            _dp1, _dp2, _dp3, _dp4 = st.columns(4)
            _dp1.metric("Adjusted Entry",
                        f"{curr}{_adj_entry:.2f}",
                        delta=f"{(_adj_entry-_base_entry)/_base_entry*100:+.1f}% vs AI entry" if _entry_delay>0 else "At AI entry",
                        delta_color="off")
            _dp2.metric("Adjusted Stop",
                        f"{curr}{_adj_stop:.2f}",
                        delta=f"{(_adj_stop-_adj_entry)/_adj_entry*100:.1f}%",
                        delta_color="inverse")
            _dp3.metric("5-10 Day Target",  _fmt_ai_val(_ai_summary.get("target_1m")) if _ai_summary else "—")
            _dp4.metric("30-Day Target", _fmt_ai_val(_ai_summary.get("target_3m")) if _ai_summary else "—")

            # Position sizing row
            with st.container(border=True):
                st.markdown(f"**💰 Position Sizing  ·  Account ${_acct:,} · Max risk {_riskp:.1f}% (${_max_loss:,.0f} max loss)**")
                _ps1, _ps2, _ps3, _ps4 = st.columns(4)
                _ps1.metric("Shares to Buy",    f"{_pos_shares:,}" if _pos_shares > 0 else "0 — skip")
                _ps2.metric("Total Investment", f"${_pos_value:,.0f}" if _pos_value > 0 else "—",
                            delta=f"{_pct_of_acct:.1f}% of account", delta_color="off")
                _ps3.metric("Max $ Loss",       f"${_pos_shares * _risk_per_share:,.0f}" if _pos_shares > 0 else "—",
                            delta=f"{_riskp * _pos_size / 100:.2f}% of account", delta_color="inverse")
                _ps4.metric("Risk per Share",   f"{curr}{_risk_per_share:.2f}",
                            delta=f"{_risk_per_share/_adj_entry*100:.1f}% stop distance", delta_color="off")

            # Final action banner
            if _pos_size >= 65:
                st.success(f"✅ **{_adj_action}** — Market supports this trade. Enter near {curr}{_adj_entry:.2f} with stop at {curr}{_adj_stop:.2f}.")
            elif _pos_size >= 25:
                st.warning(f"⚠️ **{_adj_action}** — Market is {_mr_level}. Reduce size to {_pos_size}% of planned. Wait for pullback to {curr}{_adj_entry:.2f}.")
            else:
                st.error(f"🛑 **{_adj_action}** — Market risk ({_mr_level}) overrides the stock signal. Protect capital first.")

            # Rationale
            with st.expander("📖 Why this recommendation?"):
                st.markdown(f"""
    **Stock verdict:** {(_stock_verdict or 'N/A').upper()} | **Market risk:** {_mr_level} ({_mr_score}/100)

    **Matrix logic:**
    - In a **{_mr_level}** risk market, even a {(_stock_verdict or 'BUY').upper()} only justifies **{_pos_size}% position size**
    - Entry is adjusted {_entry_delay*100:.0f}% below AI's base entry to account for market volatility
    - Stop is tightened proportionally — in a {_mr_level} market there is less room for error

    **Market conditions right now:**
    - VIX: {f'{_mr_vix:.1f}' if _mr_vix else 'N/A'} {'(elevated fear — widen stops)' if _mr_vix and _mr_vix>25 else '(calm — normal stops)'}
    - S&P 500 3-month: {f'{_mr_sp3m:+.1f}%' if _mr_sp3m else 'N/A'} {'(downtrend — extra caution)' if _mr_sp3m and _mr_sp3m<-3 else '(uptrend — supportive)'}
    """)

            # ── Chat with AI Agent ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💬 Chat with AI Agent")
            st.caption(f"Ask anything about {ticker} — entry timing, risk factors, comparison with peers, position sizing, etc.")

            _chat_key  = f"chat_history_{ticker}"
            _ctx_key   = f"chat_ctx_{ticker}"
            if _chat_key not in st.session_state:
                st.session_state[_chat_key] = []

            # Build stock context once (reuse AI analysis if available)
            _chat_ctx = (
                f"Stock: {ticker} | Price: {curr}{_t4_price:.2f} | RSI: {_t4_rsi} | MACD: {_t4_macd} | "
                f"52W High: {_t4_h52} | 52W Low: {_t4_l52} | "
                f"Bullish signals: {_t4_buy_n} | Bearish signals: {_t4_sell_n} | "
                f"Composite score: {_t4_comp_score_pct:+.1f}% | Earnings: {_t4_earn_str}\n"
            )
            if _ai_text and len(_ai_text) > 100:
                _chat_ctx += f"\nAI analysis already generated:\n{_ai_text[:2000]}"

            # Render existing messages
            for _msg in st.session_state[_chat_key]:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            # Chat input
            _chat_input = st.chat_input(f"Ask about {ticker}…", key=f"chat_input_{ticker}")
            if _chat_input:
                # Show user message
                st.session_state[_chat_key].append({"role": "user", "content": _chat_input})
                with st.chat_message("user"):
                    st.markdown(_chat_input)

                # Build messages for API
                _system_msg = (
                    f"You are an expert stock analyst and portfolio manager specialising in weekly swing trades. "
                    f"The user is analysing {ticker}. Here is the context:\n{_chat_ctx}\n\n"
                    f"Answer concisely and specifically. Use {curr} for prices. "
                    f"Always frame advice for a short-term swing trade (1 day to 1 month hold). Be direct — no disclaimers."
                )
                _api_messages = [{"role": "system", "content": _system_msg}]
                for _m in st.session_state[_chat_key]:
                    _api_messages.append({"role": _m["role"], "content": _m["content"]})

                # Stream response via Groq
                with st.chat_message("assistant"):
                    _reply_placeholder = st.empty()
                    _reply_text = ""
                    _chat_ok = False
                    try:
                        _groq_k = os.getenv("GROQ_API_KEY", "").strip()
                        if _groq_k:
                            from groq import Groq as _GC
                            _gc = _GC(api_key=_groq_k)
                            _stream = _gc.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=_api_messages,
                                temperature=0.4,
                                max_tokens=600,
                                stream=True,
                            )
                            for _chunk in _stream:
                                _delta = _chunk.choices[0].delta.content or ""
                                _reply_text += _delta
                                _reply_placeholder.markdown(_reply_text + "▌")
                            _reply_placeholder.markdown(_reply_text)
                            _chat_ok = True
                    except Exception as _ce:
                        pass

                    if not _chat_ok:
                        # Fallback: non-streaming Gemini
                        try:
                            _gem_k = os.getenv("GEMINI_API_KEY", "").strip()
                            if _gem_k:
                                from google import genai as _gensdk
                                _gclient = _gensdk.Client(api_key=_gem_k)
                                _gresp = _gclient.models.generate_content(
                                    model="gemini-2.0-flash",
                                    contents=_system_msg + "\n\nUser: " + _chat_input,
                                )
                                _reply_text = _gresp.text
                                _reply_placeholder.markdown(_reply_text)
                                _chat_ok = True
                        except Exception:
                            pass

                    if not _chat_ok:
                        _reply_text = "AI unavailable right now. Check your API keys or try again."
                        _reply_placeholder.warning(_reply_text)

                st.session_state[_chat_key].append({"role": "assistant", "content": _reply_text})

            # Clear chat button
            if st.session_state[_chat_key]:
                if st.button("🗑️ Clear chat", key=f"clear_chat_{ticker}"):
                    st.session_state[_chat_key] = []
                    st.rerun()

            # News
            st.markdown("---")
            st.markdown("#### 📰 Recent News")
            _t4_news = _get_stock_news(ticker)
            if _t4_news:
                for _item in _t4_news[:8]:
                    _ntitle, _nlink, _npub, _nage, _nsum = _app_parse_news_item(_item)
                    if not _ntitle:
                        continue
                    st.markdown(
                        f"**[{_ntitle}]({_nlink})**  \n"
                        f"<span style='color:#888;font-size:0.8rem'>{_npub} &nbsp;·&nbsp; {_nage}</span>",
                        unsafe_allow_html=True,
                    )
                    if _nsum:
                        st.caption(_nsum[:180] + ("…" if len(_nsum) > 180 else ""))
                    st.markdown("---")
            else:
                st.info("No recent news found for this ticker.")


with main_tab2:  # ← replacement block starts here
    for _tab2_once in [True]:
        # ── Tab 2 sub-nav ─────────────────────────────────────────────────────────
        _fs_t1, _fs_t2, _fs_t3, _fs_t4, _fs_t5, _fs_t6, _fs_t7 = st.tabs([
            "🎯 Quick Scan", "⚙️ Advanced Screener", "🌱 Early Stage",
            "🎯 Catalyst Scanner", "🚀 Breakout Scanner",
            "🌪️ Short Squeeze Radar", "📝 Paper Trading",
        ])

        # ── QUICK SCAN ──────────────────────────────────────────────────────────
        with _fs_t1:
            st.markdown("## 🎯 Quick Scan — Find the Right Stock Fast")
            st.caption(
                "Choose your goal. We run the scan and show you the highest-conviction "
                "candidates. No configuration required."
            )

            # ── Goal selector ──────────────────────────────────────────────────
            qs_goal = st.radio(
                "What are you trying to achieve?",
                [
                    "🏆 Maximum Conviction — All models agree, highest probability of gains",
                    "📈 Momentum — Stocks already moving up fast",
                    "💎 Undervalued Quality — Strong fundamentals, price lagging",
                    "🛡️ Safe Compounder — Low risk, steady 15-20% / year",
                ],
                key="qs_goal",
                horizontal=False,
            )

            qs_col1, qs_col2, qs_col3 = st.columns(3)
            with qs_col1:
                qs_markets = st.multiselect(
                    "Markets",
                    ["🇮🇳 NSE India", "🇺🇸 US Markets"],
                    default=["🇺🇸 US Markets"],
                    key="qs_markets",
                )
            with qs_col2:
                qs_horizon = st.radio(
                    "Horizon",
                    ["1M", "3M", "6M", "1Y"],
                    index=2,
                    horizontal=True,
                    key="qs_horizon",
                )
            with qs_col3:
                qs_smallcap = st.checkbox("Include US Small Caps", value=False, key="qs_sc")

            # Maximum Conviction explanation
            if "Maximum Conviction" in qs_goal:
                st.markdown(
                    """<div style="background:#0a1a0a;border:1px solid #00c853;border-radius:8px;
                    padding:12px 16px;margin:10px 0">
                    <span style="color:#00c853;font-weight:700;font-size:0.95rem">
                    🏆 Maximum Conviction Mode</span><br>
                    <span style="color:#aaa;font-size:0.83rem">
                    Filters for stocks where <strong>all 7 independent signals agree</strong>:
                    technical momentum + strong fundamentals + positive Monte Carlo probability +
                    positive 3M &amp; 6M returns + analyst upgrades + low downside risk.<br><br>
                    ⚠️ <strong>Honest note:</strong> No stock has 100% probability of gains.
                    Even maximum-conviction setups fail ~20-25% of the time — markets are
                    unpredictable. This filter maximises probability, not guarantees it.
                    Always use stop-losses and size positions at 5-10% of portfolio max.
                    </span></div>""",
                    unsafe_allow_html=True,
                )

            if st.button("🔍 Find Stocks", key="qs_run_btn", type="primary"):
                # Map goal → screener config
                if "Maximum Conviction" in qs_goal:
                    _qs_risk   = "Moderate"
                    _qs_bias   = "Both"
                    _qs_minsc  = 72          # high minimum composite score
                    _qs_label  = "Maximum Conviction"
                elif "Momentum" in qs_goal:
                    _qs_risk   = "Aggressive"
                    _qs_bias   = "Momentum Only"
                    _qs_minsc  = 60
                    _qs_label  = "Momentum"
                elif "Undervalued" in qs_goal:
                    _qs_risk   = "Moderate"
                    _qs_bias   = "Contrarian Only"
                    _qs_minsc  = 58
                    _qs_label  = "Undervalued Quality"
                else:
                    _qs_risk   = "Conservative"
                    _qs_bias   = "Both"
                    _qs_minsc  = 62
                    _qs_label  = "Safe Compounder"

                qs_include_in   = "🇮🇳 NSE India"  in qs_markets
                qs_include_us   = "🇺🇸 US Markets" in qs_markets

                _qs_syms = []
                if qs_include_in:
                    _qs_syms.extend(INDIA_UNIVERSE[:180])
                if qs_include_us:
                    _qs_base = build_universe_us()
                    _random.shuffle(_qs_base)
                    _qs_syms.extend(_qs_base[:250])
                    if qs_smallcap:
                        _sc = build_universe_us_smallcap()
                        _random.shuffle(_sc)
                        _qs_syms.extend(_sc[:100])
                _qs_syms = list(dict.fromkeys(_qs_syms))  # deduplicate

                with st.spinner(f"Scanning {len(_qs_syms)} stocks for {_qs_label} setups…"):
                    _qs_fast = {}
                    with ThreadPoolExecutor(max_workers=20) as _ex:
                        _futs = {_ex.submit(fetch_stock_features, s): s for s in _qs_syms}
                        for _fut in as_completed(_futs, timeout=180):
                            try:
                                _res = _fut.result()
                                if _res:
                                    _qs_fast[_res["symbol"]] = _res
                            except Exception:
                                pass

                if _qs_fast:
                    _qs_ranked = normalise_universe(_qs_fast)
                    _qs_scored = []
                    for _sym, _ff in _qs_fast.items():
                        try:
                            enrich_with_fundamentals(_ff)
                            _bkt = classify_bucket(_ff, _qs_ranked)
                            _sc, _ = score_stock(_ff, _qs_ranked, _bkt, _qs_risk, qs_horizon.replace("M"," Months").replace("Y"," Year"))
                            _qs_scored.append((_sym, _ff, _sc))
                        except Exception:
                            pass
                    _qs_scored.sort(key=lambda x: x[2], reverse=True)

                    # Conviction filter
                    def _passes_conviction(ff, score, label):
                        if label != "Maximum Conviction":
                            return score >= _qs_minsc
                        signals = []
                        signals.append(score >= 72)
                        signals.append((ff.get("revenue_growth") or 0) > 0.08)
                        signals.append((ff.get("profit_margins") or 0) > 0 or (ff.get("eps_growth") or 0) > 0.15)
                        m3 = ff.get("mom_3m") or 0
                        m6 = ff.get("mom_6m") or 0
                        signals.append(m3 > 0 and m6 > 0)
                        rsi_v = ff.get("rsi") or 50
                        signals.append(45 <= rsi_v <= 76)
                        beta_v = ff.get("beta") or 1
                        signals.append(beta_v < 2.5)
                        rec_v = (ff.get("recommendation") or "hold").lower()
                        signals.append(rec_v in ["buy","strong buy","strongbuy"])
                        ff["_conv_signals"] = signals
                        ff["_conv_score"]   = sum(signals)
                        return sum(signals) >= 5  # at least 5/7 signals agree

                    _qs_filtered = [
                        (sym, ff, sc) for sym, ff, sc in _qs_scored
                        if _passes_conviction(ff, sc, _qs_label)
                    ][:20]

                    st.session_state["qs_results"] = _qs_filtered
                    st.session_state["qs_label"]   = _qs_label
                else:
                    st.error("No data returned. Check your internet connection.")

            # ── Display Quick Scan results ──────────────────────────────────────
            if "qs_results" in st.session_state and st.session_state["qs_results"]:
                qs_label   = st.session_state.get("qs_label", "")
                qs_results = st.session_state["qs_results"]

                is_mc_mode = qs_label == "Maximum Conviction"
                st.markdown(f"#### {'🏆' if is_mc_mode else '📊'} {qs_label} — Top {len(qs_results)} Stocks")

                if is_mc_mode:
                    st.success(
                        "These stocks passed **all major signal filters**. "
                        "The number in parentheses (e.g. 7/7) shows how many independent "
                        "models/signals agree it's a buy. Higher = more conviction."
                    )

                for rank_i, (sym, ff, sc) in enumerate(qs_results, 1):
                    curr_px   = ff.get("price") or ff.get("current_price") or 0
                    name      = ff.get("name", sym)[:32]
                    rev_g     = ff.get("revenue_growth") or 0
                    margin    = ff.get("profit_margins") or 0
                    beta_v    = ff.get("beta") or 1
                    rsi_v     = ff.get("rsi") or 50
                    mom3      = ff.get("mom_3m") or 0
                    mom6      = ff.get("mom_6m") or 0
                    rec_v     = (ff.get("recommendation") or "hold").capitalize()
                    mktcap    = ff.get("market_cap") or 0
                    conv_sigs  = ff.get("_conv_signals", [])
                    conv_score = ff.get("_conv_score", 0)
                    n_sigs     = len(conv_sigs) if conv_sigs else 0
                    cap_str   = (
                        f"${mktcap/1e9:.1f}B" if mktcap >= 1e9 else
                        f"${mktcap/1e6:.0f}M" if mktcap >= 1e6 else "—"
                    )
                    score_icon = "🟢" if sc >= 75 else "🟡" if sc >= 60 else "🔵"

                    with st.container(border=True):
                        _qs_c1, _qs_c2, _qs_c3 = st.columns([0.4, 5, 1.2])

                        with _qs_c1:
                            st.markdown(f"### {rank_i}")

                        with _qs_c2:
                            # Ticker + name + cap
                            _title = f"**{sym}** &nbsp; {name} &nbsp; `{cap_str}`"
                            if is_mc_mode and n_sigs > 0:
                                _sig_icon = "✅" if conv_score >= 6 else "⚠️" if conv_score >= 5 else "❔"
                                _title += f" &nbsp; **{conv_score}/{n_sigs} signals {_sig_icon}**"
                            st.markdown(_title)

                            # Signal tags as plain text badges
                            _tags = []
                            if rev_g   > 0.15: _tags.append(f"📈 Rev +{rev_g*100:.0f}%")
                            if margin  > 0.15: _tags.append(f"💰 Margin {margin*100:.0f}%")
                            if mom3    > 0.10: _tags.append(f"🚀 3M +{mom3*100:.0f}%")
                            if mom6    > 0.10: _tags.append(f"📊 6M +{mom6*100:.0f}%")
                            if rsi_v   > 55:   _tags.append(f"⚡ RSI {rsi_v:.0f}")
                            if _tags:
                                st.caption("  ·  ".join(_tags))

                            st.caption(
                                f"Beta {beta_v:.1f}  ·  Analyst: {rec_v}  ·  "
                                f"Price {'$' if '$' in str(curr) else curr}{curr_px:.2f}  ·  Cap {cap_str}"
                            )

                        with _qs_c3:
                            st.metric("Score", f"{score_icon} {sc:.0f}")

                st.markdown("---")
                st.caption(
                    "💡 Click **⚙️ Advanced Screener** tab above to deep-dive into "
                    "individual stocks, run Monte Carlo, or change sector/theme filters."
                )

            elif "qs_results" in st.session_state and not st.session_state["qs_results"]:
                st.warning(
                    "No stocks passed the filters for this scan. "
                    "Try a different goal or include more markets."
                )

        # ── SUB-TAB 3: Fundamentals ────────────────────────────────────────────────
        with _fs_t2:
            # ── Header ───────────────────────────────────────────────────────────────
            st.markdown("## 🔮 Stock Screener & Predictive Portfolio Engine")
            st.caption(
                "Discovers undiscovered stocks using dynamic universe scanning, "
                "4-layer scoring, Monte Carlo simulation (10,000 paths), and "
                "XGBoost ML prediction. Bias toward low analyst coverage, "
                "small/mid cap, and emerging themes."
            )

            # ── Market Regime Banner (multi-signal) ──────────────────────────────────
            vix_val             = fetch_vix()
            vix_regime, vix_cfg = get_vix_regime(vix_val)
            _t2_mk = "IN" if market_label.startswith("🇮🇳") else "US"
            _t2_rd = detect_market_regime(_t2_mk)

            st.markdown(
                f"""<div style="background:{_t2_rd['bg']};border:1px solid {_t2_rd['color']};
                border-radius:8px;padding:12px 18px;margin-bottom:8px;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
                  <div>
                    <span style="color:{_t2_rd['color']};font-weight:700;font-size:1rem">
                    {_t2_rd['emoji']} {_t2_rd['regime']} MARKET REGIME</span>
                    <span style="color:#aaa;font-size:0.83rem;margin-left:14px">{_t2_rd['desc']}</span>
                  </div>
                  <div style="font-size:0.78rem;color:#888;text-align:right">
                    VIX {_t2_rd['vix']} &nbsp;·&nbsp; vs 200MA {_t2_rd['vs_200ma']:+.1f}%
                    &nbsp;·&nbsp; 3M mom {_t2_rd['mom_3m']:+.1f}%
                    &nbsp;·&nbsp; Kelly mult {_t2_rd['kelly_mult']:.0%}
                    &nbsp;·&nbsp; {'☀️ Golden' if _t2_rd['golden'] else '💀 Death'} cross
                  </div>
                </div>
                <div style="background:#111;border-radius:6px;padding:6px 14px;margin-bottom:10px;
                font-size:0.82rem;color:{_t2_rd['color']}">
                  📌 {_t2_rd['action']}
                  <span style="color:#555;margin-left:16px">
                  Active slots: Anchor({vix_cfg['anchor']}) ·
                  Growth({vix_cfg['growth']}) ·
                  Rotational({vix_cfg['rotational']}) ·
                  On-deck({vix_cfg['ondeck']})
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── User Controls ────────────────────────────────────────────────────────
            with st.container():
                st.markdown("### ⚙️ Portfolio Configuration")
                cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

                with cfg_col1:
                    portfolio_size = st.number_input(
                        "Total Portfolio Size",
                        min_value=10_000, max_value=100_000_000,
                        value=1_000_000, step=10_000,
                        help="Your total investable capital",
                    )
                    portfolio_currency = st.selectbox("Currency", ["₹ INR", "$ USD"])
                    port_curr = "₹" if "₹" in portfolio_currency else "$"

                with cfg_col2:
                    risk_appetite = st.selectbox(
                        "Risk Appetite",
                        ["Conservative", "Moderate", "Aggressive"],
                        index=1,
                    )
                    if risk_appetite == "Conservative":
                        alloc = {"anchor": 0.50, "growth": 0.35, "rotational": 0.15}
                    elif risk_appetite == "Aggressive":
                        alloc = {"anchor": 0.20, "growth": 0.40, "rotational": 0.40}
                    else:
                        alloc = {"anchor": 0.30, "growth": 0.40, "rotational": 0.30}

                    inv_horizon = st.selectbox(
                        "Investment Horizon",
                        ["3 Months", "6 Months", "1 Year", "3 Years"],
                        index=2,
                    )

                with cfg_col3:
                    markets_sel = st.multiselect(
                        "Markets",
                        ["🇮🇳 NSE India", "🇺🇸 US Markets"],
                        default=["🇮🇳 NSE India", "🇺🇸 US Markets"],
                    )
                    min_cap_opt = st.selectbox(
                        "Minimum Market Cap",
                        [
                            "Any (include micro cap)",
                            "Small Cap+ (>₹500Cr / >$300M)",
                            "Mid Cap+  (>₹5000Cr / >$2B)",
                            "Large Cap (>₹20000Cr / >$10B)",
                        ],
                        index=1,
                    )
                    include_smallcap = st.checkbox(
                        "🔬 Include US Small Caps (S&P 600)",
                        value=False,
                        help=(
                            "Adds ~600 small-cap stocks to the US scan universe. "
                            "Higher potential returns but also higher risk and volatility. "
                            "Best paired with 'Any' or 'Small Cap+' market cap filter. "
                            "Scan will take slightly longer."
                        ),
                    )
                    momentum_mode = st.radio(
                        "Stock Bias",
                        ["Both", "Momentum Only", "Contrarian Only"],
                        horizontal=True,
                    )

            st.markdown("---")

            # ── Return Objective ─────────────────────────────────────────────────────
            st.markdown("### 🎯 Return Objective *(optional)*")
            st.caption(
                "Set a specific return target and duration. The screener will rank stocks "
                "by their Monte Carlo probability of achieving your goal. "
                "Leave at 0% to use the standard quality-based scoring."
            )
            obj_col1, obj_col2, obj_col3 = st.columns([2, 2, 3])
            with obj_col1:
                return_target = st.select_slider(
                    "Target Return",
                    options=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
                    value=0,
                    format_func=lambda x: "Off" if x == 0 else f"+{x}%",
                    key="return_target",
                )
            with obj_col2:
                return_horizon = st.radio(
                    "Over",
                    ["1M", "3M", "6M", "1Y"],
                    index=1,
                    horizontal=True,
                    key="return_horizon",
                    disabled=(return_target == 0),
                )
            with obj_col3:
                if return_target > 0:
                    # Contextual warning / guidance
                    _, td = OBJECTIVE_HORIZON_MAP.get(return_horizon, ("3M", 63))
                    if return_target >= 40 and td <= 63:
                        st.warning(
                            f"⚠️ **{return_target}% in {return_horizon} is very aggressive.** "
                            f"Only high-volatility stocks (small-cap, high-beta, momentum) "
                            f"have meaningful probability of achieving this. "
                            f"These same stocks can also fall 30-50% — size positions carefully."
                        )
                    elif return_target >= 25:
                        st.info(
                            f"📊 Screener will prioritise stocks with high MC probability "
                            f"of {return_target}%+ in {return_horizon}. "
                            f"Scoring weights will shift toward momentum and volatility."
                        )
                    else:
                        st.success(
                            f"✅ {return_target}% in {return_horizon} — achievable by "
                            f"quality growth stocks. Standard + objective scoring applied."
                        )
                else:
                    st.info("Objective mode off — using standard quality-based scoring.")

            objective_mode = return_target > 0

            st.markdown("---")

            # ── Theme Explorer ───────────────────────────────────────────────────────
            st.markdown("### 🔭 Theme Explorer")
            st.caption(
                "Select themes to add undiscovered stocks to the screening "
                "universe. Gemini AI finds relevant companies automatically."
            )

            theme_col1, theme_col2 = st.columns([3, 1])
            with theme_col1:
                all_themes = INDIA_THEMES + US_THEMES
                sel_themes = st.multiselect(
                    "Select Themes to Include",
                    all_themes,
                    default=[
                        "Quantum Computing India",
                        "Quantum Computing Hardware",
                        "Defence Technology India PLI",
                        "EV Battery Components India",
                        "AI Infrastructure Data Centers",
                    ],
                    help="Gemini AI discovers small/mid cap stocks in these themes "
                         "and adds them to the screening universe.",
                )
            with theme_col2:
                custom_theme = st.text_input(
                    "Custom Theme", placeholder="e.g. underwater drone tech"
                )
                if custom_theme:
                    sel_themes = list(sel_themes) + [custom_theme]
                if st.button("🎲 Surprise Me"):
                    import random
                    random_themes = random.sample(all_themes, 3)
                    sel_themes = list(set(list(sel_themes) + random_themes))
                    st.info(f"Added: {', '.join(random_themes)}")

            # ── Industry / Sector Filter (optional, applied on results) ──────────────
            st.markdown("#### 🏭 Industry Filter *(optional)*")
            st.caption(
                "Filter screener results by sector. Leave empty to show all industries. "
                "Applied after the scan — no need to re-run the screener when changing this."
            )
            ind_col1, ind_col2 = st.columns([3, 1])
            with ind_col1:
                sel_industries = st.multiselect(
                    "Show only these industries",
                    INDUSTRY_OPTIONS,
                    default=[],
                    help="Partial match — 'Technology' will match 'Information Technology' too.",
                    key="sel_industries",
                )
            with ind_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔄 Clear Filter", key="clr_ind"):
                    sel_industries = []

            st.markdown("---")

            # ── Action Buttons ───────────────────────────────────────────────────────
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            run_screener   = btn_col1.button("🔍 Run Full Screener",   type="primary",  use_container_width=True)
            retrain_xgb    = btn_col2.button(
                "🤖 Retrain XGBoost",
                use_container_width=True,
                disabled=not XGB_AVAILABLE,
                help="XGBoost available ✅" if XGB_AVAILABLE else "XGBoost unavailable — run: brew install libomp && pip install xgboost",
            )
            validate_today = btn_col3.button("✅ Validate Predictions",                   use_container_width=True)
            show_tracker   = btn_col4.button("📊 Prediction Tracker",                     use_container_width=True)

            # ── Session state init ───────────────────────────────────────────────────
            if "screener_results" not in st.session_state:
                st.session_state["screener_results"] = None
            if "xgb_models" not in st.session_state:
                st.session_state["xgb_models"] = None

            # Detect setting changes → clear stale results so re-run is required
            _settings_key = f"{sorted(markets_sel)}|{min_cap_opt}|{risk_appetite}|{inv_horizon}|{momentum_mode}|sc={include_smallcap}|obj={return_target}@{return_horizon}"
            if st.session_state.get("_screener_settings_key") != _settings_key:
                if st.session_state.get("_screener_settings_key") is not None:
                    # Settings changed after a previous run — invalidate cached results
                    st.session_state["screener_results"] = None
                    st.info(
                        f"⚙️ Settings changed — previous results cleared. "
                        f"Click **🔍 Run Full Screener** to apply the new settings.",
                        icon="🔄",
                    )
                st.session_state["_screener_settings_key"] = _settings_key

            # ── XGBoost availability notice ───────────────────────────────────────────
            if not XGB_AVAILABLE:
                st.caption(
                    "ℹ️ XGBoost not available — predictions use Monte Carlo only. "
                    "To enable ML blending: `brew install libomp` then `pip install xgboost` and restart."
                )

            # ── Retrain XGBoost ──────────────────────────────────────────────────────
            if retrain_xgb:
                if not XGB_AVAILABLE:
                    pass   # button is disabled; this branch is unreachable
                else:
                    with st.spinner("Training XGBoost on historical data..."):
                        seed_syms = build_universe_india()[:30] + build_universe_us()[:30]
                        models    = train_xgb_models(tuple(seed_syms))
                        st.session_state["xgb_models"] = models
                    if models:
                        st.success(f"✅ XGBoost trained on {len(seed_syms)} stocks.")
                    else:
                        st.warning("Training failed — insufficient data.")

            # ── Validate Today ───────────────────────────────────────────────────────
            if validate_today:
                with st.spinner("Validating open predictions..."):
                    summary = validate_predictions()
                st.markdown("#### Validation Summary")
                vc1, vc2, vc3, vc4, vc5 = st.columns(5)
                vc1.metric("Active",          summary["active"])
                vc2.metric("On Track",        summary["on_track"])
                vc3.metric("Outperforming",   summary["outperforming"])
                vc4.metric("Underperforming", summary["underperforming"])
                vc5.metric("Missed/Stopped",  summary["missed"] + summary["stopped"])

            # ── Prediction Tracker ───────────────────────────────────────────────────
            if show_tracker:
                preds = load_predictions()
                if not preds:
                    st.info("No predictions logged yet. Run the screener and click "
                            "'Log This Prediction' on any stock card.")
                else:
                    st.markdown("### 📊 Prediction Tracker")
                    st.caption(
                        "Rows = models. Columns = daily dates from log date (actual prices) then milestone horizons. "
                        "Green cell = stock beat the model's prediction. Red = missed. — = date not reached yet."
                    )

                    # ── Controls ─────────────────────────────────────────────────────
                    tc1, tc2, tc3 = st.columns([2, 2, 2])
                    n_days = tc1.radio("Daily view", [7, 10], horizontal=True, index=0)
                    if tc2.button("🧹 Remove Duplicates", help="Keep only the latest entry per symbol"):
                        seen = {}
                        for pred in preds:
                            seen[pred["symbol"]] = pred
                        deduped = list(seen.values())
                        save_predictions(deduped)
                        st.success(f"Kept {len(deduped)} unique predictions (removed {len(preds)-len(deduped)}).")
                        preds = deduped
                    if tc3.button("🗑️ Clear All Predictions"):
                        save_predictions([])
                        st.success("Cleared.")
                        preds = []

                    # Deduplicate for display (show only latest per symbol)
                    seen_syms = {}
                    for pred in preds:
                        seen_syms[pred["symbol"]] = pred
                    display_preds = list(reversed(list(seen_syms.values())))

                    MILESTONE_HORIZONS = ["1W", "1M", "3M", "6M", "1Y"]
                    # Trading-day counts for milestones
                    MILESTONE_DAYS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}

                    def _interp(p_near, p_far, near_day, far_day, day):
                        """Linear interpolation between two stored horizon prices."""
                        if far_day == near_day: return p_near
                        t = (day - near_day) / (far_day - near_day)
                        return p_near + (p_far - p_near) * t

                    def _daily_model_price(p, model_key, day, entry):
                        """Predict price at trading day `day` for a stored model (mc_p50, mc_p90, jd_p50)."""
                        m = p.get(model_key, {})
                        p1d = m.get("1D", {}).get("price")
                        p1w = m.get("1W", {}).get("price")
                        if not p1d and not p1w:
                            # Fallback: use daily_drift
                            drift = float(p.get("daily_drift", 0) or 0)
                            return round(entry * (1 + drift) ** day, 2)
                        if not p1w: return round(p1d, 2)
                        if not p1d: return round(p1w, 2)
                        if day <= 5:
                            return round(_interp(p1d, p1w, 1, 5, max(1, day)), 2)
                        # Extrapolate beyond 1W
                        daily_rate = (p1w - p1d) / 4
                        return round(max(0.01, p1w + daily_rate * (day - 5)), 2)

                    def _score_daily_price(entry, annual_ret, day):
                        daily = annual_ret / 252
                        return round(max(0.01, entry * (1 + daily) ** day), 2)

                    def _pct(price, entry):
                        return round((price / entry - 1) * 100, 2)

                    def _price_cell(price, entry, is_actual=False, future=False):
                        if future:
                            pct = _pct(price, entry)
                            clr = "#00c853" if pct > 0 else ("#ff5252" if pct < 0 else "#888")
                            return (f"<td style='padding:4px 8px;text-align:center;color:{clr};opacity:0.5'>"
                                    f"{pct:+.1f}%<br><span style='font-size:0.7rem;color:#444'>"
                                    f"{price:.2f}</span></td>")
                        pct = _pct(price, entry)
                        clr = "#00c853" if pct > 0 else ("#ff5252" if pct < 0 else "#888")
                        fw  = "font-weight:700;" if is_actual else ""
                        return (f"<td style='padding:4px 8px;text-align:center;{fw}background:#111'>"
                                f"<span style='color:{clr};{fw}'>{pct:+.1f}%</span><br>"
                                f"<span style='font-size:0.7rem;color:#666'>{price:.2f}</span></td>")

                    def _empty_cell(label="—"):
                        return f"<td style='padding:4px 8px;text-align:center;color:#333'>{label}</td>"

                    def _milestone_model_price(p, model_key, horizon):
                        m = p.get(model_key, {})
                        h = m.get(horizon)
                        return h["price"] if h else None

                    for p in display_preds:
                        cur   = p["currency"]
                        sym   = p["symbol"]
                        entry = float(p["entry_price"])
                        name  = p.get("name", sym)[:28]
                        log_d = p["logged_at"][:10]
                        checks = p.get("validation_checks", [])
                        status = checks[-1]["result"] if checks else "ACTIVE"
                        stat_clr = {"ACTIVE":"#2979ff","ON_TRACK":"#00c853","OUTPERFORMING":"#00c853",
                                    "UNDERPERFORMING":"#ff9800","MISSED":"#ff5252","STOPPED_OUT":"#ff1744"
                                    }.get(status, "#888")

                        # Fetch actual daily prices
                        daily_actual = fetch_daily_prices(sym, p["logged_at"], n_days)
                        # Fetch milestone actual prices
                        milestone_actual = fetch_prediction_movements(sym, p["logged_at"], entry)

                        # ── Stock header ─────────────────────────────────────────────
                        st.markdown(
                            f"<div style='background:#0d0d0d;border:1px solid #222;"
                            f"border-radius:8px;padding:10px 16px;margin-top:12px'>"
                            f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
                            f"<b style='color:#fff;font-size:1rem'>{sym}</b>"
                            f"<span style='color:#888;font-size:0.82rem'>{name}</span>"
                            f"<span style='color:#aaa;font-size:0.8rem'>Entry: {cur}{entry:.2f}</span>"
                            f"<span style='color:#888;font-size:0.79rem'>Logged: {log_d}</span>"
                            f"<span style='background:{stat_clr}22;color:{stat_clr};padding:2px 8px;"
                            f"border-radius:4px;font-size:0.75rem;font-weight:700'>{status}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # ── Build column headers ──────────────────────────────────────
                        # Daily headers
                        day_headers = ""
                        for d in range(1, n_days + 1):
                            di      = daily_actual.get(d)
                            # di always has "date" (pre-generated); "price" may be None
                            lbl     = di["date"] if di else f"D+{d}"
                            is_past = (di is not None and di.get("price") is not None)
                            style   = "color:#aaa" if is_past else "color:#555;font-style:italic"
                            day_headers += (f"<th style='padding:4px 8px;{style};"
                                            f"font-weight:500;text-align:center;font-size:0.77rem'>{lbl}</th>")
                        # Separator + milestone headers
                        ms_headers = (
                            "<th style='padding:4px 8px;color:#555;text-align:center;font-size:0.72rem;"
                            "border-left:2px solid #333'>│</th>"
                        )
                        for ms in MILESTONE_HORIZONS:
                            ms_headers += (f"<th style='padding:4px 8px;color:#666;font-weight:500;"
                                           f"text-align:center;font-size:0.77rem'>{ms}</th>")

                        # ── Rows ─────────────────────────────────────────────────────
                        MODELS = [
                            ("📈 Actual Price",        "actual",   True),
                            ("📊 MC P50 (Median)",      "mc_p50",  False),
                            ("📊 MC P90 (Bull)",        "mc_p90",  False),
                            ("⚡ Jump Diffusion MC",    "jd_p50",  False),
                            ("🤖 XGBoost",              "xgb",     False),
                            ("⚡ Technical Score",      "tech",    False),
                            ("💰 Fundamental Score",    "fund",    False),
                        ]

                        rows_html = ""
                        for row_idx, (label, mkey, is_act) in enumerate(MODELS):
                            # Skip rows with no data
                            has_data = True
                            if mkey == "actual":
                                has_data = True  # always show
                            elif mkey in ("mc_p50", "mc_p90", "jd_p50"):
                                has_data = bool(p.get(mkey) or p.get("daily_drift") is not None)
                            elif mkey == "xgb":
                                has_data = bool(p.get("xgb"))
                            elif mkey == "tech":
                                has_data = p.get("tech_annual") is not None
                            elif mkey == "fund":
                                has_data = p.get("fund_annual") is not None

                            if not has_data:
                                continue

                            row_bg = "background:#0a0a0a;" if row_idx == 0 else ""
                            cells  = (f"<td style='padding:4px 10px;{row_bg}white-space:nowrap;"
                                      f"color:{'#ddd' if is_act else '#888'};font-size:0.8rem;"
                                      f"font-weight:{'600' if is_act else '400'}'>{label}</td>")

                            # Daily cells
                            for d in range(1, n_days + 1):
                                di          = daily_actual.get(d)          # {date, price|None}
                                actual_px   = di["price"] if di else None  # None = future
                                has_actual  = actual_px is not None
                                is_future   = not has_actual

                                if mkey == "actual":
                                    if has_actual:
                                        cells += _price_cell(actual_px, entry, is_actual=True)
                                    else:
                                        cells += _empty_cell()
                                elif mkey in ("mc_p50", "mc_p90", "jd_p50"):
                                    pred_px = _daily_model_price(p, mkey, d, entry)
                                    cells += _price_cell(pred_px, entry, future=is_future)
                                elif mkey == "xgb":
                                    cells += _empty_cell("—")
                                elif mkey == "tech":
                                    ann = float(p.get("tech_annual", 0) or 0)
                                    if d <= 63:
                                        pred_px = _score_daily_price(entry, ann, d)
                                        cells += _price_cell(pred_px, entry, future=is_future)
                                    else:
                                        cells += _empty_cell()
                                elif mkey == "fund":
                                    ann = float(p.get("fund_annual", 0) or 0)
                                    if d >= 21:
                                        pred_px = _score_daily_price(entry, ann, d)
                                        cells += _price_cell(pred_px, entry, future=is_future)
                                    else:
                                        cells += _empty_cell()

                            # Separator
                            cells += "<td style='padding:4px 2px;color:#333;border-left:2px solid #333'>│</td>"

                            # Milestone cells
                            for ms in MILESTONE_HORIZONS:
                                ms_days = MILESTONE_DAYS[ms]
                                if mkey == "actual":
                                    mv = milestone_actual.get(ms)
                                    if mv:
                                        cells += _price_cell(mv["actual_price"], entry, is_actual=True)
                                    else:
                                        cells += _empty_cell()
                                elif mkey in ("mc_p50", "mc_p90", "jd_p50"):
                                    mp = _milestone_model_price(p, mkey, ms)
                                    if mp:
                                        mv = milestone_actual.get(ms)
                                        is_future = mv is None
                                        cells += _price_cell(mp, entry, future=is_future)
                                    else:
                                        # legacy: try targets
                                        tgt = p.get("targets", {}).get(ms)
                                        if tgt and mkey == "mc_p50":
                                            cells += _price_cell(tgt["price"], entry, future=True)
                                        else:
                                            cells += _empty_cell()
                                elif mkey == "xgb":
                                    xp = (p.get("xgb") or {}).get(ms)
                                    if xp:
                                        mv = milestone_actual.get(ms)
                                        is_future = mv is None
                                        cells += _price_cell(xp["price"], entry, future=is_future)
                                    else:
                                        cells += _empty_cell("—")
                                elif mkey == "tech":
                                    ann = float(p.get("tech_annual", 0) or 0)
                                    if ms in ("1W", "1M", "3M"):
                                        pred_px = _score_daily_price(entry, ann, ms_days)
                                        mv = milestone_actual.get(ms)
                                        cells += _price_cell(pred_px, entry, future=(mv is None))
                                    else:
                                        cells += _empty_cell()
                                elif mkey == "fund":
                                    ann = float(p.get("fund_annual", 0) or 0)
                                    if ms in ("3M", "6M", "1Y"):
                                        pred_px = _score_daily_price(entry, ann, ms_days)
                                        mv = milestone_actual.get(ms)
                                        cells += _price_cell(pred_px, entry, future=(mv is None))
                                    else:
                                        cells += _empty_cell()

                            rows_html += (f"<tr style='{row_bg}border-bottom:1px solid #1a1a1a'>"
                                          f"{cells}</tr>")

                        st.markdown(
                            f"""<div style="overflow-x:auto;margin-top:6px">
                            <table style="width:100%;border-collapse:collapse;font-size:0.8rem">
                              <thead>
                                <tr style="border-bottom:2px solid #333">
                                  <th style="padding:4px 10px;color:#555;text-align:left;
                                  font-size:0.75rem;min-width:160px">Model</th>
                                  {day_headers}{ms_headers}
                                </tr>
                              </thead>
                              <tbody>{rows_html}</tbody>
                            </table></div>
                            <div style='font-size:0.72rem;color:#444;margin-top:4px;padding:0 4px'>
                            Solid = past (actual data exists) · Faded = future (model prediction) ·
                            XGBoost predicts only 1M/3M/1Y milestones
                            </div></div>""",
                            unsafe_allow_html=True,
                        )


            # ── Main Screener Run ────────────────────────────────────────────────────
            if run_screener:
                st.markdown("---")
                st.markdown("### 🔍 Scanning Universe...")

                # Build universe
                universe_syms = []
                if "🇮🇳 NSE India" in markets_sel:
                    universe_syms += build_universe_india()   # already includes Smallcap 250
                if "🇺🇸 US Markets" in markets_sel:
                    universe_syms += build_universe_us()
                    if include_smallcap:
                        sc_syms = build_universe_us_smallcap()
                        universe_syms += sc_syms
                        st.caption(f"Small-cap mode: added {len(sc_syms)} S&P 600 stocks to scan universe.")
                universe_syms = list(set(universe_syms))

                cap_filter_map = {
                    "Any (include micro cap)":           0,
                    "Small Cap+ (>₹500Cr / >$300M)":     3e8,
                    "Mid Cap+  (>₹5000Cr / >$2B)":       2e9,
                    "Large Cap (>₹20000Cr / >$10B)":      1e10,
                }
                min_cap = cap_filter_map.get(min_cap_opt, 0)

                st.markdown(
                    f"Universe: **{len(universe_syms)} stocks** across "
                    f"{', '.join(markets_sel)}"
                )

                # PASS 1 — Fast momentum pre-filter
                st.markdown("**Pass 1:** Quick momentum pre-screen...")
                p1_bar  = st.progress(0)
                p1_text = st.empty()

                @st.cache_data(ttl=3600, show_spinner=False)
                def fast_screen(syms_tuple):
                    results = {}
                    for sym in syms_tuple:
                        try:
                            fi     = yf.Ticker(sym).fast_info
                            price  = getattr(fi, "last_price",  None)
                            mktcap = getattr(fi, "market_cap",  None)
                            if price and price > 0:
                                results[sym] = {"price": price, "mktcap": mktcap}
                        except Exception:
                            pass
                    return results

                fast_res = fast_screen(tuple(sorted(universe_syms)))

                if min_cap > 0:
                    fast_res = {
                        k: v for k, v in fast_res.items()
                        if v.get("mktcap") and v["mktcap"] >= min_cap
                    }

                p1_bar.progress(min(1.0, 1.0))
                p1_text.text(f"Pass 1 complete: {len(fast_res)} stocks passed market cap filter.")

                # Guard: empty universe after cap filter
                if not fast_res:
                    st.error(
                        "No stocks passed the market cap filter. "
                        "Try selecting 'Any (include micro cap)' or a broader market."
                    )
                    break

                # Add theme-discovered stocks
                theme_syms = []
                if sel_themes:
                    api_key = os.getenv("GEMINI_API_KEY", "")
                    if api_key:
                        st.markdown("**Theme Discovery:** Finding stocks via Gemini...")
                        theme_bar = st.progress(0)
                        mkt_hint  = (
                            "india" if "NSE" in str(markets_sel) and "US" not in str(markets_sel)
                            else "us" if "US" in str(markets_sel) and "NSE" not in str(markets_sel)
                            else "both"
                        )
                        for ti, theme in enumerate(sel_themes[:10]):
                            discovered = discover_theme_stocks(theme, api_key, mkt_hint)
                            for d in discovered:
                                sym = d["symbol"]
                                if sym not in fast_res:
                                    fast_res[sym] = {"price": 1, "mktcap": None}
                                    theme_syms.append(sym)
                            theme_bar.progress((ti + 1) / min(len(sel_themes), 10))
                        st.caption(f"Theme discovery added {len(theme_syms)} additional stocks.")

                # Build candidate list — shuffle to avoid alphabetical bias (was cutting A-D only)
                # Theme-discovered stocks are pinned first so they always enter the scan
                import random as _random
                all_syms      = list(fast_res.keys())
                theme_pinned  = [s for s in all_syms if s in set(theme_syms)]
                rest          = [s for s in all_syms if s not in set(theme_syms)]
                _random.shuffle(rest)
                candidate_syms = (theme_pinned + rest)[:400]   # increased from 200 → 400

                # ── PASS 2 — Technical scan (price-data only, fast) ───────────────────
                st.markdown(f"**Pass 2:** Technical scan of {len(candidate_syms)} candidates "
                            f"(RSI, MACD, momentum, MA crossovers…)")
                p2_bar  = st.progress(0)
                p2_text = st.empty()

                features_dict = scan_universe_parallel(candidate_syms, p2_bar, p2_text, max_workers=20)
                p2_bar.progress(min(1.0, 1.0))
                p2_text.text(f"Pass 2 complete: {len(features_dict)} stocks analysed.")

                if not features_dict:
                    st.error(
                        "Pass 2 returned no results — yfinance may be rate-limiting. "
                        "Wait 60 seconds and try again, or select fewer markets."
                    )
                    break

                # Apply momentum / contrarian pre-filter
                if momentum_mode == "Momentum Only":
                    features_dict = {k: v for k, v in features_dict.items()
                                     if (v.get("mom_1m") or 0) > 0 and (v.get("mom_3m") or 0) > 0}
                elif momentum_mode == "Contrarian Only":
                    features_dict = {k: v for k, v in features_dict.items()
                                     if (v.get("mom_3m") or 0) < -0.10}

                if len(features_dict) < 5:
                    st.error("Not enough stocks passed filters. Try relaxing market cap or momentum filters.")
                    break

                # Quick pre-rank → narrow to top 100 for fundamental fetch
                # Scoring is bi-modal: momentum signals for short/aggressive, quality for long/conservative
                def quick_tech_score(f, rp=risk_appetite, ih=inv_horizon):
                    s = 0
                    mom3  = float(f.get("mom_3m")      or 0)
                    mom1y = float(f.get("mom_1y")      or 0)
                    ma200 = int(f.get("above_ma200")   or 0)
                    cross = int(f.get("golden_cross")  or 0)
                    rsi   = float(f.get("rsi")         or 50)
                    shar  = float(f.get("sharpe")      or 0)
                    vsurg = float(f.get("vol_surge")   or 1)

                    if rp == "Conservative" or ih in ("1 Year", "3 Years"):
                        # Quality / stability biased — rewards steady uptrend and low drawdown
                        s += 30 if ma200                         else 0
                        s += 25 if shar > 0.5                    else (10 if shar > 0.2 else 0)
                        s += 20 if mom1y > 0.05                  else 0
                        s += 15 if 30 <= rsi <= 65               else 0  # not overbought
                        s += 10 if cross                         else 0
                    elif rp == "Aggressive" or ih == "3 Months":
                        # Momentum biased — rewards hot stocks
                        s += 30 if mom3 > 0.10                   else (15 if mom3 > 0.03 else 0)
                        s += 20 if ma200                         else 0
                        s += 15 if cross                         else 0
                        s += 20 if 45 <= rsi <= 70               else 0
                        s += 15 if vsurg > 1.5                   else 0
                    else:
                        # Balanced default
                        s += 25 if mom3 > 0.05                   else (10 if mom3 > 0 else 0)
                        s += 20 if ma200                         else 0
                        s += 15 if cross                         else 0
                        s += 20 if 40 <= rsi <= 65               else 0
                        s += 10 if vsurg > 1.5                   else 0
                        s += 10 if shar > 0.3                    else 0
                    return s

                top50_syms = sorted(
                    features_dict.keys(),
                    key=lambda s: quick_tech_score(features_dict[s]),
                    reverse=True,
                )[:100]   # increased 80 → 100 for better bucket diversity

                # ── PASS 3 — Fundamental fetch for top 80 (parallel) ─────────────────
                # All 4 scoring layers get real data before ranking.
                st.markdown(f"**Pass 3:** Fetching fundamentals for top **{len(top50_syms)}** "
                            f"technically-ranked candidates (P/E, ROE, margins, analyst data…)")
                st.caption("⏱ This takes 2–4 minutes. Fundamentals are used to correctly weight "
                           "Anchor vs Growth vs Rotational — without this, selection would be "
                           "momentum-only and miss quality/value stocks.")
                p3_bar  = st.progress(0)
                p3_text = st.empty()
                done_p3 = 0

                def _enrich_task(sym):
                    enrich_with_fundamentals(features_dict[sym])
                    return sym

                with ThreadPoolExecutor(max_workers=10) as ex:
                    futs = {ex.submit(_enrich_task, s): s for s in top50_syms}
                    for fut in as_completed(futs):
                        try:
                            fut.result(timeout=15)
                        except Exception:
                            pass
                        done_p3 += 1
                        p3_bar.progress(min(1.0, done_p3 / len(top50_syms)))
                        p3_text.text(f"Fundamentals: {done_p3}/{len(top50_syms)} enriched "
                                     f"(P/E, ROE, margins, debt, analyst ratings…)")

                p3_bar.progress(1.0)
                p3_text.text(f"Pass 3 complete — all {len(top50_syms)} candidates have full data.")

                # Narrow features_dict to only the fundamental-enriched top50
                features_dict = {s: features_dict[s] for s in top50_syms if s in features_dict}

                # ── Full 4-layer scoring on enriched universe ─────────────────────────
                ranked = normalise_universe(features_dict)

                # Objective-mode: override weights toward momentum/volatility
                _obj_weights = get_objective_scoring_weights(return_target, return_horizon) if objective_mode else None

                all_scored = {}
                for sym, f_data in features_dict.items():
                    bucket_key  = classify_bucket(f_data, ranked)
                    # In objective mode, force all stocks into growth/rotational buckets
                    if objective_mode and return_target >= 30:
                        bucket_key = "rotational" if (f_data.get("mom_3m") or 0) > 0.05 else "growth"
                    score, bkdn = score_stock(f_data, ranked, bucket_key,
                                               risk_pref=risk_appetite, inv_horizon=inv_horizon)
                    all_scored[sym] = {
                        "features":  f_data,
                        "bucket":    bucket_key,
                        "score":     score,
                        "breakdown": bkdn,
                    }

                # Sort within each bucket, apply VIX regime counts
                bucket_stocks = {"anchor": [], "growth": [], "rotational": []}
                for sym, data in all_scored.items():
                    bucket_stocks[data["bucket"]].append((sym, data))
                for bk in bucket_stocks:
                    bucket_stocks[bk].sort(key=lambda x: x[1]["score"], reverse=True)

                # ── Identify short candidates from the universe ───────────────────────
                # Stocks that scan well for shorting: weak fundamentals, overbought, expensive
                short_candidates_raw = []
                for sym, data in all_scored.items():
                    f_s = data["features"]
                    ss, ss_bkdn = score_short_candidate(f_s)
                    if ss >= 35:   # minimum short threshold
                        short_candidates_raw.append((sym, ss, ss_bkdn, f_s))
                short_candidates_raw.sort(key=lambda x: x[1], reverse=True)
                top_shorts = short_candidates_raw[:8]

                final_picks       = {}
                ondeck_list       = []
                all_bucket_scores = {}

                for bk in ["anchor", "growth", "rotational"]:
                    bk_list  = bucket_stocks[bk]
                    all_bucket_scores[bk] = [x[1]["score"] for x in bk_list]
                    count = vix_cfg[bk]
                    final_picks[bk] = bk_list[:count]
                    ondeck_list.extend(bk_list[count: count + 3])

                # ── Monte Carlo + XGBoost on final picks ──────────────────────────────
                xgb_models  = st.session_state.get("xgb_models")
                enriched    = {}
                total_picks = sum(len(v) for v in final_picks.values())
                done_mc     = 0

                mc_bar  = st.progress(0)
                mc_text = st.empty()
                mc_text.text(f"Running Monte Carlo simulations for {total_picks} final picks…")

                for bk, picks in final_picks.items():
                    enriched[bk] = []
                    for sym, data in picks:
                        f_data = data["features"]
                        mc     = monte_carlo_targets(f_data)
                        jd     = monte_carlo_jump_diffusion(f_data, n_simulations=3000)
                        xgb_p  = get_xgb_prediction(f_data, xgb_models)
                        pclass = assign_percentile(data["score"], all_bucket_scores[bk])
                        obj_score, obj_bkdn, prob_tgt = (
                            score_for_objective(f_data, mc, return_target, return_horizon)
                            if objective_mode else (None, None, None)
                        )
                        enriched[bk].append({
                            "sym":         sym,
                            "features":    f_data,
                            "score":       data["score"],
                            "breakdown":   data["breakdown"],
                            "mc":          mc,
                            "jd":          jd,
                            "xgb":         xgb_p,
                            "pclass":      pclass,
                            "obj_score":   obj_score,
                            "obj_bkdn":    obj_bkdn,
                            "prob_target": prob_tgt,
                        })
                        done_mc += 1
                        mc_bar.progress(min(1.0, done_mc / max(total_picks, 1)))
                        mc_text.text(f"Monte Carlo: {done_mc}/{total_picks} done")

                mc_bar.empty()
                mc_text.empty()

                # Objective mode: re-rank by probability of hitting target, remove hopeless stocks
                if objective_mode:
                    for bk in enriched:
                        enriched[bk].sort(key=lambda x: x.get("prob_target") or 0, reverse=True)
                        enriched[bk] = [x for x in enriched[bk] if (x.get("prob_target") or 0) >= 3]

                st.session_state["screener_results"] = {
                    "enriched":       enriched,
                    "ondeck":         ondeck_list,
                    "alloc":          alloc,
                    "port_size":      portfolio_size,
                    "port_curr":      port_curr,
                    "run_at":         datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                    "top_shorts":     top_shorts,
                    "regime":         _t2_rd,
                    "objective_mode": objective_mode,
                    "return_target":  return_target,
                    "return_horizon": return_horizon,
                }

            # ── Render Results ───────────────────────────────────────────────────────
            res = st.session_state.get("screener_results")
            if res:
                enriched    = res["enriched"]
                ondeck      = res["ondeck"]
                alloc       = res["alloc"]
                port_size   = res["port_size"]
                port_curr   = res["port_curr"]
                top_shorts      = res.get("top_shorts", [])
                res_regime      = res.get("regime", _t2_rd)
                run_at          = res["run_at"]
                res_obj_mode    = res.get("objective_mode", False)
                res_ret_target  = res.get("return_target", 0)
                res_ret_horizon = res.get("return_horizon", "3M")

                # Objective mode banner
                if res_obj_mode and res_ret_target > 0:
                    st.markdown(
                        f"<div style='background:#1a0a00;border:1px solid #ff9100;"
                        f"border-radius:8px;padding:10px 16px;margin-bottom:12px'>"
                        f"<span style='color:#ff9100;font-weight:700'>🎯 Objective Mode Active: "
                        f"+{res_ret_target}% in {res_ret_horizon}</span>"
                        f"<span style='color:#888;font-size:0.82rem;margin-left:14px'>"
                        f"Stocks ranked by MC probability of hitting your target. "
                        f"Only stocks with ≥3% probability shown. Weights shifted to momentum + volatility."
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )

                # Apply industry filter (view-only — no re-scan needed)
                _ind_filter = st.session_state.get("sel_industries", [])
                if _ind_filter:
                    def _sector_match(sector_str):
                        s = (sector_str or "").lower()
                        return any(ind.lower() in s or s in ind.lower() for ind in _ind_filter)
                    enriched_filtered = {
                        bk: [item for item in picks
                             if _sector_match(item["features"].get("sector", ""))]
                        for bk, picks in enriched.items()
                    }
                    _removed = sum(len(enriched[bk]) - len(enriched_filtered[bk]) for bk in enriched)
                    if _removed > 0:
                        st.info(
                            f"🏭 Industry filter active: **{', '.join(_ind_filter)}** — "
                            f"{_removed} stocks hidden. {sum(len(v) for v in enriched_filtered.values())} shown.",
                            icon="🔍",
                        )
                    enriched = enriched_filtered

                # Portfolio summary banner
                total_active  = sum(len(v) for v in enriched.values())
                all_base_rets = []
                for bk, picks in enriched.items():
                    bk_alloc = alloc[bk] / max(len(picks), 1)
                    for p in picks:
                        if p["mc"]:
                            all_base_rets.append(p["mc"]["3M"]["ret_base"] * bk_alloc)
                port_exp_ret = sum(all_base_rets)

                st.markdown("---")
                st.markdown(
                    f"""<div style="background:#0a0a0a;border:2px solid #333;
                    border-radius:12px;padding:20px 28px;margin-bottom:20px">
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:12px">
                    <div>
                      <div style="color:#888;font-size:0.8rem">
                      SCREENED PORTFOLIO · {run_at}</div>
                      <div style="font-size:1.4rem;font-weight:800;color:#fff;margin-top:4px">
                      {total_active} Active Stocks · {len(ondeck)} On-Deck</div>
                      <div style="color:#aaa;font-size:0.85rem;margin-top:4px">
                      XGBoost + Monte Carlo (10,000 simulations per stock)</div>
                    </div>
                    <div style="text-align:right">
                      <div style="color:#888;font-size:0.8rem">
                      PORTFOLIO EXPECTED RETURN (3M base)</div>
                      <div style="font-size:2rem;font-weight:900;
                      color:{'#00c853' if port_exp_ret > 0 else '#ff5252'}">
                      {port_exp_ret:+.1f}%</div>
                      <div style="color:#aaa;font-size:0.8rem">
                      Total capital: {port_curr}{portfolio_size:,}</div>
                    </div>
                    </div></div>""",
                    unsafe_allow_html=True,
                )

                # Allocation tiles
                ac1, ac2, ac3 = st.columns(3)
                for col, bk in zip([ac1, ac2, ac3], ["anchor", "growth", "rotational"]):
                    cfg_bk  = BUCKET_CONFIG[bk]
                    amt     = portfolio_size * alloc[bk]
                    n_picks = len(enriched[bk])
                    per_pos = amt / max(n_picks, 1)
                    col.markdown(
                        f"""<div style="background:{cfg_bk['bg']};
                        border:1px solid {cfg_bk['border']};border-radius:8px;
                        padding:14px;text-align:center">
                        <div style="color:{cfg_bk['color']};font-weight:700;
                        font-size:1rem">{cfg_bk['name']}</div>
                        <div style="color:#fff;font-size:1.3rem;font-weight:800;
                        margin:4px 0">{port_curr}{amt:,.0f}</div>
                        <div style="color:#888;font-size:0.8rem">
                        {alloc[bk]*100:.0f}% · {n_picks} stocks ·
                        {port_curr}{per_pos:,.0f}/position</div>
                        <div style="color:#666;font-size:0.75rem;margin-top:4px">
                        {cfg_bk['rebalance']} · {cfg_bk['horizon']}</div></div>""",
                        unsafe_allow_html=True,
                    )

                # ── Render each bucket ────────────────────────────────────────────────
                PCLASS_COLORS = {
                    "P99":   ("#ffd700", "#1a1400"),
                    "P95":   ("#00c853", "#002010"),
                    "P90":   ("#2979ff", "#001233"),
                    "P80":   ("#888888", "#1a1a1a"),
                    "WATCH": ("#ff9100", "#1a0d00"),
                }

                def score_bar(val, color):
                    try:
                        w = int(np.clip(float(val or 0), 0, 100))
                    except (ValueError, TypeError):
                        w = 0
                    return (
                        f'<div style="background:#222;border-radius:4px;height:8px;margin:2px 0">'
                        f'<div style="background:{color};width:{w}%;height:100%;'
                        f'border-radius:4px"></div></div>'
                    )

                for bk in ["anchor", "growth", "rotational"]:
                    cfg_bk = BUCKET_CONFIG[bk]
                    picks  = enriched[bk]
                    if not picks:
                        continue

                    amt     = portfolio_size * alloc[bk]
                    per_pos = amt / max(len(picks), 1)

                    st.markdown("---")
                    st.markdown(
                        f"""<div style="border-left:4px solid {cfg_bk['border']};
                        padding:8px 16px;margin-bottom:16px">
                        <span style="color:{cfg_bk['color']};font-size:1.2rem;
                        font-weight:800">{cfg_bk['name']}</span>
                        <span style="color:#888;font-size:0.85rem;margin-left:16px">
                        {cfg_bk['description']}</span></div>""",
                        unsafe_allow_html=True,
                    )

                    for item in picks:
                        f_data   = item["features"]
                        mc       = item["mc"]
                        xgb_p    = item["xgb"]
                        score    = item["score"]
                        bkdn     = item["breakdown"]
                        pclass   = item["pclass"]
                        sym      = item["sym"]
                        cur      = f_data["currency"]
                        price    = f_data["price"]
                        name_str = f_data["name"]
                        sector   = f_data["sector"]
                        disc_lbl = f_data.get("discovery_label", "quality")
                        disc_cfg = DISCOVERY_LABELS.get(disc_lbl, DISCOVERY_LABELS["quality"])
                        n_ana    = int(f_data.get("n_analysts") or 0)

                        tgt_1m_p, tgt_1m_r = blend_predictions(mc, xgb_p, "1M")
                        tgt_3m_p, tgt_3m_r = blend_predictions(mc, xgb_p, "3M")
                        tgt_1y_p, tgt_1y_r = blend_predictions(mc, xgb_p, "1Y")

                        stop   = mc["stop_loss"] if mc else price * 0.93
                        rr     = mc.get("rr_3m", 0) if mc else 0
                        risk_p = mc.get("risk_pct", 3.0) if mc else 3.0

                        def mc_val(horizon, key, fallback=0):
                            if mc and horizon in mc:
                                return mc[horizon].get(key, fallback)
                            return fallback

                        # Signal pills
                        pills = []
                        if (f_data.get("mom_3m") or 0) > 0.10:
                            pills.append(("🟢 Momentum Strong", "#002010", "#00c853"))
                        if f_data.get("golden_cross") == 1:
                            pills.append(("🟢 Golden Cross", "#002010", "#00c853"))
                        rsi_v = f_data.get("rsi", 50) or 50
                        if   rsi_v < 40: pills.append(("🟢 Oversold",    "#002010", "#00c853"))
                        elif rsi_v > 65: pills.append(("🔴 Overbought",  "#2a0000", "#ff5252"))
                        else:            pills.append(("🟡 RSI Neutral",  "#332900", "#ffcc00"))
                        if (f_data.get("inst_holding") or 0) > 0.50:
                            pills.append(("🟢 Inst. Backing", "#002010", "#00c853"))
                        if (f_data.get("obv_slope") or 0) > 0.05:
                            pills.append(("🟢 OBV Rising", "#002010", "#00c853"))
                        if n_ana == 0:
                            pills.append(("🔭 Zero Analyst Coverage", "#1a0033", "#9c27b0"))
                        elif n_ana <= 3:
                            pills.append(("💎 Low Coverage", "#001a1f", "#00bcd4"))

                        pills_html = " ".join([
                            f'<span style="background:{bg};color:{fg};border:1px solid {fg};'
                            f'border-radius:12px;padding:2px 10px;font-size:0.75rem;margin-right:4px">'
                            f'{lbl}</span>'
                            for lbl, bg, fg in pills
                        ])

                        # Position sizing (2% risk rule)
                        risk_per_share = max(price - stop, 0.01)
                        risk_2pct      = port_size * 0.02
                        shares_2pct    = int(risk_2pct / risk_per_share)
                        pos_value      = shares_2pct * price
                        warn_flag = (
                            "<span style='color:#ff9100;font-size:0.78rem'>"
                            "⚠️ Exceeds 25% of portfolio — consider reducing</span>"
                            if pos_value > port_size * 0.25 else ""
                        )

                        pc_fg, pc_bg = PCLASS_COLORS.get(pclass, ("#888", "#111"))
                        prob_pos  = mc_val("3M", "prob_positive", 50)
                        prob_40   = mc_val("3M", "prob_40pct", 0)
                        prob_loss = mc_val("3M", "prob_loss10", 0)

                        # Objective-mode: show target probability badge
                        prob_tgt_val = item.get("prob_target")
                        if res_obj_mode and prob_tgt_val is not None:
                            tgt_clr = ("#00c853" if prob_tgt_val >= 15 else
                                       "#ffcc00" if prob_tgt_val >= 7  else "#ff9800")
                            obj_badge = (
                                f"<div style='background:#111;border:1px solid {tgt_clr};"
                                f"border-radius:6px;padding:6px 12px;margin-bottom:8px;"
                                f"display:inline-flex;align-items:center;gap:10px'>"
                                f"<span style='color:{tgt_clr};font-weight:700;font-size:0.9rem'>"
                                f"🎯 P(+{res_ret_target}% in {res_ret_horizon}) = "
                                f"<b>{prob_tgt_val:.1f}%</b></span>"
                                f"<span style='color:#555;font-size:0.78rem'>|</span>"
                                f"<span style='color:#888;font-size:0.78rem'>"
                                f"Obj score: {item.get('obj_score', 0):.0f}/100 · "
                                f"Momentum: {(item.get('obj_bkdn') or {}).get('momentum', 0):.0f} · "
                                f"Vol: {(item.get('obj_bkdn') or {}).get('volatility', 0):.0f} · "
                                f"Catalyst: {(item.get('obj_bkdn') or {}).get('catalyst', 0):.0f}"
                                f"</span></div>"
                            )
                            st.markdown(obj_badge, unsafe_allow_html=True)

                        # Horizon tiles: P50 (median) Monte Carlo path.
                        # For high-volatility stocks P50 can decline even when upside exists —
                        # this is "volatility drag" (geometric < arithmetic mean). The P90-P99
                        # rows in the confidence table show the actual upside scenarios.
                        horizons_html = (
                            '<span style="color:#555;font-size:0.72rem;margin-right:6px">'
                            'Median path (P50) →</span>'
                            + " ".join([
                                f'<span style="background:#111;border:1px solid #222;border-radius:6px;'
                                f'padding:4px 10px;font-size:0.78rem">'
                                f'<span style="color:#888">{hl}:</span> '
                                f'<span style="color:#fff;font-weight:600">'
                                f'{cur}{mc_val(hl, "price_base", price):.2f}</span> '
                                f'<span style="color:{"#00c853" if mc_val(hl,"ret_base",0)>0 else "#ff5252"}">'
                                f'({mc_val(hl,"ret_base",0):+.1f}%)</span></span>'
                                for hl in ["1D", "1W", "1M", "3M", "1Y"]
                            ])
                        )

                        breakdown_html = "".join([
                            f'<div><div style="color:#888;font-size:0.72rem">{lbl}</div>'
                            f'{score_bar(val, clr)}'
                            f'<div style="color:#aaa;font-size:0.72rem">{val:.0f}/100</div></div>'
                            for lbl, val, clr in [
                                ("Technical",   float(bkdn.get("technical",   50) or 50), "#2979ff"),
                                ("Fundamental", float(bkdn.get("fundamental", 50) or 50), "#00c853"),
                                ("Risk/Macro",  float(bkdn.get("risk",        50) or 50), "#ff9100"),
                                ("Sentiment",   float(bkdn.get("sentiment",   50) or 50), "#9c27b0"),
                            ]
                        ])

                        st.markdown(f"""
        <div style="background:#0d0d0d;border:1px solid {cfg_bk['border']}33;
        border-left:3px solid {cfg_bk['border']};border-radius:10px;
        padding:18px 22px;margin-bottom:14px">

          <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap">
            <span style="background:{pc_bg};color:{pc_fg};border:1px solid {pc_fg};
            border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:700">{pclass}</span>
            <span style="color:{disc_cfg[1]};background:{disc_cfg[2]};border:1px solid {disc_cfg[1]};
            border-radius:6px;padding:3px 10px;font-size:0.78rem;font-weight:600">{disc_cfg[0]}</span>
            <span style="color:#fff;font-size:1.1rem;font-weight:800">{sym}</span>
            <span style="color:#aaa;font-size:0.9rem">{name_str}</span>
            <span style="color:#666">·</span>
            <span style="color:#888;font-size:0.8rem">{sector}</span>
            <span style="color:#666">·</span>
            <span style="color:#666;font-size:0.8rem">{f_data['exchange']}</span>
            <span style="margin-left:auto;color:{cfg_bk['color']};font-weight:700;
            font-size:0.9rem">Score: {score:.1f}/100</span>
          </div>

          <div style="display:flex;gap:32px;margin-bottom:12px;flex-wrap:wrap;align-items:flex-end">
            <div>
              <div style="color:#888;font-size:0.75rem">Current Price</div>
              <div style="color:#fff;font-size:1.3rem;font-weight:800">{cur}{price:.2f}</div>
            </div>
            <div style="color:#555;font-size:1.5rem;align-self:center">→</div>
            <div>
              <div style="color:#888;font-size:0.75rem">Base Target (3M · blended)</div>
              <div style="color:{cfg_bk['color']};font-size:1.3rem;font-weight:800">
              {cur}{tgt_3m_p or 0:.2f}
              <span style="font-size:0.9rem">({tgt_3m_r or 0:+.1f}%)</span></div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">Entry</div>
              <div style="color:#fff;font-size:1rem;font-weight:700">{cur}{price:.2f}</div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">Stop Loss</div>
              <div style="color:#ff5252;font-size:1rem;font-weight:700">
              {cur}{stop:.2f} <span style="color:#888;font-size:0.8rem">(-{risk_p:.1f}%)</span></div>
            </div>
            <div>
              <div style="color:#888;font-size:0.75rem">R/R Ratio</div>
              <div style="color:{'#00c853' if rr >= 2 else '#ff9100'};font-size:1rem;font-weight:700">
              {rr:.1f}:1 {'✅' if rr >= 2 else '⚠️'}</div>
            </div>
          </div>

          <div style="background:#111;border-radius:8px;padding:12px;margin-bottom:12px">
            <div style="color:#888;font-size:0.75rem;margin-bottom:6px;font-weight:600">
            CONFIDENCE BANDS (3M) · 10,000 Monte Carlo Paths</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
              <tr style="color:#666">
                <td style="padding:3px 8px">Scenario</td>
                <td style="padding:3px 8px">Price Target</td>
                <td style="padding:3px 8px">Return %</td>
                <td style="padding:3px 8px">Probability</td>
              </tr>
              <tr style="color:#ff5252">
                <td style="padding:3px 8px">Bear (P10)</td>
                <td style="padding:3px 8px">{cur}{mc_val('3M','price_bear',price):.2f}</td>
                <td style="padding:3px 8px">{mc_val('3M','ret_bear',0):+.1f}%</td>
                <td style="padding:3px 8px">10%</td>
              </tr>
              <tr style="color:#aaa">
                <td style="padding:3px 8px">Base (P50)</td>
                <td style="padding:3px 8px">{cur}{mc_val('3M','price_base',price):.2f}</td>
                <td style="padding:3px 8px">{mc_val('3M','ret_base',0):+.1f}%</td>
                <td style="padding:3px 8px">50%</td>
              </tr>
              <tr style="color:#2979ff">
                <td style="padding:3px 8px">P90 Target</td>
                <td style="padding:3px 8px;font-weight:700">{cur}{mc_val('3M','price_p90',price):.2f}</td>
                <td style="padding:3px 8px;font-weight:700">{mc_val('3M','ret_p90',0):+.1f}%</td>
                <td style="padding:3px 8px">{prob_pos:.1f}% positive</td>
              </tr>
              <tr style="color:#00c853">
                <td style="padding:3px 8px">P95 Target</td>
                <td style="padding:3px 8px;font-weight:700">{cur}{mc_val('3M','price_p95',price):.2f}</td>
                <td style="padding:3px 8px;font-weight:700">{mc_val('3M','ret_p95',0):+.1f}%</td>
                <td style="padding:3px 8px">{prob_40:.1f}% chance >40%</td>
              </tr>
              <tr style="color:#ffd700">
                <td style="padding:3px 8px">P99 Stretch</td>
                <td style="padding:3px 8px;font-weight:700">{cur}{mc_val('3M','price_p99',price):.2f}</td>
                <td style="padding:3px 8px;font-weight:700">{mc_val('3M','ret_p99',0):+.1f}%</td>
                <td style="padding:3px 8px;color:#666">⚠️ Loss>10%: {prob_loss:.1f}%</td>
              </tr>
            </table>
          </div>

          <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
            {horizons_html}
          </div>

          <div style="margin-bottom:12px">{pills_html}</div>

          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
          gap:8px;margin-bottom:12px">
            {breakdown_html}
          </div>

          <div style="background:#111;border-radius:6px;padding:10px 14px;
          margin-bottom:10px;font-size:0.82rem">
            <span style="color:#888">Position Size (2% risk rule):</span>
            <span style="color:#fff;font-weight:700;margin-left:8px">{shares_2pct:,} shares</span>
            <span style="color:#aaa;margin-left:8px">
            = {cur}{pos_value:,.0f} ({pos_value/port_size*100:.1f}% of portfolio)</span>
            {'  ' + warn_flag if warn_flag else ''}
          </div>

        </div>""", unsafe_allow_html=True)

                        # AI one-liner + Log button
                        ai_col, log_col = st.columns([5, 1])
                        with ai_col:
                            if os.getenv("GEMINI_API_KEY"):
                                @st.cache_data(ttl=3600, show_spinner=False)
                                def get_one_line_reason(sym_r, score_r, bucket_r,
                                                        mom3m_r, roe_r, n_ana_r):
                                    try:
                                        from google import genai as _g
                                        _c = _g.Client(api_key=os.getenv("GEMINI_API_KEY"))
                                        _p = (
                                            f"Stock: {sym_r}, Bucket: {bucket_r}, "
                                            f"Score: {score_r:.1f}/100, "
                                            f"3M Momentum: {(mom3m_r or 0)*100:.1f}%, "
                                            f"ROE: {(roe_r or 0)*100:.1f}%, "
                                            f"Analyst Coverage: {n_ana_r} analysts. "
                                            f"Write ONE sentence (max 20 words) explaining why "
                                            f"this stock is in the {bucket_r} bucket. "
                                            f"Plain English. No jargon. Be specific."
                                        )
                                        _r = _c.models.generate_content(
                                            model="gemini-2.0-flash", contents=_p
                                        )
                                        return _r.text.strip()
                                    except Exception:
                                        return ""
                                reason = get_one_line_reason(
                                    sym, score, bk,
                                    f_data.get("mom_3m"), f_data.get("roe"), n_ana,
                                )
                                if reason:
                                    st.caption(f"🤖 {reason}")
                        with log_col:
                            if mc and st.button("📌 Log", key=f"log_{sym}_{bk}",
                                                use_container_width=True):
                                pid = log_prediction(
                                    f_data, mc, bk, score, pclass,
                                    jd_result=item.get("jd"),
                                    bkdn=item.get("breakdown"),
                                    xgb_pred=item.get("xgb"),
                                )
                                st.success(f"Logged #{pid}")

                        # ── Analysis bullets (Tab 2 per-card) ────────────────────────
                        s2_bullets = generate_analysis_bullets(
                            f_data, mc, score, bkdn,
                            dcf=None,        # skip DCF in screener for speed
                            curr=f_data.get("currency", "$"),
                        )
                        with st.expander("💡 Analysis & Recommendation", expanded=True):
                            for bullet in s2_bullets:
                                st.markdown(bullet)

                # ── On-Deck Queue ─────────────────────────────────────────────────────
                if ondeck:
                    st.markdown("---")
                    st.markdown("### 📋 On-Deck Queue")
                    st.caption("Next stocks to rotate in when a position closes. Monitor weekly.")
                    od_rows = []
                    for sym_od, data_od in ondeck[:vix_cfg["ondeck"]]:
                        f_od   = data_od["features"]
                        cur_od = f_od["currency"]
                        od_rows.append({
                            "Symbol":    sym_od,
                            "Name":      f_od["name"][:30],
                            "Sector":    f_od["sector"],
                            "Bucket":    data_od["bucket"].title(),
                            "Score":     f"{data_od['score']:.1f}",
                            "Price":     f"{cur_od}{f_od['price']:.2f}",
                            "3M Mom":    f"{(f_od.get('mom_3m') or 0)*100:+.1f}%",
                            "Coverage":  f"{int(f_od.get('n_analysts') or 0)} analysts",
                            "Discovery": DISCOVERY_LABELS.get(
                                f_od.get("discovery_label", "quality"),
                                ("—", "", "")
                            )[0],
                        })
                    st.dataframe(pd.DataFrame(od_rows), use_container_width=True, hide_index=True)

                # ── Short Candidates ─────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🐻 Short Candidates")
                st.caption(
                    "Stocks from the same universe that score HIGH on short criteria: "
                    "overbought technically + weak fundamentals + expensive valuation. "
                    "Use these as hedges or pair-trade against your long picks."
                )
                if top_shorts:
                    sh_rows = []
                    for sym_s, ss, ss_bkdn, f_s in top_shorts[:6]:
                        cur_s = f_s.get("currency", port_curr)
                        rsi_s = float(f_s.get("rsi") or 50)
                        mom_s = float(f_s.get("mom_3m") or 0)
                        pe_s  = _safe_float(f_s.get("pe"))
                        sh_rows.append({
                            "Symbol":       sym_s,
                            "Name":         str(f_s.get("name", sym_s))[:28],
                            "Short Score":  f"{ss:.0f}/100",
                            "Price":        f"{cur_s}{f_s['price']:.2f}",
                            "RSI":          f"{rsi_s:.0f} {'🔴 overbought' if rsi_s > 70 else ''}",
                            "3M Mom":       f"{mom_s*100:+.1f}%",
                            "P/E":          f"{pe_s:.1f}x" if pe_s else "—",
                            "Tech (short)": f"{ss_bkdn.get('technical', 0)}/40",
                            "Fund (short)": f"{ss_bkdn.get('fundamental', 0)}/40",
                            "Val (short)":  f"{ss_bkdn.get('valuation', 0)}/20",
                        })
                    sh_df = pd.DataFrame(sh_rows)

                    def _color_short(val):
                        try:
                            score_val = int(str(val).split("/")[0])
                            if score_val >= 60: return "color:#ff5252;font-weight:700"
                            if score_val >= 40: return "color:#ff9800"
                            return "color:#888"
                        except Exception:
                            return ""

                    st.dataframe(sh_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "⚠️ Short selling carries theoretically unlimited loss. "
                        "Always use defined-risk structures (put spreads, stop-buy orders). "
                        "Use these signals as awareness, not as standalone trade signals."
                    )
                else:
                    st.info("No strong short candidates found in the current universe scan. "
                            "Run the screener to populate this section.")

                # ── Portfolio Correlation Matrix ──────────────────────────────────────
                st.markdown("---")
                st.markdown("### 🔗 Portfolio Correlation Matrix")
                st.caption(
                    "Are your long picks actually diversified, or are they all the same bet? "
                    ">0.70 correlation = positions move together — you have concentration risk, not diversification."
                )
                long_syms = []
                for bk, picks in enriched.items():
                    for item in picks:
                        long_syms.append(item["sym"])

                if len(long_syms) >= 2:
                    with st.spinner("Computing 6-month return correlations…"):
                        corr_df, warn_pairs = compute_correlation_matrix(tuple(long_syms))

                    if corr_df is not None and not corr_df.empty:
                        import plotly.graph_objects as _go_corr
                        fig_corr = _go_corr.Figure(data=_go_corr.Heatmap(
                            z=corr_df.values,
                            x=corr_df.columns.tolist(),
                            y=corr_df.index.tolist(),
                            colorscale=[
                                [0.0, "#003366"],
                                [0.5, "#111111"],
                                [1.0, "#660000"],
                            ],
                            zmin=-1, zmax=1,
                            text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                            texttemplate="%{text}",
                            textfont={"size": 11},
                            showscale=True,
                        ))
                        fig_corr.update_layout(
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            font=dict(color="#ccc"),
                            margin=dict(l=10, r=10, t=30, b=10),
                            height=max(300, len(long_syms) * 45 + 80),
                            title=dict(text="6-Month Return Correlations (long picks)",
                                       font=dict(size=13, color="#aaa")),
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                        if warn_pairs:
                            st.warning(
                                "⚠️ **High correlation detected — you may be less diversified than you think:**\n\n"
                                + "\n".join(
                                    f"- **{a}** ↔ **{b}**: {v:+.2f} correlation (moves together {abs(v)*100:.0f}% of the time)"
                                    for a, b, v in warn_pairs
                                )
                            )
                        else:
                            st.success("✅ All long picks have correlation <0.70 — your portfolio is genuinely diversified.")
                    else:
                        st.info("Correlation matrix unavailable (insufficient shared price history).")
                else:
                    st.info("Need at least 2 long picks to compute a correlation matrix.")

                # ── Export ────────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                exp_rows = []
                for bk, picks in enriched.items():
                    for item in picks:
                        f_d  = item["features"]
                        mc_d = item["mc"]
                        exp_rows.append({
                            "Symbol":    item["sym"],
                            "Name":      f_d["name"],
                            "Bucket":    bk,
                            "Score":     item["score"],
                            "Percentile":item["pclass"],
                            "Price":     f_d["price"],
                            "Currency":  f_d["currency"],
                            "3M_Target": mc_d["3M"]["price_base"] if mc_d else "",
                            "3M_Return": mc_d["3M"]["ret_base"]   if mc_d else "",
                            "1Y_Target": mc_d["1Y"]["price_base"] if mc_d else "",
                            "Stop_Loss": mc_d["stop_loss"]        if mc_d else "",
                            "RR_Ratio":  mc_d.get("rr_3m", "")   if mc_d else "",
                            "Analysts":  f_d.get("n_analysts", ""),
                            "Discovery": f_d.get("discovery_label", ""),
                        })
                if exp_rows:
                    exp_df = pd.DataFrame(exp_rows)
                    st.download_button(
                        "📊 Download Full Results (CSV)",
                        exp_df.to_csv(index=False),
                        file_name=f"screener_{datetime.date.today()}.csv",
                        mime="text/csv",
                    )

            else:
                st.markdown("---")
                st.info(
                    "👆 Configure your portfolio above and click "
                    "**🔍 Run Full Screener** to discover stocks.",
                    icon="🔮",
                )

            # ── Disclaimer footer ─────────────────────────────────────────────────────
            st.markdown("---")
            st.caption(
                "⚠️ DISCLAIMER: This tool is for educational and research purposes only. "
                "Predictions are probabilistic estimates based on historical data. "
                "Past performance does not guarantee future results. "
                "This is NOT financial advice. Always consult a SEBI registered advisor "
                "before investing. You are solely responsible for your investment decisions."
            )


        # ══════════════════════════════════════════════════════════════════════════════
        # TAB 3 — My Portfolio (INDmoney US stocks integration)
        # ══════════════════════════════════════════════════════════════════════════════

        def _load_portfolio():
            if PORTFOLIO_FILE.exists():
                try:
                    return json.loads(PORTFOLIO_FILE.read_text())
                except Exception:
                    return []
            return []

        def _save_portfolio(holdings):
            PORTFOLIO_FILE.write_text(json.dumps(holdings, indent=2))

        def _parse_indmoney_csv(df):
            """
            Normalise an INDmoney (or generic broker) CSV export into a standard list of holdings.
            Returns list of {symbol, name, qty, avg_price, currency}.
            Tries several common column-name patterns used by INDmoney / Groww / Zerodha exports.
            """
            df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

            # Symbol
            sym_candidates = ["symbol", "ticker", "scrip", "stock_symbol", "isin", "scrip_code"]
            sym_col = next((c for c in sym_candidates if c in df.columns), None)

            # Name
            name_candidates = ["name", "stock_name", "company", "company_name", "description", "instrument"]
            name_col = next((c for c in name_candidates if c in df.columns), None)

            # Quantity
            qty_candidates = ["qty", "quantity", "shares", "units", "quantity_held", "total_qty"]
            qty_col = next((c for c in qty_candidates if c in df.columns), None)

            # Average buy price
            price_candidates = ["avg_price", "average_price", "avg_buy_price", "avg_cost",
                                "average_cost", "buy_price", "purchase_price", "invested_at"]
            price_col = next((c for c in price_candidates if c in df.columns), None)

            if not sym_col or not qty_col:
                return None, f"Could not find symbol column (tried: {sym_candidates}) or quantity column (tried: {qty_candidates})."

            holdings = []
            for _, row in df.iterrows():
                sym = str(row.get(sym_col, "")).strip().upper()
                if not sym or sym in ("NAN", "SYMBOL", ""):
                    continue
                try:
                    qty = float(str(row.get(qty_col, 0)).replace(",", ""))
                except Exception:
                    qty = 0
                try:
                    avg_px = float(str(row.get(price_col, 0)).replace(",", "").replace("$", "").replace("₹", "")) if price_col else 0
                except Exception:
                    avg_px = 0
                name = str(row.get(name_col, sym)).strip() if name_col else sym
                if qty > 0:
                    holdings.append({"symbol": sym, "name": name[:40], "qty": qty,
                                      "avg_price": round(avg_px, 2), "currency": "$"})
            return holdings, None

        @st.cache_data(ttl=300, show_spinner=False)
        def _analyse_holding(sym):
            """Run full technical + fundamental analysis for one US stock."""
            try:
                f = fetch_stock_features(sym)
                if not f:
                    return None
                enrich_with_fundamentals(f)
                mc = monte_carlo_targets(f)
                ranked = normalise_universe({sym: f})
                bkt = classify_bucket(f, ranked)
                score, bkdn = score_stock(f, ranked, bkt)
                return {"f": f, "mc": mc, "score": score, "bkdn": bkdn, "bucket": bkt}
            except Exception:
                return None

        def _holding_recommendation(score, mc, f):
            """Return (label, color, reason) recommendation for a held stock."""
            if not mc:
                return "HOLD", "#888", "Insufficient data for recommendation."
            ret_3m = mc["3M"]["ret_base"] if mc and "3M" in mc else 0
            prob_pos = mc["3M"].get("prob_positive", 50) if mc and "3M" in mc else 50
            rsi = float(f.get("rsi") or 50)
            above_200 = bool(f.get("above_ma200"))
            if score >= 65 and ret_3m > 5 and prob_pos >= 55:
                return "ADD MORE", "#00c853", f"Strong score ({score:.0f}), MC median +{ret_3m:.1f}%."
            elif score >= 50 and ret_3m > 0 and above_200:
                return "HOLD", "#2979ff", f"Score {score:.0f}, above 200MA, positive MC outlook."
            elif score < 35 or ret_3m < -10:
                return "CONSIDER EXIT", "#ff5252", f"Score {score:.0f}, MC median {ret_3m:.1f}% — review position."
            elif rsi > 75:
                return "TRIM / TAKE PROFITS", "#ff9800", f"RSI {rsi:.0f} — overbought. Consider partial profit-taking."
            else:
                return "HOLD", "#888", f"Score {score:.0f} — monitor. No strong signal to act."


        # ══════════════════════════════════════════════════════════════════════════════
        # HF PORTFOLIO ANALYSIS ENGINE
        # ══════════════════════════════════════════════════════════════════════════════

        # Historical stress scenarios with per-stock estimated impacts
        # Based on actual 2020-2024 returns for each stock in each regime
        _STRESS_SCENARIOS = {
            "🔴 2022 Rate Hike": {
                "desc":   "Fed +500bps, growth multiples compressed. S&P -25%.",
                "market": -0.25,
                "hits": {
                    "NOW": -0.45, "META": -0.64, "MSFT": -0.28, "NVDA": -0.50,
                    "TSM": -0.35, "MELI": -0.70, "IONQ": -0.80, "PATH": -0.70,
                },
                "default": -0.38,
            },
            "💥 AI Bubble Burst": {
                "desc":   "AI hype unwinds. Valuations revert 40-70% from peak. S&P -20%.",
                "market": -0.20,
                "hits": {
                    "NOW": -0.55, "META": -0.35, "MSFT": -0.40, "NVDA": -0.70,
                    "TSM": -0.45, "MELI": -0.20, "IONQ": -0.85, "PATH": -0.60,
                },
                "default": -0.45,
            },
            "⚔️ Taiwan Conflict": {
                "desc":   "China blockade of Taiwan. Semiconductor supply chain halted.",
                "market": -0.15,
                "hits": {
                    "TSM":  -0.80, "NVDA": -0.55, "MSFT": -0.18, "NOW":  -0.12,
                    "META": -0.15, "MELI": -0.10, "IONQ": -0.25, "PATH": -0.15,
                },
                "default": -0.18,
            },
            "💵 Dollar Surge +15%": {
                "desc":   "USD strengthens 15%. Emerging market currencies collapse.",
                "market": -0.08,
                "hits": {
                    "MELI": -0.35, "TSM":  -0.15, "NVDA": -0.08, "MSFT": -0.10,
                    "NOW":  -0.06, "META": -0.10, "IONQ": -0.05, "PATH": -0.08,
                },
                "default": -0.10,
            },
            "🦠 2020 COVID Crash": {
                "desc":   "Pandemic shock. Market -34% in 5 weeks, then rapid recovery.",
                "market": -0.34,
                "hits": {
                    "MELI": +0.10, "MSFT": -0.15, "NOW":  -0.22, "META": -0.28,
                    "NVDA": -0.22, "TSM":  -0.28, "IONQ": -0.55, "PATH": -0.40,
                },
                "default": -0.28,
            },
        }

        # Factor exposure per known ticker (adds up to 1.0)
        _FACTOR_MAP = {
            "NOW":  {"AI / Cloud SaaS": 0.75, "Growth":      0.15, "Momentum":     0.10},
            "META": {"AI / Social":     0.50, "Digital Ads": 0.40, "Growth":       0.10},
            "MSFT": {"AI / Cloud":      0.55, "Enterprise":  0.30, "Value / Qual": 0.15},
            "NVDA": {"AI / Chips":      0.85, "Cyclical":    0.15},
            "TSM":  {"AI / Chips":      0.45, "Geopolitical":0.40, "Cyclical":     0.15},
            "MELI": {"EM Growth":       0.55, "E-commerce":  0.35, "FX / Macro":   0.10},
            "IONQ": {"Quantum":         0.75, "Speculative": 0.25},
            "PATH": {"AI / Enterprise": 0.70, "Growth":      0.20, "Speculative":  0.10},
            "AAPL": {"Consumer Tech":   0.55, "Value / Qual":0.30, "Momentum":     0.15},
            "TSLA": {"EV / Energy":     0.60, "Momentum":    0.25, "Speculative":  0.15},
        }

        # Profit-taking milestone rules (gain% → sell what % of position)
        _PROFIT_RULES = [
            (2.00, 0.50, "2× bagger → sell half, let rest ride risk-free"),
            (1.00, 0.30, "100% gain → take 30% off table, trail stop on rest"),
            (0.50, 0.20, "50% gain → book 20%, tighten stop"),
            (0.25, 0.10, "25% gain → optional trim 10% to rebalance"),
        ]

        def _compute_growth_adjusted_value(f, curr_px):
            """
            For hypergrowth companies (revenue growth >40%), DCF severely undervalues
            because it caps growth at 40% and uses high WACC.

            HF managers use these instead:
              1. PEG ratio  — P/E ÷ growth rate. PEG < 1.0 = cheap. PEG 1-2 = fair.
              2. Forward P/E vs sector — compare to sector median forward P/E
              3. Earnings power value — if growth rate normalises to 20%, what's it worth?

            Returns (fair_value_estimate, method_used, valuation_note, is_hypergrowth)
            """
            rev_g  = _safe_float(f.get("rev_growth")) or 0
            earn_g = _safe_float(f.get("earn_growth")) or 0
            pe     = _safe_float(f.get("pe"))
            fwd_pe = _safe_float(f.get("fwd_pe"))
            peg    = _safe_float(f.get("peg"))
            growth = max(rev_g, earn_g)

            is_hypergrowth = growth > 0.35   # >35% growth = hypergrowth, DCF not reliable

            if not is_hypergrowth:
                return None, "dcf", "", False

            # PEG-based valuation
            # Rule: PEG = 1.0 is "fair" (paying 1x P/E per 1% growth)
            # For hypergrowth: PEG < 1.5 = reasonable, 1.5-2.5 = premium, >2.5 = expensive
            if peg and peg > 0:
                if   peg < 1.0: rating = "CHEAP (PEG < 1.0)";  peg_clr = "#00c853"
                elif peg < 1.5: rating = "FAIR (PEG 1.0–1.5)"; peg_clr = "#2979ff"
                elif peg < 2.5: rating = "PREMIUM (PEG 1.5–2.5)"; peg_clr = "#ff9800"
                else:           rating = "EXPENSIVE (PEG > 2.5)"; peg_clr = "#ff5252"
            elif pe and pe > 0 and growth > 0:
                peg = pe / (growth * 100)   # compute PEG manually
                if   peg < 1.0: rating = "CHEAP (PEG < 1.0)";  peg_clr = "#00c853"
                elif peg < 1.5: rating = "FAIR (PEG 1.0–1.5)"; peg_clr = "#2979ff"
                elif peg < 2.5: rating = "PREMIUM (PEG 1.5–2.5)"; peg_clr = "#ff9800"
                else:           rating = "EXPENSIVE (PEG > 2.5)"; peg_clr = "#ff5252"
            else:
                rating = "INSUFFICIENT DATA"; peg_clr = "#888"

            # Earnings power value: if earnings grow at current rate for 3 years then normalise
            # This is what many growth-focused hedge funds use
            ep_value = None
            if fwd_pe and fwd_pe > 0 and earn_g > 0:
                # Estimate 3-year-forward EPS, then apply a normalised P/E of 25
                curr_eps_est = curr_px / fwd_pe
                eps_3y = curr_eps_est * ((1 + min(earn_g, 0.80)) ** 3)
                ep_value = round(eps_3y * 25, 2)   # 25x terminal P/E (mature growth)

            note = (
                f"⚠️ DCF understates value for hypergrowth stocks (growth {growth*100:.0f}%). "
                f"PEG ratio = {peg:.2f} → **{rating}**. "
                f"DCF is designed for mature businesses (growth <10%), not AI platform companies."
            )

            return ep_value, "earnings_power", note, True


        def _hf_position_decision(h, res):
            """
            Full HF-style per-position decision using the RIGHT valuation model per company type:
            • Hypergrowth (>35% growth): PEG ratio + Earnings Power Value
            • Mature / value: DCF with margin of safety
            • Both: ATR trailing stop, Kelly sizing, profit milestone rules
            """
            if not res:
                return None
            f     = res["f"]
            mc    = res["mc"]
            score = res["score"]

            sym       = h["symbol"]
            avg_price = float(h["avg_price"])
            curr_px   = float(f.get("price", avg_price))
            gain_pct  = (curr_px / avg_price - 1) if avg_price > 0 else 0

            # ── Valuation: choose model based on growth profile ───────────────────
            dcf = compute_dcf(f)
            dcf_intrinsic = dcf["intrinsic"] if dcf else None

            ep_value, val_method, val_note, is_hypergrowth = _compute_growth_adjusted_value(f, curr_px)

            # Use earnings power value for hypergrowth, DCF for mature
            if is_hypergrowth and ep_value and ep_value > 0:
                intrinsic       = ep_value
                val_model_label = f"Earnings Power Value (3Y EPS × 25x P/E)"
                val_warning     = val_note
            elif dcf_intrinsic and dcf_intrinsic > 0:
                intrinsic       = dcf_intrinsic
                val_model_label = "DCF (5-year discounted cash flow)"
                val_warning     = None
            else:
                intrinsic       = None
                val_model_label = "Insufficient data"
                val_warning     = None

            # ── Technical support levels ──────────────────────────────────────────
            ma50  = _safe_float(f.get("sma_50"))  or curr_px
            ma200 = _safe_float(f.get("sma_200")) or curr_px

            # ── Optimal entry price ───────────────────────────────────────────────
            if intrinsic and intrinsic > curr_px:
                # Stock below fair value — buy at current or slight pullback
                optimal_entry = round(curr_px * 0.97, 2)
                entry_basis   = f"Already below {val_model_label} fair value. Buy on any 3% dip."
            elif is_hypergrowth:
                # Hypergrowth: enter on pullback to MA50 (momentum stock — don't chase)
                if ma50 > 0 and ma50 < curr_px:
                    optimal_entry = round(ma50 * 0.99, 2)
                    entry_basis   = f"Pullback to MA50 (${ma50:.2f}) — standard entry for momentum stocks."
                else:
                    optimal_entry = round(curr_px * 0.93, 2)
                    entry_basis   = "7% pullback from current — wait for a dip before adding."
            else:
                if intrinsic:
                    optimal_entry = round(intrinsic * 0.80, 2)
                    entry_basis   = f"Graham 20% margin of safety below DCF (${intrinsic:.2f})."
                else:
                    optimal_entry = round(curr_px * 0.92, 2)
                    entry_basis   = "8% pullback from current (conservative entry)."

            # ── Profit target ─────────────────────────────────────────────────────
            mc1y_p90 = mc["1Y"]["price_p90"] if (mc and "1Y" in mc) else curr_px * 1.20
            mc3m_p90 = mc["3M"]["price_p90"] if (mc and "3M" in mc) else curr_px * 1.15

            if intrinsic and intrinsic > curr_px * 1.10:
                profit_target = round(intrinsic, 2)
                target_basis  = f"{val_model_label} fair value"
            else:
                profit_target = round(mc1y_p90, 2)
                target_basis  = "MC P90 1-year bull case"

            # ── Trailing stop ─────────────────────────────────────────────────────
            atr_pct = float(f.get("atr_pct") or 2.0) / 100
            # Hypergrowth stocks are volatile — use wider stop (3× ATR)
            stop_mult  = 3.0 if is_hypergrowth else 2.5
            trail_stop = max(
                round(curr_px * (1 - stop_mult * atr_pct), 2),
                round(avg_price * 0.90, 2),   # max 10% loss from cost for hypergrowth
            )
            stop_basis = f"{stop_mult:.0f}× ATR ({atr_pct*100:.1f}%) trailing — {'wider stop for volatile growth stock' if is_hypergrowth else 'standard stop'}"

            # ── Profit milestone rules ────────────────────────────────────────────
            sell_pct = 0
            sell_rule = None
            for thr, pct, reason in _PROFIT_RULES:
                if gain_pct >= thr:
                    sell_pct  = pct
                    sell_rule = reason
                    break

            # ── Main action signal ────────────────────────────────────────────────
            rsi         = float(f.get("rsi") or 50)
            above_ma200 = bool(f.get("above_ma200"))
            above_ma50  = bool(f.get("above_ma50"))
            prob_pos    = mc["3M"].get("prob_positive", 50) if (mc and "3M" in mc) else 50
            peg_val     = _safe_float(f.get("peg")) or 999

            if is_hypergrowth:
                # For hypergrowth, use PEG + momentum instead of DCF
                if peg_val < 1.0 and above_ma50 and score >= 55:
                    action     = "STRONG BUY MORE"
                    action_clr = "#00c853"
                    action_msg = f"PEG {peg_val:.2f} — cheap relative to growth. Trend is up. Add on pullback to ${optimal_entry:.2f}."
                elif peg_val < 1.5 and above_ma200 and score >= 45:
                    action     = "ADD ON PULLBACK"
                    action_clr = "#2979ff"
                    action_msg = f"PEG {peg_val:.2f} — fair value for growth rate. Entry target: ${optimal_entry:.2f}."
                elif sell_rule and rsi > 72:
                    action     = "TAKE PARTIAL PROFIT"
                    action_clr = "#ff9800"
                    action_msg = f"{sell_rule}. Overbought (RSI {rsi:.0f}). Sell {sell_pct*100:.0f}%, trail stop rest."
                elif peg_val > 3.0 and rsi > 70 and gain_pct > 0.30:
                    action     = "TRIM — VERY EXPENSIVE"
                    action_clr = "#ff9800"
                    action_msg = f"PEG {peg_val:.2f} > 3.0 and RSI overbought. Consider trimming 20-25%."
                elif gain_pct < -0.20 and not above_ma200:
                    action     = "REVIEW THESIS"
                    action_clr = "#ff5252"
                    action_msg = f"Down {gain_pct*100:.0f}% and below 200MA. Is the growth thesis still intact?"
                else:
                    action     = "HOLD"
                    action_clr = "#888"
                    action_msg = f"Trend intact, valuation reasonable. Trail stop at ${trail_stop:.2f}. Add at ${optimal_entry:.2f} dip."
            else:
                # DCF-based logic for mature companies
                if intrinsic and curr_px < intrinsic * 0.80 and score >= 50:
                    action     = "STRONG BUY MORE"
                    action_clr = "#00c853"
                    action_msg = f"Trading at {curr_px/intrinsic*100:.0f}% of DCF fair value. Strong margin of safety."
                elif intrinsic and curr_px < intrinsic * 0.93 and score >= 45 and prob_pos >= 55:
                    action     = "ADD ON PULLBACK"
                    action_clr = "#2979ff"
                    action_msg = f"Below intrinsic. Buy at ${optimal_entry:.2f} for better entry."
                elif sell_rule and (rsi > 72 or (intrinsic and curr_px > intrinsic * 1.15)):
                    action     = "TAKE PARTIAL PROFIT"
                    action_clr = "#ff9800"
                    action_msg = f"{sell_rule}. Sell {sell_pct*100:.0f}%, trail stop on rest."
                elif intrinsic and curr_px > intrinsic * 1.25:
                    action     = "TRIM — ABOVE FAIR VALUE"
                    action_clr = "#ff9800"
                    action_msg = f"Trading {(curr_px/intrinsic-1)*100:.0f}% above DCF value. Consider trimming."
                elif gain_pct < -0.20 and not above_ma200 and score < 40:
                    action     = "CUT LOSS / REVIEW"
                    action_clr = "#ff1744"
                    action_msg = f"Down {gain_pct*100:.0f}%, below 200MA, weak score. Review thesis urgently."
                else:
                    action     = "HOLD"
                    action_clr = "#888"
                    action_msg = f"No strong signal. Monitor stop at ${trail_stop:.2f}."

            # ── Kelly sizing ──────────────────────────────────────────────────────
            p             = prob_pos / 100
            win           = max((profit_target - curr_px) / curr_px, 0.01)
            los           = max((curr_px - trail_stop)    / curr_px, 0.01)
            b             = win / los
            kelly_full    = max(0, (p * b - (1 - p)) / b)
            kelly_quarter = min(kelly_full * 0.25, 0.12)

            return {
                "action":          action,
                "action_clr":      action_clr,
                "action_msg":      action_msg,
                "curr_px":         curr_px,
                "avg_price":       avg_price,
                "gain_pct":        gain_pct,
                "optimal_entry":   optimal_entry,
                "entry_basis":     entry_basis,
                "profit_target":   profit_target,
                "target_basis":    target_basis,
                "trail_stop":      trail_stop,
                "stop_basis":      stop_basis,
                "intrinsic":       intrinsic,
                "val_model_label": val_model_label,
                "val_warning":     val_warning,
                "is_hypergrowth":  is_hypergrowth,
                "sell_pct":        sell_pct,
                "sell_rule":       sell_rule,
                "kelly_pct":       round(kelly_quarter * 100, 1),
                "dcf":             dcf,
                "dcf_intrinsic":   dcf_intrinsic,
            }


        def _hf_stress_test(holdings, analysis_results):
            """Return {scenario_name: {portfolio_loss, per_stock}} for 5 scenarios."""
            results = {}
            for name, cfg in _STRESS_SCENARIOS.items():
                per_stock = {}
                total_loss = 0.0
                total_val  = 0.0
                for h in holdings:
                    sym = h["symbol"]
                    res = analysis_results.get(sym)
                    px  = float(res["f"]["price"]) if res else float(h["avg_price"])
                    val = float(h["qty"]) * px
                    impact = cfg["hits"].get(sym, cfg["default"])
                    loss   = val * impact
                    per_stock[sym] = {"impact": impact, "loss": loss, "val": val}
                    total_loss += loss
                    total_val  += val
                results[name] = {
                    "desc":       cfg["desc"],
                    "total_loss": total_loss,
                    "total_val":  total_val,
                    "pct":        total_loss / max(total_val, 1),
                    "per_stock":  per_stock,
                }
            return results


        def _hf_risk_attribution(holdings, analysis_results):
            """
            Compute each position's contribution to total portfolio risk.
            Risk proxy = position value × annualised volatility (hist_std × sqrt(252))
            """
            rows = []
            for h in holdings:
                sym = h["symbol"]
                res = analysis_results.get(sym)
                if not res:
                    continue
                f    = res["f"]
                px   = float(f.get("price", h["avg_price"]))
                val  = float(h["qty"]) * px
                sig  = float(f.get("hist_std") or 0.015)
                ann  = sig * np.sqrt(252)          # annualised vol
                beta = float(f.get("beta") or 1.0)
                risk = val * ann                    # dollar risk
                rows.append({"sym": sym, "val": val, "vol": ann, "beta": beta, "risk": risk})
            total_risk = sum(r["risk"] for r in rows) or 1
            total_val  = sum(r["val"]  for r in rows) or 1
            for r in rows:
                r["risk_pct"] = r["risk"] / total_risk
                r["val_pct"]  = r["val"]  / total_val
            return sorted(rows, key=lambda x: x["risk_pct"], reverse=True)


        def _hf_factor_exposure(holdings, analysis_results):
            """Aggregate factor exposures across the portfolio (capital-weighted)."""
            factor_dollars = {}
            total_val = 0
            for h in holdings:
                sym = h["symbol"]
                res = analysis_results.get(sym)
                px  = float(res["f"]["price"]) if res else float(h["avg_price"])
                val = float(h["qty"]) * px
                total_val += val
                factors = _FACTOR_MAP.get(sym, {"Uncategorised": 1.0})
                for factor, wt in factors.items():
                    factor_dollars[factor] = factor_dollars.get(factor, 0) + val * wt
            return {f: v / total_val for f, v in sorted(factor_dollars.items(), key=lambda x: -x[1])}


        def _hf_health_score(holdings, analysis_results, risk_rows, factor_exp):
            """
            0-100 portfolio health score across 4 dimensions:
              Diversification (25), Risk Management (25), Conviction Quality (25), Valuation (25)
            """
            n = len(holdings)
            # 1. Diversification (penalise concentration)
            max_risk_pct   = max((r["risk_pct"] for r in risk_rows), default=1)
            n_factors      = len(factor_exp)
            top_factor_pct = max(factor_exp.values(), default=1)
            div_score      = int(min(25, (1 - max_risk_pct) * 20 + min(n, 8) * 1.5 + (1 - top_factor_pct) * 10))

            # 2. Risk management (stops, not too many losers)
            avg_score   = np.mean([analysis_results[h["symbol"]]["score"]
                                   for h in holdings if analysis_results.get(h["symbol"])]) if holdings else 50
            losers      = sum(1 for h in holdings
                              if analysis_results.get(h["symbol"]) and
                                 float(analysis_results[h["symbol"]]["f"].get("price", 0)) < float(h["avg_price"]) * 0.85)
            risk_score  = int(min(25, avg_score * 0.20 + max(0, 25 - losers * 5)))

            # 3. Conviction quality (how many positions have strong DCF support)
            dcf_ok = sum(1 for h in holdings
                         if analysis_results.get(h["symbol"]) and
                            compute_dcf(analysis_results[h["symbol"]]["f"]) and
                            compute_dcf(analysis_results[h["symbol"]]["f"])["upside_pct"] > 0)
            conv_score = int(min(25, dcf_ok / max(n, 1) * 25))

            # 4. Valuation (% of portfolio trading below intrinsic)
            below_intrinsic = 0
            for h in holdings:
                res = analysis_results.get(h["symbol"])
                if not res: continue
                dcf = compute_dcf(res["f"])
                if dcf and float(res["f"].get("price", 0)) < dcf["intrinsic"]:
                    below_intrinsic += 1
            val_score = int(below_intrinsic / max(n, 1) * 25)

            total = div_score + risk_score + conv_score + val_score
            return {
                "total":         total,
                "diversification": div_score,
                "risk_mgmt":     risk_score,
                "conviction":    conv_score,
                "valuation":     val_score,
            }



        with _fs_t3:
            st.markdown(
                "> *'The best time to invest is when a company is too small for institutions to notice, "
                "too boring for media to cover, and too early for most investors to understand.'*"
            )
            st.caption(
                "This tab is the opposite of the momentum screener. It finds companies with "
                "solid fundamentals in emerging industries — before the crowd discovers them. "
                "Low analyst coverage, flat stock price, high insider ownership = the sweet spot."
            )

            # ── Industry Selection ────────────────────────────────────────────────────
            st.markdown("### 🗺️ Step 1 — Choose Your Emerging Industry")
            st.caption("Pick an industry you believe will be mainstream in 3-5 years but isn't yet.")

            # Show industry cards in a grid
            industry_names = list(EMERGING_INDUSTRIES.keys())
            n_cols = 3
            rows   = [industry_names[i:i+n_cols] for i in range(0, len(industry_names), n_cols)]

            for row in rows:
                cols = st.columns(len(row))
                for col, ind_name in zip(cols, row):
                    cfg = EMERGING_INDUSTRIES[ind_name]
                    with col:
                        st.markdown(
                            f"<div style='background:#0d0d0d;border:1px solid #222;border-radius:8px;"
                            f"padding:10px 12px;margin-bottom:8px;min-height:100px'>"
                            f"<div style='font-size:0.95rem;font-weight:700;color:#fff;margin-bottom:4px'>{ind_name}</div>"
                            f"<div style='font-size:0.72rem;color:#666;margin-bottom:6px'>{cfg['horizon']} horizon</div>"
                            f"<div style='font-size:0.75rem;color:#aaa'>{cfg['tam']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            sel_col1, sel_col2 = st.columns([2, 1])
            with sel_col1:
                selected_industry = st.selectbox(
                    "Select Industry to Scan",
                    industry_names,
                    help="Choose the emerging industry you want to find hidden gems in.",
                    key="gem_industry_select",
                )
            with sel_col2:
                custom_industry_name = ""
                if selected_industry == "🔭 Custom Industry":
                    custom_industry_name = st.text_input("Enter industry name", placeholder="e.g. Holographic Display")

            ind_cfg = EMERGING_INDUSTRIES[selected_industry]

            # Show industry context
            display_name = custom_industry_name or selected_industry
            st.markdown(
                f"<div style='background:#0a0a1a;border:1px solid #2979ff33;border-radius:8px;"
                f"padding:14px 18px;margin-bottom:16px'>"
                f"<div style='font-size:1rem;font-weight:700;color:#fff;margin-bottom:8px'>{display_name}</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.82rem'>"
                f"<div><span style='color:#666'>Why it's overlooked:</span><br>"
                f"<span style='color:#aaa'>{ind_cfg.get('boring_factor','')}</span></div>"
                f"<div><span style='color:#666'>TAM:</span><br>"
                f"<span style='color:#00c853'>{ind_cfg.get('tam','')}</span></div>"
                f"<div><span style='color:#666'>Key catalyst to watch:</span><br>"
                f"<span style='color:#ffcc00'>{ind_cfg.get('catalyst','')}</span></div>"
                f"<div><span style='color:#666'>Investment horizon:</span><br>"
                f"<span style='color:#2979ff'>{ind_cfg.get('horizon','')}</span></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # ── Scan Configuration ────────────────────────────────────────────────────
            st.markdown("### ⚙️ Step 2 — Configure Your Scan")
            cfg1, cfg2, cfg3 = st.columns(3)
            with cfg1:
                gem_market = st.radio("Market", ["🇺🇸 US", "🇮🇳 India", "Both"], horizontal=True, key="gem_market_select")
            with cfg2:
                max_mktcap_opt = st.selectbox(
                    "Max Market Cap (hidden gems only)",
                    ["$500M (micro-cap)", "$1B (small-cap)", "$2B (small/mid)", "$5B (mid-cap)", "Any size"],
                    index=2, key="gem_cap_select",
                )
                cap_map = {"$500M (micro-cap)": 500e6, "$1B (small-cap)": 1e9,
                           "$2B (small/mid)": 2e9, "$5B (mid-cap)": 5e9, "Any size": 1e15}
                max_cap = cap_map[max_mktcap_opt]
            with cfg3:
                max_analysts = st.slider("Max Analyst Coverage", 0, 15, 5,
                    help="0-5 analysts = undiscovered. Keep this low.", key="gem_ana_select")
                use_ai_discovery = st.checkbox("🤖 Use AI to find more companies",
                    value=True, help="Uses Gemini to discover lesser-known companies beyond the seed list.")

            # ── Scan Button ───────────────────────────────────────────────────────────
            st.markdown("---")
            if st.button("🔍 Find Hidden Gems", type="primary", use_container_width=True, key="gem_scan"):
                api_key    = os.getenv("GEMINI_API_KEY", "")
                mkt_hint   = ("us" if "US" in gem_market else "india" if "India" in gem_market else "both")
                seed_syms  = []

                if "US" in gem_market or "Both" in gem_market:
                    seed_syms += ind_cfg.get("seeds_us", [])
                if "India" in gem_market or "Both" in gem_market:
                    seed_syms += ind_cfg.get("seeds_in", [])

                # AI discovery
                ai_found = []
                if use_ai_discovery and api_key:
                    with st.spinner("🤖 AI scanning for undiscovered companies in this industry…"):
                        query = custom_industry_name if custom_industry_name else selected_industry
                        ai_found = discover_industry_stocks_ai(query, ind_cfg, mkt_hint, api_key)
                    st.caption(f"AI discovered {len(ai_found)} additional companies to evaluate.")

                # Combine seeds + AI discoveries
                all_syms = list(dict.fromkeys(seed_syms + [d["symbol"] for d in ai_found]))

                if not all_syms:
                    st.warning("No companies to scan. Add seeds or enable AI discovery.")
                else:
                    st.info(f"Scanning {len(all_syms)} companies: {', '.join(all_syms[:15])}{'…' if len(all_syms) > 15 else ''}")

                    # Score each company
                    progress = st.progress(0)
                    status   = st.empty()
                    results  = []

                    for i, sym in enumerate(all_syms):
                        status.text(f"Analysing {sym} ({i+1}/{len(all_syms)})…")
                        gem = compute_hidden_gem_score(sym)
                        if gem:
                            # Apply filters
                            if gem["mktcap"] > max_cap:
                                progress.progress((i+1)/len(all_syms))
                                continue
                            if gem["n_ana"] > max_analysts:
                                progress.progress((i+1)/len(all_syms))
                                continue
                            gem["symbol"] = sym   # always store ticker
                            ai_reason = next((d["reason"] for d in ai_found if d["symbol"] == sym), "")
                            gem["ai_reason"] = ai_reason
                            results.append(gem)
                        progress.progress((i+1)/len(all_syms))

                    progress.empty()
                    status.empty()

                    results.sort(key=lambda x: x["score"], reverse=True)
                    st.session_state["gem_scan_results"]   = results
                    st.session_state["gem_scan_industry"]  = display_name
                    st.session_state["gem_scan_cfg"]       = ind_cfg

            # ── Results ───────────────────────────────────────────────────────────────
            gem_results = st.session_state.get("gem_scan_results",  [])
            gem_ind     = st.session_state.get("gem_scan_industry", "")
            gem_cfg     = st.session_state.get("gem_scan_cfg",      {})

            if gem_results:
                st.markdown(f"---\n### 💎 {len(gem_results)} Hidden Gems Found in {gem_ind}")
                st.caption(
                    "Sorted by Hidden Gem Score. Higher = more undiscovered + better fundamentals + flat stock price. "
                    "Green badge = strong early-stage opportunity."
                )

                for gem in gem_results:
                    sym      = gem.get("symbol", "UNKNOWN")
                    sym_idx  = gem_results.index(gem)
                    # We need to track the symbol — add it during scoring
                    score    = gem["score"]
                    score_clr = "#00c853" if score >= 60 else ("#ffcc00" if score >= 40 else "#888")
                    mom_6m   = gem["mom_6m"]
                    mom_clr  = "#00c853" if mom_6m > 0.05 else ("#ff5252" if mom_6m < -0.15 else "#888")
                    timing_lbl = ("⚡ NOT YET DISCOVERED" if gem["timing"] >= 12 else
                                  "📈 Starting to move" if gem["timing"] >= 6 else "🏃 Already running")

                    with st.expander(
                        f"**{gem['name']}** · Score {score:.0f}/100 · {timing_lbl} · {mom_6m*100:+.1f}% (6M)",
                        expanded=(score >= 65)
                    ):
                        # Header metrics
                        hm1, hm2, hm3, hm4, hm5, hm6 = st.columns(6)
                        hm1.metric("Price",         f"${gem['price']:.2f}")
                        hm2.metric("Market Cap",    f"${gem['mktcap']/1e6:.0f}M" if gem["mktcap"] < 1e9 else f"${gem['mktcap']/1e9:.1f}B")
                        hm3.metric("Analysts",      str(gem["n_ana"]),
                                    help="Fewer = more undiscovered. 0-3 is ideal.")
                        hm4.metric("Rev Growth",    f"{gem['rev_g']*100:+.0f}%")
                        hm5.metric("Gross Margin",  f"{gem['gross_m']*100:.0f}%")
                        hm6.metric("Insider Own.",  f"{gem['insider']*100:.1f}%",
                                    help=">10% = management has skin in the game.")

                        # Hidden Gem Score breakdown
                        st.markdown(
                            f"<div style='background:#0d0d0d;border:1px solid {score_clr}33;"
                            f"border-radius:8px;padding:10px 16px;margin:8px 0;"
                            f"display:flex;align-items:center;gap:20px;flex-wrap:wrap'>"
                            f"<span style='color:{score_clr};font-weight:800;font-size:1.1rem'>"
                            f"Hidden Gem Score: {score:.0f}/100</span>"
                            f"<span style='color:#555'>|</span>"
                            f"<span style='color:#888;font-size:0.8rem'>"
                            f"Undiscovery {gem['undis']}/35 · "
                            f"Fundamentals {gem['fund']}/30 · "
                            f"Timing {gem['timing']}/25 · "
                            f"Mgmt {gem['mgmt']}/10"
                            f"</span></div>",
                            unsafe_allow_html=True,
                        )

                        if gem.get("ai_reason"):
                            st.caption(f"🤖 AI insight: {gem['ai_reason']}")

                        if gem.get("summary"):
                            st.caption(f"📋 {gem['summary'][:300]}…")

                        # Score signal breakdown
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.markdown("**🔍 Why It's Undiscovered**")
                            if gem["n_ana"] == 0:
                                st.success("✅ Zero analyst coverage — completely off Wall Street's radar.")
                            elif gem["n_ana"] <= 3:
                                st.success(f"✅ Only {gem['n_ana']} analyst(s) covering this. Most institutions can't buy (too small).")
                            else:
                                st.info(f"📊 {gem['n_ana']} analysts. Growing but still underfollowed.")
                            if gem["mktcap"] < 500e6:
                                st.info("💡 Micro-cap: most funds can't invest due to liquidity constraints — this is your edge.")
                            st.metric("6M Price Change", f"{mom_6m*100:+.1f}%",
                                       help="Flat or slightly negative = not yet discovered by momentum investors.")

                        with sc2:
                            st.markdown("**📊 Fundamental Quality**")
                            # Revenue growth signal
                            if gem["rev_g"] > 0.30:
                                st.success(f"✅ Revenue growing {gem['rev_g']*100:.0f}%/yr — high growth, undiscovered.")
                            elif gem["rev_g"] > 0.10:
                                st.info(f"📈 Revenue growing {gem['rev_g']*100:.0f}%/yr — steady.")
                            else:
                                st.warning(f"⚠️ Revenue growth {gem['rev_g']*100:.0f}%/yr — low. Check if pre-revenue.")
                            # Gross margin
                            if gem["gross_m"] > 0.50:
                                st.success(f"✅ {gem['gross_m']*100:.0f}% gross margin — software-like economics.")
                            elif gem["gross_m"] > 0.25:
                                st.info(f"📊 {gem['gross_m']*100:.0f}% gross margin — acceptable.")
                            # Cash strength
                            if gem["fcf_yield"] and gem["fcf_yield"] > 0.02:
                                st.success(f"✅ FCF positive ({gem['fcf_yield']*100:.1f}% yield) — self-funding.")
                            elif gem["curr_r"] > 2:
                                st.info(f"💰 Current ratio {gem['curr_r']:.1f}x — good cash position.")

                        with sc3:
                            st.markdown("**🎯 When to Invest**")
                            # Entry timing
                            if gem["timing"] >= 15:
                                st.success("✅ **INVEST NOW** — Stock is flat/sideways despite good fundamentals. This is the ideal early entry window.")
                            elif gem["timing"] >= 8:
                                st.info("📊 **WATCH & BUILD** — Stock is early stage. Start a small position, add on confirmation.")
                            else:
                                st.warning("⚠️ **WAIT** — Stock may already be moving. Wait for a pullback.")
                            # Position size guidance
                            if score >= 65:
                                st.markdown("**Position sizing:** 2-5% of portfolio. These are high-conviction but high-risk early bets.")
                            elif score >= 45:
                                st.markdown("**Position sizing:** 0.5-2% of portfolio. Small exploratory position.")
                            else:
                                st.markdown("**Position sizing:** Track only — not enough conviction yet.")

                        # 10x potential analysis
                        st.markdown("**🚀 What Needs to Happen for 10× Return**")
                        tenx = compute_10x_potential(gem)
                        for line in tenx:
                            st.markdown(f"- {line}")

                        # Key risk
                        st.markdown("**⚠️ Key Risks**")
                        risks = []
                        if gem["mktcap"] < 200e6:
                            risks.append("Micro-cap liquidity risk — hard to exit large positions quickly.")
                        if gem["rev_g"] < 0:
                            risks.append("Revenue declining — business model may not be working.")
                        if gem["de"] and gem["de"] > 100:
                            risks.append(f"High debt (D/E {gem['de']:.0f}) — could face solvency issues if growth slows.")
                        if gem["curr_r"] < 1:
                            risks.append("Current ratio < 1 — may need to raise capital (dilution risk).")
                        risks.append(f"Main sector risk: {gem_cfg.get('boring_factor', 'Industry-specific execution risk.')}")
                        for risk in risks[:4]:
                            st.markdown(f"- ⚠️ {risk}")

                        # Watchlist add
                        if st.button(f"➕ Add {gem['name'][:20]} to Watchlist", key=f"gem_wl_{sym_idx}"):
                            st.info("Use the main Stock Analysis tab to add this ticker to your watchlist.")

                # Summary table
                st.markdown("---")
                st.markdown("### 📋 Quick Comparison Table")
                tbl_rows = []
                for g in gem_results:
                    tbl_rows.append({
                        "Name":          g["name"][:25],
                        "Score":         f"{g['score']:.0f}/100",
                        "Mkt Cap":       f"${g['mktcap']/1e6:.0f}M" if g["mktcap"] < 1e9 else f"${g['mktcap']/1e9:.1f}B",
                        "Analysts":      g["n_ana"],
                        "Rev Growth":    f"{g['rev_g']*100:+.0f}%",
                        "Gross Margin":  f"{g['gross_m']*100:.0f}%",
                        "Insider Own.":  f"{g['insider']*100:.1f}%",
                        "6M Price":      f"{g['mom_6m']*100:+.1f}%",
                        "Undiscovery":   f"{g['undis']}/35",
                        "Timing":        f"{g['timing']}/25",
                        "Signal":        ("INVEST NOW" if g["timing"] >= 15 and g["score"] >= 60 else
                                          "WATCH" if g["score"] >= 40 else "MONITOR"),
                    })
                if tbl_rows:
                    df_gem = pd.DataFrame(tbl_rows)
                    def _style_signal(val):
                        c = {"INVEST NOW": "#00c853", "WATCH": "#2979ff", "MONITOR": "#888"}.get(str(val), "#888")
                        return f"color:{c};font-weight:700"
                    st.dataframe(
                        df_gem.style.map(_style_signal, subset=["Signal"]),
                        use_container_width=True, hide_index=True
                    )

            elif "gem_scan_results" not in st.session_state:
                st.markdown("---")
                st.info(
                    "👆 Select an emerging industry above and click **🔍 Find Hidden Gems** to discover "
                    "undiscovered early-stage companies.",
                    icon="🌱",
                )

            st.markdown("---")
            st.caption(
                "⚠️ Early-stage investing carries significant risk. These companies are small, illiquid, "
                "and many will fail. Size positions at 0.5-5% of portfolio maximum. "
                "This is NOT financial advice."
            )


            st.markdown("## ⚡ Alpha Hunter — High-Velocity Strategy Lab")
            st.caption(
                "Tools for finding setups that can move 30 %+ in a month: "
                "earnings catalysts, technical breakouts, short squeezes, and options flow. "
                "Paper-trade first — see what actually works before risking real money."
            )

            # ── Reality-check panel ───────────────────────────────────────────────────
            with st.expander("📊 What 30 %/month really means — read this first", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("""
        **30 %/month = 2,230 % annualised.**

        Even the world's greatest traders:
        - Warren Buffett: ~20 %/yr over 60 years
        - Renaissance Medallion (best hedge fund ever): ~66 %/yr
        - Peter Lynch at Magellan: ~29 %/yr

        **No model can reliably predict short-term price movement.**
        Monte Carlo, XGBoost, DCF — they show *probability distributions*, not certainties.

        What CAN produce 30 %+ moves in a month:
        1. 🎯 **Earnings surprise** — stock jumps 20-40 % in one day
        2. 🚀 **Momentum breakout** — stock runs 30-100 % over 2-4 weeks
        3. 🌪️ **Short squeeze** — forced buying sends stock up 50-200 %
        4. 📈 **Options leverage** — a 10 % stock move becomes 100 %+ on calls

        **Win rate on these setups: 20-35 %.**
        That means 2-3 out of every 10 trades will actually pay off.
        The key is: *size positions so the 3 winners more than cover the 7 losers.*
        """)
                with c2:
                    st.markdown("""
        **Suggested position sizing for these plays:**

        | Setup | Max position size |
        |---|---|
        | Earnings play | 3-5 % of portfolio |
        | Breakout momentum | 5-8 % of portfolio |
        | Short squeeze | 2-3 % of portfolio |
        | Options (calls/puts) | 1-2 % of portfolio |

        **Stop loss rule:** Exit if position falls 15 % from entry.
        This means on a 3 % position, max loss per trade = 0.45 % of portfolio.

        **The math that makes this work:**
        - 10 trades × 3 % position = 30 % deployed
        - 3 winners × 30 % gain = +2.7 % portfolio
        - 7 losers × 15 % loss = -3.15 % portfolio
        - **Net = -0.45 % per cycle if you have no edge at all**

        With a real catalyst edge, winners can be 50-100 %, changing the math completely.
        """)
                st.warning(
                    "⚠️ **Never invest based solely on this dashboard.** "
                    "Paper-trade for at least 2-4 weeks first. "
                    "Track your real win rate. Only go live when paper trading is consistently profitable."
                )

            # ── Sub-tabs ──────────────────────────────────────────────────────────────
            _fs_t4, _fs_t5, _fs_t6, _fs_t7 = st.tabs([
                "🎯 Catalyst Scanner",
                "🚀 Breakout Scanner",
                "🌪️ Short Squeeze Radar",
                "📝 Paper Trading",
            ])

            # ─────────────────────────────────────────────────────────────────────────
            # AH TAB 1 — Catalyst Scanner (Earnings plays)
        # ─────────────────────────────────────────────────────────────────────────
        with _fs_t4:
            st.markdown("### 🎯 Earnings Catalyst Scanner")
            st.markdown(
                "Finds stocks with **upcoming earnings** and historically **large post-earnings moves**. "
                "Stocks that regularly move ≥15 % on earnings day are prime targets for catalyst plays."
            )

            col_a, col_b = st.columns([2, 1])
            with col_a:
                cat_input = st.text_area(
                    "Stocks to scan (comma-separated tickers)",
                    value="NVDA, META, MSFT, NOW, TSM, MELI, AAPL, AMZN, GOOGL, TSLA, AMD, CRWD, SNOW, DDOG, PLTR, IONQ, SMCI, ARM, MU, AVGO",
                    height=80,
                    key="cat_input",
                )
            with col_b:
                days_window = st.slider("Earnings within next N days", 7, 60, 21, key="cat_days")
                min_hist_move = st.slider("Min avg historical move (%)", 5, 30, 10, key="cat_min_move")

            if st.button("🔍 Scan for Upcoming Earnings", key="cat_scan_btn", type="primary"):
                syms = [s.strip().upper() for s in cat_input.split(",") if s.strip()]
                prog = st.progress(0, text="Scanning earnings calendars…")
                cat_results = []
                for i, sym in enumerate(syms):
                    prog.progress((i + 1) / len(syms), text=f"Checking {sym}…")
                    ei = _get_earnings_info(sym)
                    if ei["days_to_earn"] is not None and 0 <= ei["days_to_earn"] <= days_window:
                        cat_results.append({"sym": sym, **ei})
                prog.empty()
                st.session_state["cat_results"] = cat_results

            if "cat_results" in st.session_state and st.session_state["cat_results"]:
                cat_results = st.session_state["cat_results"]
                # Filter by min move
                filtered = [r for r in cat_results if r["avg_earn_move"] * 100 >= min_hist_move]
                if not filtered:
                    filtered = cat_results   # show all if none pass filter

                filtered.sort(key=lambda x: x["avg_earn_move"], reverse=True)

                st.markdown(f"#### {len(filtered)} stocks with upcoming earnings (within {days_window} days)")

                for r in filtered:
                    move_pct  = r["avg_earn_move"] * 100
                    urgency   = "🔴" if (r["days_to_earn"] or 99) <= 7 else ("🟡" if (r["days_to_earn"] or 99) <= 14 else "🟢")
                    move_col  = "#00c853" if move_pct >= 15 else ("#ff9100" if move_pct >= 10 else "#aaa")

                    with st.container():
                        st.markdown(
                            f"""<div style="background:#111;border:1px solid #222;border-radius:8px;
                            padding:12px 16px;margin-bottom:10px;display:flex;
                            align-items:center;gap:20px;flex-wrap:wrap">
                              <div style="min-width:80px">
                                <span style="color:#fff;font-weight:700;font-size:1.1rem">{r['sym']}</span>
                              </div>
                              <div>
                                <div style="color:#888;font-size:0.75rem">Earnings Date</div>
                                <div style="color:#fff;font-weight:600">{urgency} {r['next_earn']}</div>
                              </div>
                              <div>
                                <div style="color:#888;font-size:0.75rem">Days Away</div>
                                <div style="color:#fff;font-weight:600">{r['days_to_earn']} days</div>
                              </div>
                              <div>
                                <div style="color:#888;font-size:0.75rem">Avg Historical Move</div>
                                <div style="color:{move_col};font-weight:700;font-size:1.05rem">
                                  ±{move_pct:.1f}%</div>
                              </div>
                              <div>
                                <div style="color:#888;font-size:0.75rem">Samples</div>
                                <div style="color:#aaa">{r['n_samples']} earnings</div>
                              </div>
                              <div style="margin-left:auto">
                                {'<span style="color:#00c853;font-weight:700">⭐ HIGH CONVICTION</span>'
                                  if move_pct >= 15 and (r["days_to_earn"] or 99) <= 14
                                  else '<span style="color:#888">WATCH</span>'}
                              </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                st.markdown("""
    **How to trade earnings catalysts:**
    - Buy **before** earnings if you expect a beat (but you're guessing — 50/50 without edge)
    - **Safer approach**: wait for earnings, then buy the breakout if reaction is strongly positive
    - **Options approach**: buy a straddle (call + put) before earnings to profit from the *size* of the move regardless of direction
    - **Never hold through earnings** with a large position — the stock can gap down 20 %+ even on decent results
    """)

            elif "cat_results" in st.session_state and not st.session_state["cat_results"]:
                st.info("No upcoming earnings found in the selected window. Try widening the date range or adding more tickers.")

        # ─────────────────────────────────────────────────────────────────────────
        # AH TAB 2 — Breakout Scanner
        # ─────────────────────────────────────────────────────────────────────────
        with _fs_t5:
            st.markdown("### 🚀 Breakout Momentum Scanner")
            st.markdown(
                "Finds stocks in **technical breakout** setups — near 52-week highs, "
                "surging volume, and strong RSI momentum. These are the stocks most likely "
                "to continue running 20-50 % over the next 2-4 weeks."
            )

            col_ba, col_bb = st.columns([2, 1])
            with col_ba:
                bo_input = st.text_area(
                    "Stocks to scan (comma-separated)",
                    value="NVDA, META, MSFT, NOW, AAPL, AMZN, GOOGL, TSLA, AMD, AVGO, "
                          "CRWD, PLTR, DDOG, SNOW, ARM, MU, SMCI, NET, CRM, PANW, "
                          "UBER, SHOP, COIN, MELI, TSM, MRVL, LRCX, AMAT, KLAC, TXN, "
                          "IONQ, PATH, GTLB, BILL, FRSH, MNDY, ZS, OKTA, S, FTNT",
                    height=100,
                    key="bo_input",
                )
            with col_bb:
                min_bo_score = st.slider("Min breakout score", 30, 90, 50, key="bo_min_score")
                bo_top_n     = st.slider("Show top N stocks", 5, 30, 15, key="bo_top_n")

            if st.button("🔍 Scan Breakouts", key="bo_scan_btn", type="primary"):
                syms = [s.strip().upper() for s in bo_input.split(",") if s.strip()]
                prog = st.progress(0, text="Scanning breakout setups…")
                bo_results = []
                for i, sym in enumerate(syms):
                    prog.progress((i + 1) / len(syms), text=f"Scanning {sym}…")
                    r = _get_breakout_data(sym)
                    if r and r["score"] >= min_bo_score:
                        bo_results.append(r)
                prog.empty()
                bo_results.sort(key=lambda x: x["score"], reverse=True)
                st.session_state["bo_results"] = bo_results[:bo_top_n]

            if "bo_results" in st.session_state and st.session_state["bo_results"]:
                bo_results = st.session_state["bo_results"]
                st.markdown(f"#### Top {len(bo_results)} Breakout Candidates")

                # Score bar chart
                fig_bo = go.Figure()
                fig_bo.add_trace(go.Bar(
                    x=[r["sym"] for r in bo_results],
                    y=[r["score"] for r in bo_results],
                    marker_color=["#00c853" if r["score"] >= 70 else "#ff9100" if r["score"] >= 50 else "#2979ff"
                                  for r in bo_results],
                    text=[f'{r["score"]}' for r in bo_results],
                    textposition="outside",
                ))
                fig_bo.update_layout(
                    title="Breakout Score (100 = perfect setup)",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#ccc", height=300,
                    yaxis=dict(range=[0, 110]),
                    showlegend=False,
                )
                st.plotly_chart(fig_bo, use_container_width=True)

                # Detail table
                tbl_bo = []
                for r in bo_results:
                    signals = []
                    if r["ath_pct"] > -3:    signals.append("📍Near 52W High")
                    if r["vol_ratio"] > 2:   signals.append("🔊 Vol Surge")
                    if r["bb_breakout"]:     signals.append("📈 BB Break")
                    if r["above_ma20"]:      signals.append("✅ >MA20")
                    if r["mom_5d"] > 5:      signals.append("⚡ 5D+5%")
                    tbl_bo.append({
                        "Symbol":     r["sym"],
                        "Price":      f"${r['curr']:.2f}",
                        "52W High":   f"${r['high_52w']:.2f}",
                        "vs 52W":     f"{r['ath_pct']:+.1f}%",
                        "RSI":        f"{r['rsi']:.0f}",
                        "Vol Ratio":  f"{r['vol_ratio']:.1f}x",
                        "5D Mom":     f"{r['mom_5d']:+.1f}%",
                        "Score":      r["score"],
                        "Signals":    "  ".join(signals),
                    })
                df_bo = pd.DataFrame(tbl_bo)

                def _bo_color(val):
                    try:
                        v = int(val)
                        if v >= 70: return "color:#00c853;font-weight:700"
                        if v >= 50: return "color:#ff9100;font-weight:700"
                        return "color:#888"
                    except Exception:
                        return ""

                st.dataframe(
                    df_bo.style.map(_bo_color, subset=["Score"]),
                    use_container_width=True, hide_index=True,
                )

                st.markdown("---")
                st.markdown("""
    **How to trade a breakout:**
    1. Enter when price **closes above** the 52-week high with 2× volume (not intraday — wait for close)
    2. Set stop at the breakout level (usually the old 52W high) — exit if it falls back below
    3. Target: 20-30 % above entry, or trail the stop with the 20-day MA
    4. **Don't chase** — if you missed the first day of a breakout, wait for a pullback to the breakout level
    """)

            elif "bo_results" in st.session_state and not st.session_state["bo_results"]:
                st.info("No breakout setups found. Try lowering the minimum score or adding more tickers.")

        # ─────────────────────────────────────────────────────────────────────────
        # AH TAB 3 — Short Squeeze Radar
        # ─────────────────────────────────────────────────────────────────────────
        with _fs_t6:
            st.markdown("### 🌪️ Short Squeeze Radar")
            st.markdown(
                "Stocks with **high short interest** that are starting to move up. "
                "When short sellers are forced to buy back, prices can spike 50-300 % rapidly. "
                "Short squeeze plays are high-risk, high-reward and require quick decision-making."
            )

            col_sa, col_sb = st.columns([2, 1])
            with col_sa:
                sq_input = st.text_area(
                    "Stocks to scan (comma-separated)",
                    value="IONQ, PLTR, RIVN, LCID, SOFI, HOOD, UPST, AFRM, OPEN, "
                          "GME, AMC, BBBY, BYND, MSTR, COIN, DKNG, RBLX, PATH, "
                          "SMCI, WOLF, PLUG, FCEL, BLNK, CHPT, NKLA, RIDE, SPCE",
                    height=100,
                    key="sq_input",
                )
            with col_sb:
                min_sq_score  = st.slider("Min squeeze score", 20, 80, 40, key="sq_min_score")
                min_short_pct = st.slider("Min short float %", 5, 30, 10, key="sq_min_short")

            if st.button("🔍 Scan Short Squeeze", key="sq_scan_btn", type="primary"):
                syms = [s.strip().upper() for s in sq_input.split(",") if s.strip()]
                prog = st.progress(0, text="Scanning short interest data…")
                sq_results = []
                for i, sym in enumerate(syms):
                    prog.progress((i + 1) / len(syms), text=f"Checking {sym}…")
                    r = _get_squeeze_data(sym)
                    if r and r["score"] >= min_sq_score and r["short_float"] * 100 >= min_short_pct:
                        sq_results.append(r)
                prog.empty()
                sq_results.sort(key=lambda x: x["score"], reverse=True)
                st.session_state["sq_results"] = sq_results

            if "sq_results" in st.session_state and st.session_state["sq_results"]:
                sq_results = st.session_state["sq_results"]
                st.markdown(f"#### {len(sq_results)} Short Squeeze Candidates")

                for r in sq_results:
                    sf_pct  = r["short_float"] * 100
                    dtc     = r["short_ratio"]
                    sq_tier = (
                        ("🔥 EXTREME", "#ff0000") if r["score"] >= 70 else
                        ("⚡ HIGH",    "#ff9100") if r["score"] >= 50 else
                        ("📊 MODERATE","#2979ff")
                    )

                    st.markdown(
                        f"""<div style="background:#111;border:1px solid #222;border-radius:8px;
                        padding:12px 16px;margin-bottom:10px">
                          <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
                            <div>
                              <span style="color:#fff;font-weight:700;font-size:1.1rem">{r['sym']}</span>
                              &nbsp;<span style="color:{sq_tier[1]};font-size:0.8rem;font-weight:700">{sq_tier[0]}</span>
                            </div>
                            <div>
                              <div style="color:#888;font-size:0.73rem">Short Float</div>
                              <div style="color:#ff5252;font-weight:700">{sf_pct:.1f}%</div>
                            </div>
                            <div>
                              <div style="color:#888;font-size:0.73rem">Days to Cover</div>
                              <div style="color:#ff9100;font-weight:700">{dtc:.1f}d</div>
                            </div>
                            <div>
                              <div style="color:#888;font-size:0.73rem">1M Price Change</div>
                              <div style="color:{'#00c853' if r['mom_1m']>0 else '#ff5252'};font-weight:700">
                                {r['mom_1m']:+.1f}%</div>
                            </div>
                            <div>
                              <div style="color:#888;font-size:0.73rem">Vol Trend</div>
                              <div style="color:#aaa">{r['vol_ratio']:.1f}× avg</div>
                            </div>
                            <div>
                              <div style="color:#888;font-size:0.73rem">vs MA20</div>
                              <div style="color:{'#00c853' if r['above_ma20'] else '#888'}">
                                {'✅ Above' if r['above_ma20'] else '❌ Below'}</div>
                            </div>
                            <div style="margin-left:auto">
                              <div style="color:#888;font-size:0.73rem">Score</div>
                              <div style="color:{sq_tier[1]};font-weight:700;font-size:1.1rem">{r['score']}/100</div>
                            </div>
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("""
    **How short squeezes work:**
    - Short sellers borrow shares and sell them, betting the price falls
    - When the price rises instead, they're forced to buy back at a loss (called "covering")
    - Mass covering = sudden price spike — the squeeze

    **Warning signs that a squeeze is starting:**
    - Price breaks above a key resistance level
    - Volume spikes 3-5× the average
    - Price starts moving fast intraday with no obvious news

    **Risk management is critical here:**
    - Short squeezes can reverse just as violently — be prepared to exit in minutes, not hours
    - Never hold a squeeze play overnight expecting it to continue
    - These are trades, not investments — max position 2-3 %
    """)

            elif "sq_results" in st.session_state and not st.session_state["sq_results"]:
                st.info("No squeeze candidates found. Try lowering the minimum score or short % requirements.")

        # ─────────────────────────────────────────────────────────────────────────
        # AH TAB 4 — Paper Trading
        # ─────────────────────────────────────────────────────────────────────────
        with _fs_t7:
            st.markdown("### 📝 Paper Trading Tracker")
            st.markdown(
                "**Paper trade for at least 4 weeks before risking real money.** "
                "This tracks your virtual trades and shows your real win rate. "
                "If you can't be consistently profitable on paper, you won't be on real money either."
            )

            trades = _load_paper_trades()

            # ── Portfolio Sync ────────────────────────────────────────────────────
            pf_raw = []
            if PORTFOLIO_FILE.exists():
                try:
                    pf_raw = json.loads(PORTFOLIO_FILE.read_text())
                except Exception:
                    pf_raw = []

            if pf_raw:
                with st.expander("🔗 Sync My Portfolio to Paper Trading", expanded=False):
                    st.markdown(
                        "Import your real holdings as paper trades so you can **track P&L, "
                        "set targets, and apply stop-loss rules** — all in one place."
                    )

                    # Defaults for target and stop
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        sync_target_pct = st.number_input(
                            "Default Target (%)", min_value=1.0, max_value=500.0,
                            value=30.0, step=1.0, key="sync_target_pct",
                            help="Applied to every holding that doesn't already have a paper trade open.",
                        )
                    with col_d2:
                        sync_stop_pct = st.number_input(
                            "Default Stop Loss (%)", min_value=1.0, max_value=50.0,
                            value=15.0, step=0.5, key="sync_stop_pct",
                            help="Stop loss % below your average buy price.",
                        )
                    with col_d3:
                        sync_setup = st.selectbox(
                            "Setup label for synced trades",
                            ["Portfolio Holding", "Earnings Catalyst", "Breakout Momentum", "Other"],
                            key="sync_setup_label",
                        )

                    # Already-open symbols so we don't duplicate
                    open_syms = {t["symbol"] for t in trades if t.get("status") == "open"}

                    # Fetch live prices + build preview
                    preview_rows  = []
                    to_sync       = []   # holdings not yet in paper trades
                    already_synced = []

                    fetch_prog = st.progress(0, text="Fetching live prices…")
                    for idx_h, h in enumerate(pf_raw):
                        fetch_prog.progress((idx_h + 1) / len(pf_raw), text=f"Fetching {h['symbol']}…")
                        sym        = h["symbol"]
                        avg_px     = float(h.get("avg_price", 0) or 0)
                        qty        = float(h.get("qty", 0) or 0)
                        invested   = round(avg_px * qty, 2)
                        target_px  = round(avg_px * (1 + sync_target_pct / 100), 2)
                        stop_px    = round(avg_px * (1 - sync_stop_pct   / 100), 2)
                        target_pctr = sync_target_pct
                        stop_pctr   = -sync_stop_pct
                        rr          = target_pctr / sync_stop_pct if sync_stop_pct else 0

                        # Live price
                        curr_px = None
                        try:
                            fi = yf.Ticker(sym).fast_info
                            curr_px = getattr(fi, "last_price", None)
                            if curr_px:
                                curr_px = round(float(curr_px), 2)
                        except Exception:
                            pass

                        pnl_pct = round((curr_px / avg_px - 1) * 100, 2) if curr_px and avg_px else None

                        row = {
                            "Symbol":       sym,
                            "Name":         h.get("name", sym)[:22],
                            "Qty":          qty,
                            "Avg Buy":      f"${avg_px:.2f}",
                            "Current":      f"${curr_px:.2f}" if curr_px else "—",
                            "P&L so far":   f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
                            "Invested ($)": f"${invested:,.0f}",
                            "Target":       f"${target_px:.2f} (+{sync_target_pct:.0f}%)",
                            "Stop":         f"${stop_px:.2f} (-{sync_stop_pct:.0f}%)",
                            "R:R":          f"{rr:.1f}x",
                            "Status":       "✅ Already tracked" if sym in open_syms else "➕ Will be added",
                        }
                        preview_rows.append(row)

                        if sym not in open_syms:
                            to_sync.append({
                                "sym": sym, "avg_px": avg_px, "qty": qty,
                                "invested": invested, "target_px": target_px,
                                "stop_px": stop_px, "target_pctr": target_pctr,
                                "stop_pctr": stop_pctr, "rr": rr,
                                "name": h.get("name", sym),
                            })
                        else:
                            already_synced.append(sym)

                    fetch_prog.empty()

                    # Preview table
                    if preview_rows:
                        df_preview = pd.DataFrame(preview_rows)

                        def _sync_status_color(val):
                            if "Already" in str(val): return "color:#2979ff;font-weight:600"
                            if "Will be" in str(val): return "color:#00c853;font-weight:600"
                            return ""

                        def _pnl_color(val):
                            try:
                                v = float(str(val).replace("%", "").replace("+", ""))
                                if v > 0:  return "color:#00c853;font-weight:600"
                                if v < 0:  return "color:#ff5252;font-weight:600"
                            except Exception:
                                pass
                            return "color:#aaa"

                        st.dataframe(
                            df_preview.style
                                .map(_sync_status_color, subset=["Status"])
                                .map(_pnl_color,         subset=["P&L so far"]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Action
                    if already_synced:
                        st.info(
                            f"**{len(already_synced)} holding(s) already tracked:** "
                            + ", ".join(already_synced)
                            + " — skipping duplicates."
                        )

                    if to_sync:
                        if st.button(
                            f"🔗 Sync {len(to_sync)} holding(s) to Paper Trading",
                            key="do_portfolio_sync_btn",
                            type="primary",
                        ):
                            added = []
                            for h in to_sync:
                                new_t = {
                                    "id":         str(uuid.uuid4())[:8],
                                    "symbol":     h["sym"],
                                    "entry":      h["avg_px"],
                                    "target":     h["target_px"],
                                    "stop":       h["stop_px"],
                                    "size":       h["invested"],
                                    "setup":      sync_setup,
                                    "reason":     f"Synced from portfolio · {h['qty']} shares @ ${h['avg_px']:.2f} avg",
                                    "date":       datetime.date.today().isoformat(),
                                    "status":     "open",
                                    "exit_price": None,
                                    "exit_date":  None,
                                    "pnl_pct":    None,
                                    "target_pct": h["target_pctr"],
                                    "stop_pct":   h["stop_pctr"],
                                    "rr_ratio":   h["rr"],
                                    "qty":        h["qty"],
                                    "source":     "portfolio_sync",
                                }
                                trades.append(new_t)
                                added.append(h["sym"])
                            _save_paper_trades(trades)
                            st.success(
                                f"✅ Synced {len(added)} holding(s): {', '.join(added)}. "
                                "Scroll down to see them in Open Paper Trades."
                            )
                            st.rerun()
                    else:
                        st.success("✅ All portfolio holdings are already being tracked below.")

            # ── Add new trade ─────────────────────────────────────────────────────
            with st.expander("➕ Add a New Paper Trade", expanded=len(trades) == 0):
                col_pt1, col_pt2, col_pt3 = st.columns(3)
                with col_pt1:
                    pt_sym    = st.text_input("Symbol", placeholder="e.g. NVDA", key="pt_sym").upper().strip()
                    pt_entry  = st.number_input("Entry Price ($)", min_value=0.01, value=100.0, step=0.01, key="pt_entry")
                    pt_size   = st.number_input("Position ($)", min_value=1.0, value=500.0, step=10.0, key="pt_size")
                with col_pt2:
                    pt_target = st.number_input("Target Price ($)", min_value=0.01, value=130.0, step=0.01, key="pt_target")
                    pt_stop   = st.number_input("Stop Loss ($)", min_value=0.01, value=85.0, step=0.01, key="pt_stop")
                    pt_setup  = st.selectbox("Setup Type", ["Earnings Catalyst", "Breakout Momentum",
                                                            "Short Squeeze", "Other"], key="pt_setup")
                with col_pt3:
                    pt_reason = st.text_area("Why this trade?", height=105, key="pt_reason",
                                             placeholder="e.g. Earnings in 5 days, historically moves ±18%...")

                if st.button("📝 Add Paper Trade", key="pt_add_btn", type="primary"):
                    if pt_sym and pt_entry > 0 and pt_target > 0 and pt_stop > 0:
                        target_pct = (pt_target / pt_entry - 1) * 100
                        stop_pct   = (pt_stop / pt_entry - 1) * 100
                        rr_ratio   = abs(target_pct / stop_pct) if stop_pct != 0 else 0
                        new_trade  = {
                            "id":         str(uuid.uuid4())[:8],
                            "symbol":     pt_sym,
                            "entry":      pt_entry,
                            "target":     pt_target,
                            "stop":       pt_stop,
                            "size":       pt_size,
                            "setup":      pt_setup,
                            "reason":     pt_reason,
                            "date":       datetime.date.today().isoformat(),
                            "status":     "open",
                            "exit_price": None,
                            "exit_date":  None,
                            "pnl_pct":    None,
                            "target_pct": target_pct,
                            "stop_pct":   stop_pct,
                            "rr_ratio":   rr_ratio,
                        }
                        trades.append(new_trade)
                        _save_paper_trades(trades)
                        st.success(
                            f"✅ Paper trade added: {pt_sym} entry ${pt_entry:.2f} → "
                            f"target ${pt_target:.2f} (+{target_pct:.1f}%) | "
                            f"stop ${pt_stop:.2f} ({stop_pct:.1f}%) | "
                            f"R:R = {rr_ratio:.1f}x"
                        )
                        st.rerun()
                    else:
                        st.error("Fill in all fields to add a paper trade.")

            # ── Open trades ───────────────────────────────────────────────────────
            open_trades = [t for t in trades if t.get("status") == "open"]
            closed_trades = [t for t in trades if t.get("status") != "open"]

            if open_trades:
                st.markdown(f"#### 📂 Open Paper Trades ({len(open_trades)})")
                for t in open_trades:
                    # Fetch current price
                    curr_px = None
                    try:
                        fi = yf.Ticker(t["symbol"]).fast_info
                        curr_px = getattr(fi, "last_price", None)
                    except Exception:
                        pass

                    unreal_pct = ((curr_px / t["entry"]) - 1) * 100 if curr_px else None
                    target_pct = t.get("target_pct", (t["target"] / t["entry"] - 1) * 100)
                    stop_pct   = t.get("stop_pct", (t["stop"] / t["entry"] - 1) * 100)
                    rr_ratio   = t.get("rr_ratio", 0)

                    px_color = ("#00c853" if (unreal_pct or 0) > 0 else "#ff5252") if unreal_pct is not None else "#888"

                    with st.container():
                        col_oa, col_ob, col_oc = st.columns([3, 2, 1])
                        with col_oa:
                            st.markdown(
                                f"""<div style="background:#111;border:1px solid #222;border-radius:8px;padding:10px 14px">
                                  <span style="color:#fff;font-weight:700;font-size:1rem">{t['symbol']}</span>
                                  &nbsp;<span style="color:#888;font-size:0.8rem">{t['setup']}</span><br>
                                  <span style="color:#aaa;font-size:0.78rem">
                                    Entry ${t['entry']:.2f} · Target ${t['target']:.2f} (+{target_pct:.1f}%) ·
                                    Stop ${t['stop']:.2f} ({stop_pct:.1f}%) · R:R {rr_ratio:.1f}x
                                  </span><br>
                                  <span style="color:#666;font-size:0.72rem">Added {t['date']} · ${t['size']:.0f} position</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with col_ob:
                            if curr_px and unreal_pct is not None:
                                progress_pct = min(max((unreal_pct / target_pct) if target_pct else 0, 0), 1.0)
                                st.markdown(
                                    f"**Current:** ${curr_px:.2f} "
                                    f"<span style='color:{px_color};font-weight:700'>({unreal_pct:+.1f}%)</span>",
                                    unsafe_allow_html=True,
                                )
                                st.progress(progress_pct)
                                if unreal_pct <= stop_pct:
                                    st.error("🚨 AT STOP LOSS — Consider closing this trade")
                                elif unreal_pct >= target_pct * 0.9:
                                    st.success("🎯 Near target — consider taking profit")
                            else:
                                st.caption("Price unavailable")
                        with col_oc:
                            exit_price_key = f"exit_px_{t['id']}"
                            ep = st.number_input("Exit at", min_value=0.01,
                                                 value=float(curr_px or t["entry"]),
                                                 key=exit_price_key, step=0.01, label_visibility="collapsed")
                            if st.button("Close", key=f"close_{t['id']}"):
                                pnl_pct = (ep / t["entry"] - 1) * 100
                                for trade in trades:
                                    if trade["id"] == t["id"]:
                                        trade["status"]     = "closed"
                                        trade["exit_price"] = round(ep, 2)
                                        trade["exit_date"]  = datetime.date.today().isoformat()
                                        trade["pnl_pct"]    = round(pnl_pct, 2)
                                        break
                                _save_paper_trades(trades)
                                st.rerun()

            # ── Statistics & closed trades ────────────────────────────────────────
            if closed_trades:
                wins  = [t for t in closed_trades if (t.get("pnl_pct") or 0) > 0]
                total = len(closed_trades)
                win_rate = len(wins) / total * 100
                avg_win  = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0
                losses   = [t for t in closed_trades if (t.get("pnl_pct") or 0) <= 0]
                avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0
                exp_value = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

                st.markdown("---")
                st.markdown("#### 📊 Paper Trading Statistics")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Trades", total)
                c2.metric("Win Rate", f"{win_rate:.0f}%",
                          delta="Need >40% to be viable" if win_rate < 40 else "Good!")
                c3.metric("Avg Win", f"+{avg_win:.1f}%")
                c4.metric("Avg Loss", f"{avg_loss:.1f}%")
                c5.metric("Expected Value / Trade",
                          f"{exp_value:+.1f}%",
                          delta="Profitable edge!" if exp_value > 0 else "No edge yet — keep paper trading")

                if exp_value > 2:
                    st.success(
                        "✅ **You have a positive expected value edge.** "
                        f"Each trade returns +{exp_value:.1f}% on average. "
                        "Consider starting with very small real positions (2-3% of portfolio max)."
                    )
                elif exp_value > 0:
                    st.warning(
                        f"⚠️ Edge exists (+{exp_value:.1f}% EV) but is small. "
                        "Keep paper trading another 2-4 weeks to confirm it's real and not luck."
                    )
                else:
                    st.error(
                        "❌ **No profitable edge yet.** Expected value is negative — "
                        "do NOT go live yet. Review which setups are losing and why."
                    )

                # Closed trade table
                st.markdown("#### 📋 Closed Trades")
                ct_rows = []
                for t in sorted(closed_trades, key=lambda x: x.get("exit_date", ""), reverse=True):
                    pnl = t.get("pnl_pct", 0) or 0
                    ct_rows.append({
                        "Symbol": t["symbol"],
                        "Setup":  t.get("setup", "—"),
                        "Entry":  f"${t['entry']:.2f}",
                        "Exit":   f"${t['exit_price']:.2f}" if t.get("exit_price") else "—",
                        "P&L":    f"{pnl:+.1f}%",
                        "$ P&L":  f"${t['size'] * pnl / 100:+.0f}",
                        "Date In": t["date"],
                        "Date Out": t.get("exit_date", "—"),
                        "Result": "✅ WIN" if pnl > 0 else "❌ LOSS",
                    })

                if ct_rows:
                    df_ct = pd.DataFrame(ct_rows)

                    def _ct_color(val):
                        if str(val) == "✅ WIN":  return "color:#00c853;font-weight:700"
                        if str(val) == "❌ LOSS": return "color:#ff5252;font-weight:700"
                        return ""

                    st.dataframe(
                        df_ct.style.map(_ct_color, subset=["Result"]),
                        use_container_width=True, hide_index=True,
                    )

            if not trades:
                st.info(
                    "No paper trades yet. Add your first paper trade above. "
                    "Try it with the setups from the Catalyst Scanner or Breakout Scanner tabs."
                )

            st.markdown("---")
            st.caption(
                "⚡ Alpha Hunter is a research and simulation tool. "
                "The 30 %/month goal requires catching high-momentum, high-risk setups. "
                "Build a verified win rate through paper trading before using real capital. "
                "This is NOT financial advice."
            )

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 6 — LIVE TRADE SIGNALS
    # ══════════════════════════════════════════════════════════════════════════════

    def _parse_pdf_for_tickers(pdf_bytes: bytes) -> list[str]:
        """Extract stock tickers from a PDF portfolio statement."""
        try:
            import pdfplumber, io, re as _re
            text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"

            # Try Groq/Gemini to intelligently parse tickers
            _api_key = os.getenv("GROQ_API_KEY", "").strip()
            if _api_key:
                try:
                    from groq import Groq as _G
                    _c = _G(api_key=_api_key)
                    _r = _c.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content":
                            f"Extract all stock ticker symbols from this portfolio statement. "
                            f"Return ONLY a comma-separated list of tickers (e.g. NVDA,META,AAPL). "
                            f"For Indian stocks add .NS suffix (e.g. RELIANCE.NS). "
                            f"No explanation, just tickers.\n\nText:\n{text[:3000]}"}],
                        temperature=0, max_tokens=200,
                    )
                    raw = _r.choices[0].message.content.strip()
                    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
                    if tickers:
                        return tickers
                except Exception:
                    pass

            # Fallback: regex for common ticker patterns
            us_tickers = _re.findall(r'\b([A-Z]{1,5})\b', text)
            in_tickers = _re.findall(r'\b([A-Z][A-Z0-9]{2,10})\b', text)
            blocklist  = {"THE","FOR","AND","ARE","NOT","HAS","USD","INR","NSE","BSE",
                          "CMP","PNL","LTP","SL","TP","QTY","AVG","NET","YTD","MTD"}
            seen, out = set(), []
            for t in us_tickers + in_tickers:
                if t not in seen and t not in blocklist and 1 < len(t) <= 6:
                    seen.add(t); out.append(t)
            return out[:30]
        except Exception:
            return []


    _IMG_TICKER_PROMPT = (
        "You are a financial data extractor. Look at this image carefully — it may be a broker app, "
        "portfolio statement, watchlist screenshot, or trading platform. "
        "Extract every stock ticker symbol you can see. "
        "Return ONLY a comma-separated list of uppercase tickers (e.g. NVDA,META,AAPL). "
        "For Indian NSE stocks add .NS suffix (e.g. RELIANCE.NS,INFY.NS). "
        "If you see no tickers, return the single word NONE. "
        "No explanation, no other text."
    )

    def _parse_image_for_tickers(img_bytes: bytes, mime: str = "image/png") -> tuple[list[str], str]:
        """Extract stock tickers from a PNG/JPG screenshot using Gemini vision → Groq vision fallback.
        Returns (tickers, error_message). On success error_message is ''.
        """
        import base64

        def _clean(raw: str) -> list[str]:
            return [t.strip().upper() for t in raw.split(",")
                    if t.strip() and t.strip().upper() not in ("NONE", "")]

        # ── 1. Try Gemini vision ──────────────────────────────────────────────────
        _gem_key = os.getenv("GEMINI_API_KEY", "").strip()
        if _gem_key:
            try:
                from google import genai as _genai_sdk
                from google.genai import types as _gtypes
                _client = _genai_sdk.Client(api_key=_gem_key)
                _resp = _client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        _gtypes.Part.from_bytes(data=img_bytes, mime_type=mime),
                        _IMG_TICKER_PROMPT,
                    ],
                )
                raw = (_resp.text or "").strip()
                tickers = _clean(raw)
                if tickers:
                    return tickers, ""
                if raw.upper() == "NONE":
                    return [], "No stock tickers found in the image. Try a clearer screenshot."
            except Exception as _e:
                _gem_err = str(_e)
                if "quota" not in _gem_err.lower() and "429" not in _gem_err:
                    return [], f"Gemini image parsing failed: {_gem_err[:120]}"
                # quota hit → fall through to Groq

        # ── 2. Groq vision fallback (llama-3.2-11b-vision-preview) ───────────────
        _groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if _groq_key:
            try:
                from groq import Groq as _G
                b64 = base64.b64encode(img_bytes).decode()
                data_url = f"data:{mime};base64,{b64}"
                _gc = _G(api_key=_groq_key)
                _gr = _gc.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": _IMG_TICKER_PROMPT},
                        ],
                    }],
                    temperature=0, max_tokens=300,
                )
                raw = (_gr.choices[0].message.content or "").strip()
                tickers = _clean(raw)
                if tickers:
                    return tickers, ""
                return [], "Groq found no stock tickers in the image. Try a clearer screenshot or add tickers manually."
            except Exception as _e:
                return [], f"Both Gemini (quota) and Groq vision failed: {str(_e)[:120]}. Add tickers manually below."

        return [], "No AI vision key configured. Add tickers manually using the text box."


    @st.cache_data(ttl=15, show_spinner=False)
    def _fetch_live_candles(sym: str, interval: str, period: str, prepost: bool = True):
        """Cached yfinance fetch with 15s TTL. Includes pre/post market when prepost=True."""
        try:
            df = yf.download(sym, period=period, interval=interval,
                             progress=False, auto_adjust=True, prepost=prepost)
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            return None


    def _hedge_fund_rules(df, fh_quote: dict = None) -> dict:
        """
        Rule-based signal engine used by systematic quant funds.
        Returns list of rules fired + composite score + verdict.
        """
        if df is None or len(df) < 20:
            return {"verdict": "INSUFFICIENT DATA", "score": 0, "rules": []}

        close  = df["Close"].dropna()
        high   = df["High"].dropna()
        low    = df["Low"].dropna()
        volume = df["Volume"].dropna()
        cur    = float(fh_quote.get("c", close.iloc[-1])) if fh_quote and fh_quote.get("c") else float(close.iloc[-1])

        rules = []  # (name, signal, strength, reason)
        # signal: +1 bullish, 0 neutral, -1 bearish
        # strength: 1 weak, 2 moderate, 3 strong

        def _rule(name, sig, strength, reason, category=""):
            rules.append({"name": name, "signal": sig, "strength": strength,
                          "score": sig * strength, "reason": reason, "category": category})

        # ── TREND RULES ───────────────────────────────────────────────────────────
        ma20  = float(close.rolling(20).mean().iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
        ema9  = float(close.ewm(span=9).mean().iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])

        # Rule 1: Price vs MA20 (Trend)
        if cur > ma20 * 1.03:
            _rule("Price above MA20", 1, 2, f"Price {cur:.2f} > MA20 {ma20:.2f} — uptrend confirmed", "Trend")
        elif cur < ma20 * 0.97:
            _rule("Price below MA20", -1, 2, f"Price {cur:.2f} < MA20 {ma20:.2f} — downtrend", "Trend")
        else:
            _rule("Price near MA20", 0, 1, f"Price at MA20 — no clear trend", "Trend")

        # Rule 2: EMA9 vs EMA20 crossover (Short-term momentum)
        ema9_prev  = float(close.ewm(span=9).mean().iloc[-2])
        ema20_prev = float(close.ewm(span=20).mean().iloc[-2])
        if ema9 > ema20 and ema9_prev <= ema20_prev:
            _rule("EMA9 × EMA20 Bullish Cross", 1, 3, "EMA9 just crossed above EMA20 — strong momentum signal", "Trend")
        elif ema9 < ema20 and ema9_prev >= ema20_prev:
            _rule("EMA9 × EMA20 Bearish Cross", -1, 3, "EMA9 just crossed below EMA20 — sell signal", "Trend")
        elif ema9 > ema20:
            _rule("EMA9 above EMA20", 1, 1, "Short-term trend above medium-term — bullish bias", "Trend")
        else:
            _rule("EMA9 below EMA20", -1, 1, "Short-term trend below medium-term — bearish bias", "Trend")

        # ── MOMENTUM RULES ────────────────────────────────────────────────────────
        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
        rsi   = float((100 - 100/(1+gain/loss)).iloc[-1])
        rsi_prev = float((100 - 100/(1+gain/loss)).iloc[-2])

        if rsi < 30:
            _rule("RSI Oversold", 1, 3, f"RSI {rsi:.0f} — deeply oversold, reversal likely", "Momentum")
        elif rsi > 75:
            _rule("RSI Overbought", -1, 3, f"RSI {rsi:.0f} — overbought, pullback risk", "Momentum")
        elif rsi < 45 and rsi > rsi_prev:
            _rule("RSI Rising from Low", 1, 2, f"RSI {rsi:.0f} rising from depressed levels — momentum building", "Momentum")
        elif rsi > 55 and rsi < rsi_prev:
            _rule("RSI Fading from High", -1, 2, f"RSI {rsi:.0f} turning down from elevated levels — momentum fading", "Momentum")
        else:
            _rule("RSI Neutral", 0, 1, f"RSI {rsi:.0f} — no extreme reading", "Momentum")

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        hist  = macd - signal_line
        h_now  = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])

        if h_now > 0 and h_prev <= 0:
            _rule("MACD Bullish Crossover", 1, 3, "MACD histogram crossed above zero — institutional buy signal", "Momentum")
        elif h_now < 0 and h_prev >= 0:
            _rule("MACD Bearish Crossover", -1, 3, "MACD histogram crossed below zero — institutional sell signal", "Momentum")
        elif h_now > 0 and h_now > h_prev:
            _rule("MACD Histogram Growing", 1, 2, f"MACD hist {h_now:.3f} — bullish momentum accelerating", "Momentum")
        elif h_now < 0 and h_now < h_prev:
            _rule("MACD Histogram Declining", -1, 2, f"MACD hist {h_now:.3f} — bearish momentum accelerating", "Momentum")
        else:
            _rule("MACD Neutral", 0, 1, "MACD showing no strong directional momentum", "Momentum")

        # ── VOLUME RULES ──────────────────────────────────────────────────────────
        vol_now = float(volume.iloc[-1])
        vol_avg = float(volume.rolling(20).mean().iloc[-1]) or 1
        vol_ratio = vol_now / vol_avg
        price_up = float(close.iloc[-1]) >= float(close.iloc[-2])

        if vol_ratio > 2.5 and price_up:
            _rule("Volume Surge on Up Move", 1, 3, f"Volume {vol_ratio:.1f}× avg — institutional buying confirmed", "Volume")
        elif vol_ratio > 2.5 and not price_up:
            _rule("Volume Surge on Down Move", -1, 3, f"Volume {vol_ratio:.1f}× avg — institutional selling/distribution", "Volume")
        elif vol_ratio > 1.5 and price_up:
            _rule("Above-Average Volume Bullish", 1, 2, f"Volume {vol_ratio:.1f}× avg — conviction behind the move up", "Volume")
        elif vol_ratio < 0.5:
            _rule("Low Volume", 0, 1, f"Volume {vol_ratio:.1f}× avg — low conviction, wait for confirmation", "Volume")
        else:
            _rule("Normal Volume", 0, 1, f"Volume {vol_ratio:.1f}× avg — average activity", "Volume")

        # ── BOLLINGER BAND RULES ──────────────────────────────────────────────────
        std20 = close.rolling(20).std()
        bb_up = float((close.rolling(20).mean() + 2*std20).iloc[-1])
        bb_lo = float((close.rolling(20).mean() - 2*std20).iloc[-1])
        bb_pct = (cur - bb_lo) / (bb_up - bb_lo) * 100 if bb_up > bb_lo else 50

        if bb_pct > 95:
            _rule("Price at BB Upper Band", -1, 3, f"Price at {bb_pct:.0f}% of BB range — statistically overbought, mean reversion likely", "Structure")
        elif bb_pct < 5:
            _rule("Price at BB Lower Band", 1, 3, f"Price at {bb_pct:.0f}% of BB range — statistically oversold, bounce likely", "Structure")
        elif bb_pct > 75:
            _rule("Price in Upper BB Zone", -1, 1, f"Price at {bb_pct:.0f}% of BB — elevated, watch for reversal", "Structure")
        elif bb_pct < 25:
            _rule("Price in Lower BB Zone", 1, 1, f"Price at {bb_pct:.0f}% of BB — suppressed, potential recovery", "Structure")
        else:
            _rule("Price in BB Midzone", 0, 1, f"Price at {bb_pct:.0f}% of BB — no extreme", "Structure")

        # ── PRICE ACTION RULES ────────────────────────────────────────────────────
        # Higher highs / lower lows over last 10 bars
        if len(high) >= 10:
            _hh = float(high.iloc[-1]) > float(high.iloc[-5]) > float(high.iloc[-10])
            _ll = float(low.iloc[-1])  < float(low.iloc[-5])  < float(low.iloc[-10])
            _hl = float(low.iloc[-1])  > float(low.iloc[-5])  > float(low.iloc[-10])
            if _hh and _hl:
                _rule("Higher Highs + Higher Lows", 1, 3, "Classic uptrend structure — price making higher highs and higher lows", "Structure")
            elif _ll:
                _rule("Lower Lows", -1, 2, "Downtrend structure — price making lower lows", "Structure")

        # ── COMPOSITE SCORE ───────────────────────────────────────────────────────
        total_score = sum(r["score"] for r in rules)
        max_score   = sum(r["strength"] for r in rules)

        if total_score >= 8:      verdict = "STRONG BUY"
        elif total_score >= 4:    verdict = "BUY"
        elif total_score <= -8:   verdict = "STRONG SELL — EXIT"
        elif total_score <= -4:   verdict = "SELL — REDUCE"
        else:                     verdict = "HOLD / WATCH"

        return {
            "verdict": verdict, "score": total_score, "max_score": max_score,
            "rsi": rsi, "bb_pct": bb_pct, "vol_ratio": vol_ratio,
            "macd_hist": h_now, "ema9": ema9, "ema20": ema20,
            "rules": rules, "cur": cur,
        }


    def _compute_live_signals(df) -> dict:
        """Compute RSI, MACD, BB from a price DataFrame and return exit signal."""
        if df is None or len(df) < 20:
            return {"signal": "INSUFFICIENT_DATA", "rsi": None, "macd": None}
        try:
            close = df["Close"].dropna()
            # RSI
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, 1e-9)
            rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line  = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_val   = float(macd_line.iloc[-1])
            sig_val    = float(signal_line.iloc[-1])
            macd_hist  = macd_val - sig_val
            # Bollinger
            ma20   = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            bb_hi  = float((ma20 + 2 * std20).iloc[-1])
            bb_lo  = float((ma20 - 2 * std20).iloc[-1])
            price  = float(close.iloc[-1])
            bb_pct = (price - bb_lo) / (bb_hi - bb_lo) * 100 if bb_hi != bb_lo else 50
            # Volume spike
            vol    = df["Volume"].dropna()
            vol_z  = (float(vol.iloc[-1]) - float(vol.mean())) / (float(vol.std()) + 1e-9) if len(vol) > 5 else 0
            # Determine signal
            exit_reasons = []
            if rsi > 75:   exit_reasons.append(f"RSI overbought ({rsi:.0f})")
            if rsi < 30:   exit_reasons.append(f"RSI oversold ({rsi:.0f}) — possible reversal")
            if macd_hist < 0 and macd_val < sig_val: exit_reasons.append("MACD bearish crossover")
            if bb_pct > 90: exit_reasons.append(f"Price at upper Bollinger Band ({bb_pct:.0f}%)")
            if vol_z > 2.5: exit_reasons.append(f"Volume spike {vol_z:.1f}σ — watch for reversal")

            if rsi > 78 or (rsi > 70 and macd_hist < 0):
                signal = "EXIT_NOW"
            elif len(exit_reasons) >= 2:
                signal = "WATCH"
            elif rsi < 35 or (macd_hist > 0 and rsi < 55):
                signal = "HOLD_BUY_DIP"
            else:
                signal = "HOLD"

            return {
                "signal": signal, "rsi": rsi, "macd": macd_val,
                "macd_hist": macd_hist, "bb_pct": bb_pct, "price": price,
                "exit_reasons": exit_reasons, "vol_z": vol_z,
            }
        except Exception:
            return {"signal": "ERROR", "rsi": None}


    def _fetch_finnhub_candles(symbol: str, interval: str, period: str):
        """Fetch OHLCV candles from Finnhub. Returns DataFrame or None.
        Falls back to None for non-US tickers (e.g. .NS) so yfinance handles them.
        """
        if "." in symbol:   # Indian / non-US — yfinance handles these
            return None
        api_key = os.getenv("FINNHUB_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            import finnhub as _fh, time as _t, pandas as _pd
            _res_map = {"1m":"1","2m":"1","5m":"5","15m":"15","30m":"30","1h":"60"}
            _res = _res_map.get(interval, "5")
            _period_secs = {"1d":86400,"2d":2*86400,"5d":5*86400,
                            "1mo":30*86400,"3mo":90*86400}.get(period, 5*86400)
            _now   = int(_t.time())
            _from  = _now - _period_secs
            _fc    = _fh.Client(api_key=api_key)
            _r     = _fc.stock_candles(symbol, _res, _from, _now)
            if _r.get("s") != "ok" or not _r.get("t"):
                return None
            _df = _pd.DataFrame({
                "Open":   _r["o"], "High":  _r["h"],
                "Low":    _r["l"], "Close": _r["c"],
                "Volume": _r["v"],
            }, index=_pd.to_datetime(_r["t"], unit="s", utc=True).tz_convert("America/New_York"))
            _df.index.name = "Datetime"
            return _df
        except Exception:
            return None


    def _fetch_finnhub_quote(symbol: str) -> dict:
        """Get real-time quote from Finnhub. Returns dict with 'c' (current price) etc."""
        api_key = os.getenv("FINNHUB_API_KEY", "").strip()
        if not api_key or "." in symbol:
            return {}
        try:
            import finnhub as _fh
            _fc = _fh.Client(api_key=api_key)
            return _fc.quote(symbol) or {}
        except Exception:
            return {}


    def _compute_sr_levels(df, n=3):
        """Pivot-based support/resistance. Returns (resistances, supports) near current price."""
        if df is None or len(df) < 25:
            return [], []
        high, low, close = df["High"], df["Low"], df["Close"]
        w = 5
        res_raw, sup_raw = [], []
        for i in range(w, len(df) - w):
            if float(high.iloc[i]) == float(high.iloc[i-w:i+w+1].max()):
                res_raw.append(float(high.iloc[i]))
            if float(low.iloc[i]) == float(low.iloc[i-w:i+w+1].min()):
                sup_raw.append(float(low.iloc[i]))

        def _cluster(levels, pct=0.006):
            if not levels: return []
            lvs = sorted(levels)
            groups, grp = [], [lvs[0]]
            for lv in lvs[1:]:
                if (lv - grp[-1]) / (grp[-1] + 1e-9) < pct:
                    grp.append(lv)
                else:
                    groups.append(sum(grp) / len(grp)); grp = [lv]
            groups.append(sum(grp) / len(grp))
            return groups

        cur = float(close.iloc[-1])
        res_cl = sorted([r for r in _cluster(res_raw) if r > cur * 1.001])
        sup_cl = sorted([s for s in _cluster(sup_raw) if s < cur * 0.999], reverse=True)
        return res_cl[:n], sup_cl[:n]


    def _detect_candle_patterns(df):
        """Detect the last 1-3 candle pattern. Returns list of (name, emoji, description)."""
        if df is None or len(df) < 3:
            return []
        def _row(i):
            o = float(df["Open"].iloc[i]); h = float(df["High"].iloc[i])
            l = float(df["Low"].iloc[i]);  c = float(df["Close"].iloc[i])
            body = abs(c - o); rng = max(h - l, 1e-9)
            uw = h - max(c, o); lw = min(c, o) - l
            return o, h, l, c, body, rng, uw, lw, c >= o

        o1,h1,l1,c1,b1,r1,uw1,lw1,bull1 = _row(-1)
        o2,h2,l2,c2,b2,r2,uw2,lw2,bull2 = _row(-2)

        patterns = []
        if b1 < 0.08 * r1:
            patterns.append(("Doji", "⚪", "Indecision — possible reversal ahead"))
        elif bull1 and lw1 > 2*b1 and uw1 < 0.5*b1 and not bull2:
            patterns.append(("Hammer", "🔨", "Bullish reversal — buyers rejected lower prices"))
        elif not bull1 and uw1 > 2*b1 and lw1 < 0.5*b1:
            patterns.append(("Shooting Star", "⭐", "Bearish reversal — sellers rejected higher prices"))
        elif bull1 and not bull2 and c1 > o2 and o1 < c2 and b1 > b2:
            patterns.append(("Bullish Engulfing", "🟢", "Strong buy signal — bulls in control"))
        elif not bull1 and bull2 and c1 < o2 and o1 > c2 and b1 > b2:
            patterns.append(("Bearish Engulfing", "🔴", "Strong sell signal — bears in control"))
        elif bull1 and b1 > 0.85 * r1:
            patterns.append(("Bullish Marubozu", "💚", "Strong momentum — trend continuation likely"))
        elif not bull1 and b1 > 0.85 * r1:
            patterns.append(("Bearish Marubozu", "🩸", "Heavy selling — watch for continuation"))
        elif bull1 and uw1 > 1.5*b1 and lw1 < 0.3*b1:
            patterns.append(("Spinning Top (bullish)", "🔵", "Mild bullish momentum, confirmation needed"))
        elif not bull1 and lw1 > 1.5*b1:
            patterns.append(("Inverted Hammer", "🟡", "Potential reversal — needs next-candle confirmation"))

        if not patterns:
            patterns.append(("No clear pattern", "⚫", "No single-candle signal detected"))
        return patterns


    def _compute_targets(df, resistances, supports):
        """ATR stop, session trailing stop, Fibonacci targets."""
        if df is None or len(df) < 15:
            return {}
        close = df["Close"]
        atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])
        cur = float(close.iloc[-1])
        recent_hi = float(df["High"].rolling(20).max().iloc[-1])
        recent_lo = float(df["Low"].rolling(20).min().iloc[-1])
        swing = recent_hi - recent_lo

        next_res = resistances[0] if resistances else round(cur + 2*atr, 2)
        next_sup = supports[0]    if supports    else round(cur - 2*atr, 2)
        session_stop   = round(cur - 1.5 * atr, 2)   # tightest intraday stop
        swing_stop     = round(cur - 2.5 * atr, 2)   # for swing traders
        fib_618_up     = round(recent_lo + swing * 1.618, 2)
        fib_100_up     = round(recent_lo + swing, 2)
        measured_move  = round(cur + (next_res - cur) * 1.0, 2) if resistances else round(cur * 1.04, 2)

        return {
            "cur": cur, "atr": atr,
            "next_resistance": next_res, "next_support": next_sup,
            "session_stop": session_stop, "swing_stop": swing_stop,
            "fib_618": fib_618_up, "fib_100": fib_100_up,
            "measured_move": measured_move,
            "recent_hi": recent_hi, "recent_lo": recent_lo,
        }


with main_tab3:
    st.markdown("## 💼 My Portfolio — INDmoney US Stocks")
    st.caption(
        "Live sync from INDmoney via Chrome Extension, or import CSV / add manually. "
        "Full analysis: MC targets, scores, recommendations, correlation matrix."
    )


    pt_tab1, pt_tab2, pt_tab3 = st.tabs([
        "💼 Holdings",
        "📊 Predictions & Tracking",
        "✅ Validate & Learn",
    ])

    with pt_tab1:
        # ── Chrome Extension Sync Panel ───────────────────────────────────────────
        def _check_sync_server():
            try:
                r = requests.get("http://localhost:8765/health", timeout=1.5)
                return r.ok
            except Exception:
                return False
    
        def _pull_from_server():
            try:
                r = requests.get("http://localhost:8765/portfolio", timeout=3)
                return r.json() if r.ok else None
            except Exception:
                return None
    
        server_running = _check_sync_server()
    
        sync_color = "#00c853" if server_running else "#ff5252"
        sync_label = "Sync server running" if server_running else "Sync server not running"
        st.markdown(
            f"<div style='background:#111;border:1px solid {sync_color}33;"
            f"border-radius:8px;padding:10px 16px;margin-bottom:14px;"
            f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<span style='width:9px;height:9px;border-radius:50%;background:{sync_color};"
            f"display:inline-block'></span>"
            f"<span style='color:{sync_color};font-weight:600;font-size:0.88rem'>{sync_label}</span>"
            f"</div>"
            f"<span style='color:#555;font-size:0.78rem'>"
            f"{'Chrome Extension ready — click Sync in the extension popup to push data' if server_running else 'Run: python sync_server.py'}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
    
        sync_col1, sync_col2 = st.columns([1, 3])
        with sync_col1:
            if st.button("🔄 Pull from Sync Server", disabled=not server_running, use_container_width=True):
                pulled = _pull_from_server()
                if pulled and len(pulled) > 0:
                    st.session_state["portfolio_holdings"] = pulled
                    _save_portfolio(pulled)
                    st.success(f"✅ Pulled {len(pulled)} holdings from INDmoney sync.")
                    st.rerun()
                else:
                    st.warning("No holdings on server yet. Click Sync in the Chrome Extension first.")
        with sync_col2:
            st.caption(
                "**Setup:** ① Run `pip install flask flask-cors` then `python sync_server.py` in a terminal.  "
                "② Install the Chrome Extension (load unpacked from `chrome_extension/` folder).  "
                "③ Navigate to indmoney.com → US Stocks → Portfolio.  "
                "④ Click the extension icon → Sync → Send to Dashboard."
            )
    
        # ── Setup Guide ───────────────────────────────────────────────────────────
        with st.expander("🔧 Setup Guide — Chrome Extension + Sync Server", expanded=not server_running):
            st.markdown("""
    **Step 1 — Start the local sync server** (one-time, keep it running):
    ```bash
    cd /Users/rohitrwt70/Desktop/stock-dashboard
    pip install flask flask-cors   # first time only
    python sync_server.py
    ```
    
    **Step 2 — Install the Chrome Extension:**
    1. Open Chrome → `chrome://extensions/`
    2. Enable **Developer mode** (toggle top-right)
    3. Click **Load unpacked**
    4. Select the folder: `stock-dashboard/chrome_extension/`
    5. The 📊 icon appears in your toolbar
    
    **Step 3 — Sync your portfolio:**
    1. Open [indmoney.com](https://indmoney.com) → log in → go to **US Stocks → Portfolio**
    2. Let the page load fully (this triggers the API calls the extension captures)
    3. Click the 📊 extension icon in Chrome toolbar
    4. Click **🔄 Sync from INDmoney** — it will show your holdings
    5. Click **📤 Send to Dashboard**
    6. Come back here and click **🔄 Pull from Sync Server**
    
    **How it works:**
    The extension runs silently on indmoney.com and intercepts the portfolio API call the page makes.
    No passwords are stored — it uses your existing browser session.
            """)
    
        # ── Load persisted portfolio ──────────────────────────────────────────────
        if "portfolio_holdings" not in st.session_state:
            st.session_state["portfolio_holdings"] = _load_portfolio()
    
        holdings = st.session_state["portfolio_holdings"]
    
        # ── Import section ────────────────────────────────────────────────────────
        imp_col1, imp_col2 = st.columns([2, 1])
    
        with imp_col1:
            st.markdown("#### 📂 Import from INDmoney CSV")
            uploaded = st.file_uploader(
                "Upload INDmoney export (CSV or Excel)",
                type=["csv", "xlsx", "xls"],
                key="portfolio_upload",
            )
            if uploaded:
                try:
                    if uploaded.name.endswith((".xlsx", ".xls")):
                        raw_df = pd.read_excel(uploaded)
                    else:
                        raw_df = pd.read_csv(uploaded)
                    parsed, err = _parse_indmoney_csv(raw_df)
                    if err:
                        st.error(f"Could not parse file: {err}\n\nPlease check the column names match expected format, or use Manual Entry.")
                    elif not parsed:
                        st.warning("No valid holdings found in the file. Check that Quantity > 0.")
                    else:
                        st.success(f"✅ Parsed {len(parsed)} holdings from file.")
                        st.session_state["portfolio_holdings"] = parsed
                        _save_portfolio(parsed)
                        holdings = parsed
                        st.rerun()
                except Exception as e:
                    st.error(f"File read error: {e}")
    
        with imp_col2:
            st.markdown("#### ✏️ Manual Entry")
            with st.form("manual_holding_form"):
                m_sym   = st.text_input("Ticker (e.g. AAPL)", key="m_sym").upper().strip()
                m_qty   = st.number_input("Quantity (shares)", min_value=0.01, value=1.0, step=0.01)
                m_price = st.number_input("Avg Buy Price ($)", min_value=0.01, value=100.0, step=0.01)
                m_name  = st.text_input("Company Name (optional)", key="m_name")
                add_btn = st.form_submit_button("➕ Add Holding")
                if add_btn and m_sym:
                    new_h = {
                        "symbol": m_sym, "name": m_name or m_sym,
                        "qty": m_qty, "avg_price": m_price, "currency": "$"
                    }
                    existing = [h for h in holdings if h["symbol"] != m_sym]
                    updated  = existing + [new_h]
                    st.session_state["portfolio_holdings"] = updated
                    _save_portfolio(updated)
                    st.success(f"Added {m_sym}")
                    st.rerun()
    
        # ── Portfolio content ─────────────────────────────────────────────────────
        if not holdings:
            st.info(
                "No holdings yet. Import your INDmoney CSV or add stocks manually above.",
                icon="💼",
            )
        else:
            # Remove button
            rm_sym = st.selectbox("Remove a holding", ["—"] + [h["symbol"] for h in holdings], key="rm_port")
            if rm_sym != "—" and st.button(f"🗑️ Remove {rm_sym}"):
                updated = [h for h in holdings if h["symbol"] != rm_sym]
                st.session_state["portfolio_holdings"] = updated
                _save_portfolio(updated)
                st.rerun()
    
            st.markdown("---")
            st.markdown("### 📊 Portfolio Analysis — Hedge Fund Approach")
            st.caption("Analysis framework used by institutional managers: risk attribution, stress testing, DCF-based entry/exit, factor exposure.")
    
            # ── Run full analysis on all holdings ─────────────────────────────────
            analysis_results = {}
            with st.spinner(f"Running institutional-grade analysis on {len(holdings)} holdings…"):
                for h in holdings:
                    analysis_results[h["symbol"]] = _analyse_holding(h["symbol"])
    
            # ── Compute all HF metrics ────────────────────────────────────────────
            total_invested = sum(float(h["qty"]) * float(h["avg_price"]) for h in holdings)
            total_current  = sum(
                float(h["qty"]) * float(analysis_results[h["symbol"]]["f"]["price"])
                if analysis_results.get(h["symbol"]) else float(h["qty"]) * float(h["avg_price"])
                for h in holdings
            )
            total_pnl     = total_current - total_invested
            total_pnl_pct = total_pnl / max(total_invested, 0.01) * 100
            port_beta     = sum(
                float(analysis_results[h["symbol"]]["f"].get("beta") or 1) *
                (float(h["qty"]) * float(analysis_results[h["symbol"]]["f"]["price"])) / max(total_current, 1)
                for h in holdings if analysis_results.get(h["symbol"])
            )
            risk_rows   = _hf_risk_attribution(holdings, analysis_results)
            factor_exp  = _hf_factor_exposure(holdings, analysis_results)
            health      = _hf_health_score(holdings, analysis_results, risk_rows, factor_exp)
            stress_data = _hf_stress_test(holdings, analysis_results)
            decisions   = {h["symbol"]: _hf_position_decision(h, analysis_results.get(h["symbol"])) for h in holdings}
    
            # ══ SECTION 1: Portfolio Health Score ═════════════════════════════════
            st.markdown("### 🏆 Portfolio Health Score")
            hs = health["total"]
            hs_clr = "#00c853" if hs >= 70 else ("#ffcc00" if hs >= 45 else "#ff5252")
            hcols = st.columns([1, 1, 1, 1, 1])
            hcols[0].metric("Overall Health",     f"{hs}/100",
                             help="Composite across diversification, risk mgmt, conviction, valuation")
            hcols[1].metric("Diversification",    f"{health['diversification']}/25")
            hcols[2].metric("Risk Management",    f"{health['risk_mgmt']}/25")
            hcols[3].metric("Conviction Quality", f"{health['conviction']}/25")
            hcols[4].metric("Valuation",          f"{health['valuation']}/25")
            st.progress(hs / 100)
    
            # ══ INDMONEY SCREENSHOT IMPORT ═══════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📱 Import from INDmoney Screenshot")
            with st.expander("📸 Upload INDmoney portfolio screenshot to sync holdings", expanded=False):
                _img_file = st.file_uploader(
                    "Take a screenshot of your INDmoney US Stocks portfolio page and upload here",
                    type=["png","jpg","jpeg"], key="indmoney_screenshot"
                )
                if _img_file:
                    st.image(_img_file, caption="Uploaded portfolio screenshot", use_container_width=True)
                    if st.button("🔍 Parse Holdings from Screenshot", key="parse_indmoney_img", type="primary"):
                        with st.spinner("Gemini reading your portfolio..."):
                            _img_bytes = _img_file.read()
                            _gem_key = os.getenv("GEMINI_API_KEY","").strip()
                            if _gem_key:
                                try:
                                    from google import genai as _gs
                                    from google.genai import types as _gt
                                    _gc = _gs.Client(api_key=_gem_key)
                                    _gr = _gc.models.generate_content(
                                        model="gemini-2.0-flash",
                                        contents=[
                                            _gt.Part.from_bytes(data=_img_bytes, mime_type="image/png"),
                                            """Extract all US stock holdings from this INDmoney portfolio screenshot.
For each stock return a JSON array like:
[{"symbol":"AAPL","quantity":2.5,"avg_price":175.50,"name":"Apple Inc"},...]
Use only actual stock tickers (NVDA, META, AAPL etc).
Return ONLY the JSON array, no other text."""
                                        ]
                                    )
                                    _raw = (_gr.text or "").strip()
                                    # Extract JSON
                                    import re as _re2
                                    _jm = _re2.search(r'\[.*\]', _raw, _re2.DOTALL)
                                    if _jm:
                                        _parsed_holdings = json.loads(_jm.group())
                                        if _parsed_holdings:
                                            # Merge with existing or replace
                                            _new_h = [{"symbol":h["symbol"],"quantity":h["quantity"],
                                                       "avg_price":h["avg_price"],
                                                       "name":h.get("name",h["symbol"])} for h in _parsed_holdings]
                                            st.session_state["portfolio_holdings"] = _new_h
                                            _save_portfolio(_new_h)
                                            st.success(f"✅ Synced {len(_new_h)} holdings from INDmoney screenshot!")
                                            for _h in _new_h:
                                                st.markdown(f"• **{_h['symbol']}** — {_h['quantity']} shares @ ${_h['avg_price']:.2f}")
                                            st.rerun()
                                        else:
                                            st.warning("No holdings found. Try a clearer screenshot.")
                                    else:
                                        st.warning(f"Could not parse: {_raw[:200]}")
                                except Exception as _ie:
                                    st.error(f"Parse error: {_ie}")
                            else:
                                st.warning("Add GEMINI_API_KEY to .env to enable screenshot parsing.")

            # ══ SECTION 2: AI PORTFOLIO ANALYSIS ═════════════════════════════════
            st.markdown("---")
            st.markdown("### 🧠 AI Portfolio Analysis — Add More / Hold / Sell")

            if st.button("🔍 Analyse All Holdings Now", key="analyse_all_holdings",
                         type="primary", use_container_width=True):
                st.session_state["run_portfolio_analysis"] = True

            if st.session_state.get("run_portfolio_analysis") and holdings:
                _mkt_risk_now = _compute_market_risk_score()
                _mkt_lvl = _mkt_risk_now.get("level","LOW")
                _mkt_vix = _mkt_risk_now.get("vix")

                for _ph in holdings:
                    _psym  = _ph.get("symbol","")
                    _pqty  = float(_ph.get("quantity",1))
                    _pavg  = float(_ph.get("avg_price",0))
                    if not _psym: continue

                    with st.container(border=True):
                        _ac1, _ac2 = st.columns([1, 3])
                        with _ac1:
                            st.markdown(f"## {_psym}")
                            st.caption(f"{_pqty} shares @ ${_pavg:.2f}")

                        with _ac2:
                            with st.spinner(f"Analysing {_psym}..."):
                                try:
                                    # Fetch data
                                    import yfinance as _yfp
                                    _pdf = _yfp.download(_psym, period="5d", interval="5m",
                                                         progress=False, auto_adjust=True, prepost=True)
                                    if hasattr(_pdf.columns,"levels"): _pdf.columns=_pdf.columns.get_level_values(0)
                                    _pcur = float(_pdf["Close"].dropna().iloc[-1]) if not _pdf.empty else _pavg
                                    _ppnl = (_pcur - _pavg) / _pavg * 100

                                    # Quick signals
                                    _phfr = _hedge_fund_rules(_pdf) if not _pdf.empty and len(_pdf)>20 else {}
                                    _phscore = _phfr.get("score", 0)
                                    _phrsi   = _phfr.get("rsi", 50) or 50
                                    _phmac   = _phfr.get("macd_hist", 0) or 0
                                    _ppat    = _detect_candle_patterns(_pdf)[0] if not _pdf.empty and len(_pdf)>3 else ("—","⚫","no data")

                                    # News headlines
                                    _pnews = _get_stock_news(_psym)
                                    _pnews_str = " | ".join([_app_parse_news_item(n)[0] for n in _pnews[:3] if _app_parse_news_item(n)[0]])[:400]

                                    # ATR stop
                                    _patr = float((_pdf["High"]-_pdf["Low"]).rolling(14).mean().iloc[-1]) if not _pdf.empty else _pcur*0.02
                                    _pstop = round(_pcur - 2*_patr, 2)
                                    _padd_entry = round(_pcur * 0.99, 2)  # add 1% below current

                                    # Claude 2-line verdict
                                    _claude_key = os.getenv("ANTHROPIC_API_KEY","").strip()
                                    _verdict_txt = ""
                                    if _claude_key:
                                        try:
                                            import anthropic as _ant2
                                            _cl2 = _ant2.Anthropic(api_key=_claude_key)
                                            _vm = _cl2.messages.create(
                                                model="claude-sonnet-4-6", max_tokens=200,
                                                messages=[{"role":"user","content":
                                                    f"Stock: {_psym} | Current: ${_pcur:.2f} | Avg cost: ${_pavg:.2f} | P&L: {_ppnl:+.1f}%\n"
                                                    f"RSI: {_phrsi:.0f} | MACD hist: {_phmac:.3f} | Rule score: {_phscore}\n"
                                                    f"Last candle: {_ppat[0]} — {_ppat[2]}\n"
                                                    f"Market risk: {_mkt_lvl} (VIX {_mkt_vix})\n"
                                                    f"Recent news: {_pnews_str}\n\n"
                                                    f"I already own this stock. In EXACTLY 2 lines:\n"
                                                    f"Line 1: ADD MORE / HOLD / SELL — and the single most important reason why\n"
                                                    f"Line 2: If ADD MORE → entry price ${_padd_entry:.2f}, stop ${_pstop:.2f}. If SELL → exit at ${_pcur:.2f}, stop was ${_pstop:.2f}. If HOLD → hold with stop at ${_pstop:.2f}.\n"
                                                    f"No disclaimers. Be direct."}]
                                            )
                                            _verdict_txt = _vm.content[0].text.strip()
                                        except Exception:
                                            pass

                                    if not _verdict_txt:
                                        # Rule-based fallback
                                        if _phscore >= 5 and _ppnl > -5:
                                            _verdict_txt = f"ADD MORE — Strong momentum (score {_phscore}), RSI {_phrsi:.0f} healthy, {_ppat[0]} pattern.\nEntry: ${_padd_entry:.2f} | Stop: ${_pstop:.2f}"
                                        elif _phscore <= -4 or _phrsi > 75:
                                            _verdict_txt = f"SELL — Weak signals (score {_phscore}), RSI {_phrsi:.0f} overbought.\nExit near ${_pcur:.2f} | Stop was ${_pstop:.2f}"
                                        else:
                                            _verdict_txt = f"HOLD — Neutral signals, trend intact (score {_phscore}).\nMaintain stop at ${_pstop:.2f}"

                                    _vlines = _verdict_txt.split("\n", 1)
                                    _v1 = _vlines[0].strip()
                                    _v2 = _vlines[1].strip() if len(_vlines) > 1 else ""

                                    # Color code
                                    _vcol = ("#00c853" if "ADD" in _v1.upper()
                                             else "#ff5252" if "SELL" in _v1.upper()
                                             else "#ff9800")

                                    # Display
                                    _dm1, _dm2, _dm3 = st.columns(3)
                                    _dm1.metric("Current", f"${_pcur:.2f}",
                                               delta=f"{_ppnl:+.1f}%",
                                               delta_color="normal" if _ppnl>=0 else "inverse")
                                    _dm2.metric("Rule Score", f"{_phscore:+d}")
                                    _dm3.metric("RSI", f"{_phrsi:.0f}",
                                               delta="Overbought" if _phrsi>70 else "Oversold" if _phrsi<30 else "OK",
                                               delta_color="inverse" if _phrsi>70 else "normal" if _phrsi<30 else "off")

                                    st.markdown(
                                        f"<div style='background:#111;border-left:4px solid {_vcol};"
                                        f"padding:12px 16px;border-radius:4px;margin-top:8px'>"
                                        f"<div style='color:{_vcol};font-weight:700;font-size:1.05rem'>{_v1}</div>"
                                        f"<div style='color:#ccc;font-size:0.9rem;margin-top:4px'>{_v2}</div>"
                                        f"</div>", unsafe_allow_html=True)

                                    if _ppat[0] != "—":
                                        st.caption(f"Pattern: {_ppat[1]} {_ppat[0]} — {_ppat[2]}")

                                except Exception as _pe2:
                                    st.warning(f"Could not analyse {_psym}: {_pe2}")

            # ══ SECTION 2: Portfolio Snapshot ════════════════════════════════════
            st.markdown("---")
            st.markdown("### 💰 Portfolio Snapshot")
            snap = st.columns(5)
            snap[0].metric("Invested",       f"${total_invested:,.0f}")
            snap[1].metric("Current Value",  f"${total_current:,.0f}")
            snap[2].metric("Total P&L",      f"${total_pnl:+,.0f}", delta=f"{total_pnl_pct:+.1f}%")
            snap[3].metric("Portfolio Beta", f"{port_beta:.2f}",
                            help=">1 = amplifies market moves. Your portfolio vs S&P 500.")
            snap[4].metric("Holdings",       str(len(holdings)))
    
            # ══ SECTION 3: Factor Exposure X-Ray ════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔬 Factor Exposure — What Are You Actually Betting On?")
            st.caption("A HF manager checks this first. Most retail portfolios look diversified but are actually 1-2 concentrated factor bets.")
    
            if factor_exp:
                # Build a horizontal bar chart using HTML
                factor_rows = ""
                for factor, pct in factor_exp.items():
                    bar_clr = ("#ff5252" if pct > 0.50 else "#ffcc00" if pct > 0.30 else "#2979ff")
                    factor_rows += (
                        f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
                        f"<div style='width:160px;color:#aaa;font-size:0.82rem;text-align:right'>{factor}</div>"
                        f"<div style='flex:1;background:#1a1a1a;border-radius:4px;height:18px'>"
                        f"<div style='background:{bar_clr};width:{pct*100:.0f}%;height:100%;"
                        f"border-radius:4px'></div></div>"
                        f"<div style='width:40px;color:{bar_clr};font-weight:700;font-size:0.82rem'>"
                        f"{pct*100:.0f}%</div></div>"
                    )
                st.markdown(f"<div style='padding:10px'>{factor_rows}</div>", unsafe_allow_html=True)
    
                top_factor, top_pct = list(factor_exp.items())[0]
                if top_pct > 0.50:
                    st.warning(f"⚠️ **{top_pct*100:.0f}% of your portfolio is a single factor bet: {top_factor}.** "
                               f"One sector shock hits everything simultaneously.")
                elif top_pct > 0.35:
                    st.info(f"📊 {top_pct*100:.0f}% concentrated in {top_factor}. Acceptable but monitor.")
                else:
                    st.success("✅ Factor exposure reasonably diversified.")
    
            # ══ SECTION 4: Risk Attribution ══════════════════════════════════════
            st.markdown("---")
            st.markdown("### ⚡ Risk Attribution — Who Is Consuming Your Risk Budget?")
            st.caption("Capital allocation ≠ risk allocation. A small volatile position can dominate your risk.")
    
            if risk_rows:
                ra_header = (
                    "<table style='width:100%;border-collapse:collapse;font-size:0.82rem'>"
                    "<thead><tr style='border-bottom:2px solid #333'>"
                    "<th style='padding:6px 10px;color:#666;text-align:left'>Stock</th>"
                    "<th style='padding:6px 10px;color:#666;text-align:right'>Capital %</th>"
                    "<th style='padding:6px 10px;color:#666;text-align:right'>Risk %</th>"
                    "<th style='padding:6px 10px;color:#666;text-align:right'>Ann. Vol</th>"
                    "<th style='padding:6px 10px;color:#666;text-align:right'>Beta</th>"
                    "<th style='padding:6px 10px;color:#666'>Risk vs Capital</th>"
                    "</tr></thead><tbody>"
                )
                ra_rows = ""
                for r in risk_rows:
                    diff = r["risk_pct"] - r["val_pct"]
                    diff_clr = "#ff5252" if diff > 0.08 else ("#ffcc00" if diff > 0.03 else "#00c853")
                    diff_lbl = f"{'▲' if diff > 0 else '▼'} {abs(diff)*100:.0f}pp {'over' if diff > 0 else 'under'}-sized risk"
                    ra_rows += (
                        f"<tr style='border-bottom:1px solid #1a1a1a'>"
                        f"<td style='padding:5px 10px;color:#fff;font-weight:700'>{r['sym']}</td>"
                        f"<td style='padding:5px 10px;color:#aaa;text-align:right'>{r['val_pct']*100:.1f}%</td>"
                        f"<td style='padding:5px 10px;color:#fff;font-weight:700;text-align:right'>{r['risk_pct']*100:.1f}%</td>"
                        f"<td style='padding:5px 10px;color:#888;text-align:right'>{r['vol']*100:.1f}%</td>"
                        f"<td style='padding:5px 10px;color:#888;text-align:right'>{r['beta']:.2f}</td>"
                        f"<td style='padding:5px 10px;color:{diff_clr};font-size:0.78rem'>{diff_lbl}</td>"
                        f"</tr>"
                    )
                st.markdown(ra_header + ra_rows + "</tbody></table>", unsafe_allow_html=True)
    
            # ══ SECTION 5: Stress Tests ═══════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔥 Stress Test — What Happens to Your Portfolio In Each Scenario?")
            st.caption("HF managers run these before every trade. Know your downside before the market tells you.")
    
            stress_cols = st.columns(len(_STRESS_SCENARIOS))
            for ci, (sc_name, sc_data) in enumerate(stress_data.items()):
                pct   = sc_data["pct"]
                loss  = sc_data["total_loss"]
                clr   = "#ff5252" if pct < -0.20 else ("#ff9800" if pct < -0.10 else "#ffcc00")
                worst_sym  = min(sc_data["per_stock"].items(), key=lambda x: x[1]["impact"])[0]
                worst_pct  = sc_data["per_stock"][worst_sym]["impact"]
                stress_cols[ci].markdown(
                    f"<div style='background:#111;border:1px solid {clr}33;border-radius:8px;"
                    f"padding:10px 12px;text-align:center'>"
                    f"<div style='font-size:0.78rem;color:#888;margin-bottom:4px'>{sc_name}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{clr}'>{pct*100:+.1f}%</div>"
                    f"<div style='font-size:0.8rem;color:#aaa'>${loss:,.0f}</div>"
                    f"<div style='font-size:0.72rem;color:#555;margin-top:4px'>"
                    f"Worst: {worst_sym} {worst_pct*100:+.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    
            worst_scenario = min(stress_data.items(), key=lambda x: x[1]["pct"])
            st.caption(
                f"💡 Worst scenario: **{worst_scenario[0]}** — portfolio down "
                f"${abs(worst_scenario[1]['total_loss']):,.0f} ({worst_scenario[1]['pct']*100:.1f}%). "
                f"Can you stomach that loss? If yes, you're sized correctly. If no, reduce position sizes."
            )
    
            # ══ SECTION 6: Buy / Sell / Hold Decision Table ═══════════════════════
            st.markdown("---")
            st.markdown("### 🎯 Position Decisions — Should You Buy More, Hold, or Sell?")
            st.caption(
                "Based on: DCF intrinsic value (margin of safety entry), Kelly Criterion sizing, "
                "ATR-based trailing stop, and profit milestone rules."
            )
    
            for h in holdings:
                sym = h["symbol"]
                d   = decisions.get(sym)
                res = analysis_results.get(sym)
                if not d or not res:
                    continue
    
                gain_pct  = d["gain_pct"]
                gain_clr  = "#00c853" if gain_pct > 0 else "#ff5252"
                act_clr   = d["action_clr"]
    
                with st.expander(
                    f"**{sym}** · {d['action']} · "
                    f"P&L {gain_pct*100:+.1f}% · Entry ${d['optimal_entry']:.2f} · Stop ${d['trail_stop']:.2f}",
                    expanded=True
                ):
                    # Valuation model warning for hypergrowth stocks
                    if d.get("is_hypergrowth") and d.get("val_warning"):
                        st.markdown(
                            f"<div style='background:#1a1000;border:1px solid #ff9800;"
                            f"border-radius:6px;padding:8px 14px;margin-bottom:8px;font-size:0.82rem'>"
                            f"<span style='color:#ff9800;font-weight:700'>⚠️ Valuation model: </span>"
                            f"<span style='color:#aaa'>{d['val_warning']}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if d.get("dcf_intrinsic"):
                            st.caption(
                                f"📊 Standard DCF gives ${d['dcf_intrinsic']:.2f} — this is a model artifact, not the real fair value. "
                                f"DCF caps growth at 40%/yr and applies 14%+ WACC, which crushes hypergrowth valuations. "
                                f"The model above uses **{d['val_model_label']}** instead."
                            )
    
                    dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
                    dc1.metric("Your Avg Cost",    f"${d['avg_price']:.2f}")
                    dc2.metric("Current Price",    f"${d['curr_px']:.2f}", delta=f"{gain_pct*100:+.1f}%")
                    # Show appropriate fair value label depending on model used
                    fv_label = "Earnings Power" if d.get("is_hypergrowth") else "DCF Intrinsic"
                    fv_help  = d.get("val_model_label", "Fair value estimate")
                    dc3.metric(fv_label, f"${d['intrinsic']:.2f}" if d["intrinsic"] else "N/A", help=fv_help)
                    dc4.metric("Optimal Entry",    f"${d['optimal_entry']:.2f}", help=d["entry_basis"])
                    dc5.metric("Profit Target",    f"${d['profit_target']:.2f}", help=d["target_basis"])
                    dc6.metric("Trailing Stop",    f"${d['trail_stop']:.2f}",    help=d["stop_basis"])
    
                    # Action banner
                    st.markdown(
                        f"<div style='background:#0d0d0d;border-left:4px solid {act_clr};"
                        f"border-radius:6px;padding:10px 16px;margin:8px 0'>"
                        f"<span style='color:{act_clr};font-weight:800;font-size:1rem'>{d['action']}</span>"
                        f"<span style='color:#888;font-size:0.83rem;margin-left:14px'>{d['action_msg']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    
                    # Decision logic breakdown
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1:
                        st.markdown("**📥 When to Buy More**")
                        if d.get("is_hypergrowth"):
                            # For hypergrowth: use PEG and MA50 logic, not DCF
                            peg_v = _safe_float(res["f"].get("peg"))
                            if peg_v and peg_v < 1.5:
                                st.success(f"✅ PEG {peg_v:.2f} — reasonable valuation for growth rate. Add at ${d['optimal_entry']:.2f}.")
                            elif peg_v and peg_v < 2.5:
                                st.info(f"📊 PEG {peg_v:.2f} — premium but justified if growth sustains. Enter at ${d['optimal_entry']:.2f} pullback.")
                            elif peg_v:
                                st.warning(f"⚠️ PEG {peg_v:.2f} — expensive. Only add if you have very high conviction. Wait for ${d['optimal_entry']:.2f}.")
                            else:
                                st.info(f"Add on pullback to ${d['optimal_entry']:.2f}. {d['entry_basis']}")
                        elif d["intrinsic"] and d["curr_px"] < d["intrinsic"]:
                            upside = (d["intrinsic"] / d["curr_px"] - 1) * 100
                            st.success(f"✅ Below DCF fair value. {upside:.1f}% upside to intrinsic. Add now or on dip.")
                        else:
                            gap = ((d["curr_px"] / d["intrinsic"] - 1) * 100) if d["intrinsic"] else 0
                            st.warning(f"⚠️ Trading {gap:.1f}% above fair value. Wait for pullback to ${d['optimal_entry']:.2f}.")
                        st.caption(f"Entry basis: {d['entry_basis']}")
                        if d["kelly_pct"] > 0:
                            st.markdown(f"**Kelly sizing**: Add up to **{d['kelly_pct']:.1f}% of portfolio** if adding.")
    
                    with lc2:
                        st.markdown("**📤 When to Take Profit**")
                        if d["sell_rule"]:
                            st.warning(f"⚠️ Profit milestone triggered: {d['sell_rule']}")
                            st.markdown(f"Consider selling **{d['sell_pct']*100:.0f}%** of position.")
                        else:
                            st.info(f"Hold until ${d['profit_target']:.2f} target.")
                        st.caption(f"Target basis: {d['target_basis']}")
    
                    with lc3:
                        st.markdown("**🛡️ Stop Loss Management**")
                        stop_dist = (d["curr_px"] - d["trail_stop"]) / d["curr_px"] * 100
                        st.markdown(
                            f"Trail stop at **${d['trail_stop']:.2f}**  \n"
                            f"({stop_dist:.1f}% below current price)  \n"
                            f"Locks in {max(0, (d['trail_stop'] - d['avg_price']) / d['avg_price'] * 100):.1f}% "
                            f"gain from your cost basis."
                        )
                        st.caption(d["stop_basis"])
    
                    # Stress impact on this specific position
                    f_h = res["f"]
                    worst_impact_name = min(_STRESS_SCENARIOS.keys(),
                        key=lambda s: _STRESS_SCENARIOS[s]["hits"].get(sym, _STRESS_SCENARIOS[s]["default"]))
                    worst_impact_pct  = _STRESS_SCENARIOS[worst_scenario[0]]["hits"].get(sym,
                        _STRESS_SCENARIOS[worst_scenario[0]]["default"])
                    pos_val = float(h["qty"]) * d["curr_px"]
                    pos_loss = pos_val * worst_impact_pct
                    st.caption(
                        f"**Worst stress scenario for {sym}**: {worst_scenario[0]} → "
                        f"estimated {worst_impact_pct*100:.0f}% decline = ${abs(pos_loss):,.0f} loss on this position."
                    )
    
            # ══ SECTION 7: Portfolio Correlation Matrix ═══════════════════════════
            st.markdown("---")
            port_syms = [h["symbol"] for h in holdings if analysis_results.get(h["symbol"])]
            if len(port_syms) >= 2:
                st.markdown("### 🔗 Return Correlation Matrix")
                st.caption(">0.70 correlation = these positions move together — you don't have true diversification.")
                with st.spinner("Computing 6-month correlations…"):
                    corr_df, warn_pairs = compute_correlation_matrix(tuple(port_syms))
                if corr_df is not None:
                    import plotly.graph_objects as _go_port2
                    fig_p = _go_port2.Figure(data=_go_port2.Heatmap(
                        z=corr_df.values, x=corr_df.columns.tolist(), y=corr_df.index.tolist(),
                        colorscale=[[0,"#003366"],[0.5,"#111"],[1,"#660000"]],
                        zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                        texttemplate="%{text}", textfont={"size":11}, showscale=True,
                    ))
                    fig_p.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#ccc"), margin=dict(l=10,r=10,t=30,b=10),
                        height=max(300, len(port_syms)*45+80))
                    st.plotly_chart(fig_p, use_container_width=True)
                    if warn_pairs:
                        st.warning("⚠️ High correlation pairs (>0.70): " +
                                   ", ".join(f"{a}↔{b} ({v:+.2f})" for a,b,v in warn_pairs))
                    else:
                        st.success("✅ All positions correlation <0.70 — genuinely diversified.")
    
        st.markdown("---")
        st.caption("⚠️ DISCLAIMER: Portfolio analysis is for informational and educational purposes only. "
                   "Not financial advice. All models use probabilistic estimates. Past performance does not guarantee future results.")

    with pt_tab2:
        st.markdown("### 📊 Portfolio & Watchlist — Price Predictions")

        _ph = load_price_history()
        _wl = load_watchlist()
        _pf = []
        if PORTFOLIO_FILE.exists():
            try: _pf = json.loads(PORTFOLIO_FILE.read_text())
            except Exception: pass

        # Merge portfolio + watchlist into one tracking universe
        _tracked = {}
        for h in _pf:
            sym = h["symbol"]
            _tracked[sym] = {
                "symbol": sym, "name": h.get("name", sym),
                "avg_price": h.get("avg_price", 0),
                "qty": h.get("qty", 0),
                "source": "portfolio",
            }
        for w in _wl:
            sym = w["symbol"]
            if sym not in _tracked:
                _tracked[sym] = {
                    "symbol": sym, "name": w.get("name", sym),
                    "avg_price": w.get("entry", w.get("price", 0)),
                    "qty": 0,
                    "source": "watchlist",
                }

        _pct_mode = st.toggle("Show in % mode", value=True, key="pt2_pct_mode")

        if not _tracked:
            st.info("No portfolio or watchlist stocks found. Add stocks to your watchlist or sync your portfolio.")
        else:
            # Fetch current prices for all
            _syms = list(_tracked.keys())
            with st.spinner("Fetching latest prices…"):
                for sym in _syms:
                    try:
                        px = float(yf.Ticker(sym).fast_info.last_price or 0)
                        if px > 0:
                            _tracked[sym]["current_price"] = px
                    except Exception:
                        _tracked[sym]["current_price"] = _tracked[sym]["avg_price"]

            # Build table rows
            rows_html = ""
            _cal = load_calibration()
            for sym, td in sorted(_tracked.items()):
                ep  = td.get("avg_price", 0)
                cp  = td.get("current_price", ep)
                chg = ((cp - ep) / ep * 100) if ep > 0 else 0
                src_badge = (
                    '<span style="background:#002010;color:#00c853;border:1px solid #00c853;'
                    'border-radius:4px;padding:1px 7px;font-size:0.68rem">Portfolio</span>'
                    if td["source"] == "portfolio" else
                    '<span style="background:#001133;color:#2979ff;border:1px solid #2979ff;'
                    'border-radius:4px;padding:1px 7px;font-size:0.68rem">Watchlist</span>'
                )
                chg_clr = "#00c853" if chg >= 0 else "#ff5252"

                # Predictions from price_history or watchlist
                ph_data = _ph.get(sym, {})
                ph_preds = ph_data.get("predictions", {})
                # Also check watchlist targets
                wl_entry = next((w for w in _wl if w["symbol"] == sym), {})
                wl_tgts  = wl_entry.get("targets", {})

                def _get_pred(h):
                    # Price history takes priority (recorded at add time)
                    if h in ph_preds:
                        return ph_preds[h].get("return_pct", 0), ph_preds[h].get("target", 0)
                    if h in wl_tgts:
                        r = wl_tgts[h].get("ret", 0)
                        p = wl_tgts[h].get("price", 0)
                        return r, p
                    return None, None

                cells = ""
                for h in ["1M","3M","6M","1Y"]:
                    ret, tgt = _get_pred(h)
                    if ret is None:
                        cells += f'<td style="color:#444;text-align:right;padding:6px 12px">—</td>'
                    else:
                        adj_ret = round(ret * _cal.get("return_multiplier", 1.0), 1)
                        clr = "#00c853" if adj_ret >= 0 else "#ff5252"
                        if _pct_mode:
                            val = f'{adj_ret:+.1f}%'
                        else:
                            val = f'{curr}{tgt:.2f}' if tgt else f'{adj_ret:+.1f}%'
                        cells += (
                            f'<td style="color:{clr};font-weight:600;text-align:right;padding:6px 12px">'
                            f'{val}</td>'
                        )

                qty_str = f"{td.get('qty',0):.4f} shares" if td.get('qty',0) else "—"

                rows_html += f"""
            <tr style="border-bottom:1px solid #1a1a1a">
              <td style="padding:8px 12px;white-space:nowrap">
                {src_badge}
                <span style="color:#fff;font-weight:700;margin-left:8px">{sym}</span>
                <span style="color:#666;font-size:0.78rem;margin-left:6px">{td['name'][:22]}</span>
              </td>
              <td style="text-align:right;padding:6px 12px;color:#aaa">{qty_str}</td>
              <td style="text-align:right;padding:6px 12px;color:#aaa">{curr}{ep:.2f}</td>
              <td style="text-align:right;padding:6px 12px;color:#fff;font-weight:600">{curr}{cp:.2f}</td>
              <td style="text-align:right;padding:6px 12px;color:{chg_clr};font-weight:700">{chg:+.1f}%</td>
              {cells}
            </tr>"""

            # Calibration warning
            if _cal.get("n_validations", 0) >= 5:
                mult = _cal.get("return_multiplier", 1.0)
                corr_str = f"Model multiplier: {mult:.2f}x (from {_cal['n_validations']} validations)"
                st.caption(f"📐 Calibrated predictions · {corr_str}")
            else:
                st.caption("📐 Predictions shown at face value — validate more data points to calibrate the model.")

            st.markdown(f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
          <thead>
            <tr style="background:#111;color:#555;font-size:0.72rem">
              <th style="text-align:left;padding:8px 12px">Stock</th>
              <th style="text-align:right;padding:6px 12px">Qty</th>
              <th style="text-align:right;padding:6px 12px">Avg / Entry</th>
              <th style="text-align:right;padding:6px 12px">Live Price</th>
              <th style="text-align:right;padding:6px 12px">P&amp;L %</th>
              <th style="text-align:right;padding:6px 12px">1M {'%' if _pct_mode else 'Target'}</th>
              <th style="text-align:right;padding:6px 12px">3M {'%' if _pct_mode else 'Target'}</th>
              <th style="text-align:right;padding:6px 12px">6M {'%' if _pct_mode else 'Target'}</th>
              <th style="text-align:right;padding:6px 12px">1Y {'%' if _pct_mode else 'Target'}</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table></div>""", unsafe_allow_html=True)

            st.divider()
            if st.button("📡 Sync Today's Prices to History", key="pt2_sync_px", type="primary", use_container_width=False):
                with st.spinner("Syncing prices…"):
                    done = sync_daily_prices(_syms)
                st.success(f"Updated prices for {len(done)} stocks.")

            # ── 15-Day Daily Tracking Table ───────────────────────────────────
            st.divider()
            st.markdown("#### 📅 15-Day Daily Price vs Prediction")
            st.caption(
                "Predicted price = linear interpolation from entry → MC target for the relevant horizon. "
                "Delta = how far today's price is from what the model expected on that day."
            )

            _d15_sym = st.selectbox(
                "Select stock to inspect",
                options=sorted(_tracked.keys()),
                key="pt2_d15_sym",
            )

            if _d15_sym:
                _d15_ph   = _ph.get(_d15_sym, {})
                _d15_snaps = sorted(_d15_ph.get("daily_prices", []), key=lambda x: x["date"])
                _d15_ep   = float(_d15_ph.get("entry_price", 0) or
                                  _tracked[_d15_sym].get("avg_price", 0))
                _d15_added = _d15_ph.get("added", "")
                _d15_preds = _d15_ph.get("predictions", {})

                # Also pull watchlist targets as fallback
                _d15_wl = next((w for w in _wl if w["symbol"] == _d15_sym), {})
                _d15_tgts = _d15_wl.get("targets", {})

                def _d15_get_target(h):
                    if h in _d15_preds:
                        return float(_d15_preds[h].get("target", 0) or 0)
                    if h in _d15_tgts:
                        return float(_d15_tgts[h].get("price", 0) or 0)
                    return 0.0

                _d15_targets = {
                    "1M":  (_d15_get_target("1M"),  30),
                    "3M":  (_d15_get_target("3M"),  90),
                    "6M":  (_d15_get_target("6M"), 180),
                    "1Y":  (_d15_get_target("1Y"), 365),
                }

                def _interp_predicted(day_n: int) -> float | None:
                    """Linear interpolation: entry → nearest horizon target."""
                    if _d15_ep <= 0:
                        return None
                    for h in ["1M", "3M", "6M", "1Y"]:
                        tgt_px, h_days = _d15_targets[h]
                        if tgt_px > 0 and day_n <= h_days:
                            return _d15_ep + (tgt_px - _d15_ep) * (day_n / h_days)
                    # Beyond 1Y — extrapolate from 1Y target
                    tgt_px, h_days = _d15_targets["1Y"]
                    if tgt_px > 0 and h_days > 0:
                        return _d15_ep + (tgt_px - _d15_ep) * (day_n / h_days)
                    return None

                # Take last 15 snapshots
                _d15_last = _d15_snaps[-15:] if len(_d15_snaps) >= 15 else _d15_snaps

                if not _d15_last:
                    st.info(
                        f"No daily price data yet for **{_d15_sym}**. "
                        f"Click **📡 Sync Today's Prices** above or add it to the watchlist first."
                    )
                else:
                    import datetime as _dtmod
                    # Build table rows
                    _d15_rows = ""
                    for snap in _d15_last:
                        snap_date = snap["date"]
                        actual_px = float(snap["price"])

                        # Calculate day number since entry
                        day_n = 0
                        if _d15_added:
                            try:
                                _base = _dtmod.date.fromisoformat(_d15_added[:10])
                                _snap_d = _dtmod.date.fromisoformat(snap_date[:10])
                                day_n = (_snap_d - _base).days
                            except Exception:
                                day_n = 0

                        pred_px   = _interp_predicted(day_n)
                        if pred_px and pred_px > 0 and actual_px > 0:
                            delta_abs = actual_px - pred_px
                            delta_pct = delta_abs / pred_px * 100
                            clr = "#00c853" if delta_abs >= 0 else "#ff5252"
                            arrow = "▲" if delta_abs >= 0 else "▼"
                            pred_str  = f"{curr}{pred_px:.2f}"
                            delta_str = f"{arrow} {curr}{abs(delta_abs):.2f} ({delta_pct:+.1f}%)"
                        else:
                            pred_str  = "—"
                            delta_str = "—"
                            clr       = "#444"

                        # Colour the row lightly if delta is large
                        row_bg = ""
                        if pred_px and actual_px:
                            dp = abs((actual_px - pred_px) / pred_px * 100)
                            row_bg = (
                                "background:#001a00;" if actual_px > pred_px and dp > 2 else
                                "background:#1a0000;" if actual_px < pred_px and dp > 2 else ""
                            )

                        _d15_rows += f"""
                        <tr style="border-bottom:1px solid #1a1a1a;{row_bg}">
                          <td style="padding:7px 14px;color:#aaa;font-size:0.8rem">{snap_date}</td>
                          <td style="text-align:right;padding:7px 14px;color:#fff;font-weight:600">
                            {curr}{actual_px:.2f}</td>
                          <td style="text-align:right;padding:7px 14px;color:#555">
                            {pred_str}</td>
                          <td style="text-align:right;padding:7px 14px;color:{clr};font-weight:700">
                            {delta_str}</td>
                        </tr>"""

                    # Summary stats
                    _d15_deltas = []
                    for snap in _d15_last:
                        ap = float(snap["price"])
                        try:
                            _base = _dtmod.date.fromisoformat(_d15_added[:10])
                            _sd   = _dtmod.date.fromisoformat(snap["date"][:10])
                            dn    = (_sd - _base).days
                        except Exception:
                            dn = 0
                        pp = _interp_predicted(dn)
                        if pp and pp > 0 and ap > 0:
                            _d15_deltas.append((ap - pp) / pp * 100)

                    _avg_delta = sum(_d15_deltas) / len(_d15_deltas) if _d15_deltas else 0
                    _max_delta = max(_d15_deltas, key=abs) if _d15_deltas else 0
                    _above_n   = sum(1 for d in _d15_deltas if d > 0)
                    _below_n   = len(_d15_deltas) - _above_n

                    _sc1, _sc2, _sc3 = st.columns(3)
                    with _sc1:
                        st.metric("Avg Delta (15D)",
                                  f"{_avg_delta:+.1f}%",
                                  delta_color="normal" if _avg_delta >= 0 else "inverse")
                    with _sc2:
                        st.metric("Days Above Prediction", f"{_above_n}/{len(_d15_deltas)}")
                    with _sc3:
                        st.metric("Max Deviation", f"{_max_delta:+.1f}%",
                                  delta_color="off")

                    st.markdown(f"""
                    <div style="overflow-x:auto;margin-top:10px">
                    <table style="width:100%;border-collapse:collapse;font-size:0.84rem">
                      <thead>
                        <tr style="background:#111;color:#555;font-size:0.72rem;text-transform:uppercase">
                          <th style="text-align:left;padding:8px 14px">Date</th>
                          <th style="text-align:right;padding:8px 14px">Actual Price</th>
                          <th style="text-align:right;padding:8px 14px">Predicted Price</th>
                          <th style="text-align:right;padding:8px 14px">Delta</th>
                        </tr>
                      </thead>
                      <tbody>{_d15_rows}</tbody>
                    </table></div>""", unsafe_allow_html=True)

                    st.caption(
                        f"Entry: {_d15_added} @ {curr}{_d15_ep:.2f} · "
                        f"Predicted path interpolated linearly to MC targets · "
                        f"Green row = tracking above model · Red = tracking below"
                    )

    with pt_tab3:
        st.markdown("### ✅ Validate Predictions & Train the Model")
        st.caption(
            "Click **Check Due** to find predictions whose time horizon has elapsed. "
            "The model compares predicted vs actual return, explains the delta with AI, "
            "and adjusts future predictions to be more accurate."
        )

        _ph  = load_price_history()
        _vlog = load_validation_log()
        _cal  = load_calibration()

        # Metrics summary
        v_c1, v_c2, v_c3, v_c4 = st.columns(4)
        with v_c1:
            st.metric("Validations Done", _cal.get("n_validations", 0))
        with v_c2:
            dir_acc = _cal.get("direction_accuracy", 0)
            st.metric("Direction Accuracy", f"{dir_acc*100:.0f}%",
                      delta="Good" if dir_acc >= 0.65 else ("OK" if dir_acc >= 0.5 else "Poor"),
                      delta_color="normal" if dir_acc >= 0.65 else ("off" if dir_acc >= 0.5 else "inverse"))
        with v_c3:
            mae = _cal.get("mean_abs_error", 0)
            st.metric("Mean Abs Error", f"{mae:.1f}%")
        with v_c4:
            mult = _cal.get("return_multiplier", 1.0)
            st.metric("Return Multiplier", f"{mult:.2f}x",
                      delta="calibrated" if _cal.get("n_validations",0)>=5 else "not enough data",
                      delta_color="normal" if _cal.get("n_validations",0)>=5 else "off")

        st.divider()

        v_btn_col1, v_btn_col2, v_btn_col3 = st.columns([1.5, 1.5, 3])
        with v_btn_col1:
            _run_check = st.button("🔍 Check Due Predictions", key="pt3_check", type="primary", use_container_width=True)
        with v_btn_col2:
            _sync_and_check = st.button("📡 Sync Prices + Check", key="pt3_sync_check", use_container_width=True)

        if _sync_and_check:
            all_syms = list(_ph.keys())
            with st.spinner(f"Syncing {len(all_syms)} stocks…"):
                sync_daily_prices(all_syms)
            _ph = load_price_history()
            st.success("Prices synced.")
            _run_check = True  # fall through to check

        if _run_check:
            new_entries = check_due_predictions(_ph, _vlog)
            if not new_entries:
                st.info("No predictions are due yet, or all due ones have already been validated.")
            else:
                st.success(f"Found {len(new_entries)} due prediction(s) to validate.")
                # Run delta analysis + save
                for entry in new_entries:
                    with st.spinner(f"Analysing delta for {entry['ticker']} ({entry['horizon']})…"):
                        expl, factors = analyze_delta_with_gemini(
                            entry["ticker"],
                            entry["predicted_return_pct"],
                            entry["actual_return_pct"],
                            entry["added"],
                            entry["validation_date"],
                        )
                        entry["delta_explanation"] = expl
                        entry["delta_factors"]     = factors
                _vlog.extend(new_entries)
                save_validation_log(_vlog)
                # Recalibrate
                new_cal = recalculate_calibration(_vlog)
                save_calibration(new_cal)
                st.success(
                    f"Model updated: direction accuracy {new_cal['direction_accuracy']*100:.0f}%, "
                    f"return multiplier {new_cal['return_multiplier']:.2f}x"
                )
                st.rerun()

        # ── Validation History Table ────────────────────────────────────────────────────────────────────
        if _vlog:
            st.markdown("#### 📋 Validation History")

            # Sort newest first
            _vlog_sorted = sorted(_vlog, key=lambda x: x.get("validation_date",""), reverse=True)

            for v in _vlog_sorted:
                pred_r = v.get("predicted_return_pct", 0)
                act_r  = v.get("actual_return_pct", 0)
                err    = v.get("model_error_pct", 0)
                dev    = v.get("deviation_pct", 0)
                dir_ok = v.get("direction_correct", False)

                pred_clr = "#00c853" if pred_r >= 0 else "#ff5252"
                act_clr  = "#00c853" if act_r  >= 0 else "#ff5252"
                dir_icon = "✅" if dir_ok else "❌"
                err_clr  = "#00c853" if abs(err) <= 5 else "#ff9100" if abs(err) <= 15 else "#ff5252"

                # Deviation bar (0–30%)
                dev_pct  = min(int(dev / 30 * 100), 100)
                dev_bar  = (
                    f'<div style="background:#222;border-radius:3px;height:6px;width:100%;margin-top:3px">'
                    f'<div style="background:{err_clr};width:{dev_pct}%;height:100%;border-radius:3px"></div>'
                    f'</div>'
                )
                factors_str = " · ".join(v.get("delta_factors", [])) or "—"
                expl = v.get("delta_explanation") or "No analysis available."

                with st.expander(
                    f"{v['ticker']} · {v['horizon']} · "
                    f"predicted {pred_r:+.1f}% → actual {act_r:+.1f}% · "
                    f"deviation {dev:.1f}% {dir_icon}",
                    expanded=False,
                ):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Predicted", f"{pred_r:+.1f}%")
                    with col_b:
                        st.metric("Actual", f"{act_r:+.1f}%")
                    with col_c:
                        st.metric("Model Error", f"{err:+.1f}%",
                                  delta_color="off" if abs(err)<=5 else "inverse")
                    with col_d:
                        st.metric("Direction", dir_icon)

                    st.markdown(
                        f'Deviation: **{dev:.1f}%** {dev_bar}',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Factors:** {factors_str}")
                    st.info(expl)
                    st.caption(
                        f"Entry: {v.get('added','?')}" + f" @ {curr}{v.get('entry_price',0):.2f} · "
                        f"Validated: {v.get('validation_date','?')}" + f" @ {curr}{v.get('actual_price',0):.2f}"
                    )
        else:
            st.info(
                "No validations yet. Add stocks to your watchlist, wait for their prediction "
                "horizon to elapse (1M / 3M / 6M / 1Y), then click **Check Due Predictions**."
            )



# ══════════════════════════════════════════════════════════════════════════════
# EARLY STAGE DISCOVERY ENGINE — Hidden Gem Framework
# ══════════════════════════════════════════════════════════════════════════════

# Emerging industries with seed tickers, TAM context, and why they're undervalued
EMERGING_INDUSTRIES = {
    "⚛️ Quantum Computing": {
        "why": "Will break current encryption and solve optimization problems impossible for classical computers. Most investors don't understand it — they avoid it.",
        "catalyst": "First fault-tolerant quantum computer. Enterprise contracts from banks/pharma.",
        "tam": "$850B+ TAM by 2040. Currently <$2B in revenue globally.",
        "boring_factor": "Too technical to understand. Seems like 'sci-fi'. Years from mainstream.",
        "seeds_us": ["IONQ", "RGTI", "QUBT", "ARQQ", "QMCO", "QTUM"],
        "seeds_in": [],
        "horizon": "1–3 years",
    },
    "🔋 Small Modular Reactors (SMR)": {
        "why": "AI data centers need 10–100× more power. SMRs are the only clean, always-on baseload solution. Nuclear = old and scary to most investors.",
        "catalyst": "First commercial SMR power purchase agreement. NRC approval.",
        "tam": "$300B TAM by 2040. Microsoft/Google already signed SMR deals.",
        "boring_factor": "Nuclear is 50-year-old technology. Most ESG funds avoid it. Media treats it negatively.",
        "seeds_us": ["SMR", "OKLO", "BWXT", "LEU", "NNE", "UUUU", "UEC"],
        "seeds_in": ["RVNL.NS", "LTTS.NS"],
        "horizon": "2–5 years",
    },
    "🛸 Satellite-to-Mobile (Direct-to-Device)": {
        "why": "Every phone on earth gets satellite coverage without a special device. Eliminates dead zones. Massive for insurance, IoT, emergency services.",
        "catalyst": "Commercial launch + carrier agreements. FCC approval.",
        "tam": "$100B+ TAM. Carriers pay per subscriber connected.",
        "boring_factor": "Satellite companies have failed before (Iridium, etc.). History makes investors cautious.",
        "seeds_us": ["ASTS", "IRDM", "GSAT", "SPOK"],
        "seeds_in": [],
        "horizon": "1–2 years",
    },
    "🧬 Longevity / Anti-Aging Biotech": {
        "why": "Drugs that extend healthy lifespan by targeting aging itself. GLP-1 showed biology can be hacked. Rapamycin, senolytics, NAD+ are next.",
        "catalyst": "Clinical trial data on aging biomarkers. FDA accepting 'aging' as treatable disease.",
        "tam": "$600B+ TAM. Every human is the customer.",
        "boring_factor": "Seen as science fiction. No clear FDA pathway yet. Long timelines.",
        "seeds_us": ["LLYP", "ARWR", "UNITY", "ADTX", "SRRK", "BIO"],
        "seeds_in": [],
        "horizon": "2–5 years",
    },
    "🛡️ Defence Tech (Autonomous + AI-Driven)": {
        "why": "Ukraine/Israel/Taiwan conflicts driving massive re-arming. AI-driven drones, cyber defence, and electronic warfare replacing legacy hardware.",
        "catalyst": "Government contracts. NATO/QUAD defence spending increases.",
        "tam": "$2T global defence spending. AI defence is <1% of that.",
        "boring_factor": "Defence is 'old economy'. Ethical concerns. Contracts take years.",
        "seeds_us": ["KTOS", "AVAV", "RCAT", "JOBY", "ACHR", "PLTR"],
        "seeds_in": ["PARAS.NS", "MTAR.NS", "DRALAL.NS", "BEL.NS", "HAL.NS"],
        "horizon": "1–3 years",
    },
    "🏗️ AI Infrastructure (Not Nvidia — the boring enablers)": {
        "why": "Everyone buys NVDA GPUs. Nobody buys the power management, cooling, and data centre networking companies that make them run. Classic 'picks and shovels' play.",
        "catalyst": "AI capex continues. Hyperscaler earnings calling out specific bottlenecks.",
        "tam": "AI data center market $700B by 2030. Power + cooling = 20-30% of that.",
        "boring_factor": "Power management and cooling are not exciting. No headlines.",
        "seeds_us": ["VRT", "SMCI", "CRDO", "LSCC", "AEIS", "ACLS", "ONTO"],
        "seeds_in": ["IDFCFIRSTB.NS", "CESC.NS", "TORNTPOWER.NS"],
        "horizon": "6–18 months",
    },
    "🌿 Precision / Vertical Farming": {
        "why": "Climate change + water scarcity is making outdoor farming unreliable. Vertical farms use 95% less water, no pesticides, 365-day yield.",
        "catalyst": "Cost of LED lighting falling 40%/year. First profitable vertical farm IPO.",
        "tam": "$50B TAM by 2030. Food security is a national priority.",
        "boring_factor": "Agriculture = slow and low-margin in investor minds.",
        "seeds_us": ["APPH", "IIPR", "VITL", "AGRI", "NVTS"],
        "seeds_in": ["KAVERI.NS", "PIIND.NS", "COROMANDEL.NS"],
        "horizon": "2–4 years",
    },
    "🧠 Brain-Computer Interface (BCI)": {
        "why": "Neuralink showed it's real. Next: treating paralysis, depression, memory loss. Then: human-AI interface. Adjacents are public and cheap.",
        "catalyst": "First FDA-approved BCI for paralysis treatment. Neuralink going public.",
        "tam": "$20B TAM by 2030 (medical). $500B+ eventually (consumer).",
        "boring_factor": "Sounds like sci-fi. Only Neuralink is known — it's private. Public adjacents are ignored.",
        "seeds_us": ["NURO", "PRCT", "CEVA", "MXGY", "AXNX"],
        "seeds_in": [],
        "horizon": "2–5 years",
    },
    "🌊 Water Technology": {
        "why": "Fresh water scarcity is the silent crisis. Desalination, water recycling, pipe leak detection. Every country's infrastructure is aging.",
        "catalyst": "Drought events triggering emergency infrastructure spending. Government mandates.",
        "tam": "$1T+ global water infrastructure market.",
        "boring_factor": "Water utilities are seen as boring/regulated. No growth narrative.",
        "seeds_us": ["ERII", "CWCO", "ARTW", "MSEX", "YORW", "PNTM"],
        "seeds_in": ["VA.NS", "WELSPUNIND.NS", "WABAG.NS", "KRBL.NS"],
        "horizon": "1–3 years",
    },
    "⚡ Grid-Scale Energy Storage": {
        "why": "Solar and wind are intermittent. The missing link is cheap long-duration storage. Whoever solves this = $10T market.",
        "catalyst": "Battery price below $50/kWh. Utility contract announcements.",
        "tam": "$500B+ TAM by 2035. Every solar/wind project needs storage.",
        "boring_factor": "Not batteries for cars — boring utility infrastructure. Analysts don't cover it.",
        "seeds_us": ["FLUX", "STEM", "AMPS", "FREYR", "BLNK", "NRGV"],
        "seeds_in": ["AMARARAJA.NS", "EXIDE.NS", "GREENZO.NS"],
        "horizon": "2–4 years",
    },
    "💊 GLP-1 Supply Chain (Not Novo/Eli Lilly — the enablers)": {
        "why": "GLP-1 drugs are the fastest-growing drug class ever. But Novo Nordisk and Eli Lilly need raw materials, cold chain, injection devices, and contract manufacturers. These are cheap and ignored.",
        "catalyst": "GLP-1 prescription volumes double. API supply contracts announced.",
        "tam": "$130B GLP-1 market by 2030. Supply chain = 15-20% of that.",
        "boring_factor": "Contract pharma manufacturing is not sexy. Raw material companies are never covered.",
        "seeds_us": ["CTLT", "PACB", "AMAG", "LHCG", "PRGO", "PCRX"],
        "seeds_in": ["DIVIS.NS", "SOLARA.NS", "NEULAND.NS", "SUVEN.NS"],
        "horizon": "1–2 years",
    },
    "🔭 Custom Industry": {
        "why": "Enter your own emerging industry for AI-powered discovery.",
        "catalyst": "User-defined",
        "tam": "User-defined",
        "boring_factor": "User-defined",
        "seeds_us": [],
        "seeds_in": [],
        "horizon": "User-defined",
    },
}

# Hidden gem scoring weights — opposite of momentum screener
# We WANT: low coverage, flat stock, great fundamentals
HIDDEN_GEM_WEIGHTS = {
    "undiscovery": 0.35,   # low analyst count, small cap, not in indices
    "fundamental":  0.30,  # revenue growth, margins, cash
    "early_timing": 0.25,  # stock hasn't run yet, price is flat/sideways
    "management":   0.10,  # insider ownership, insider buying
}


@st.cache_data(ttl=600, show_spinner=False)
def compute_hidden_gem_score(sym):
    """
    Score a stock as a 'hidden gem' early-stage opportunity.
    Returns (score 0-100, breakdown_dict, features_dict) or None.
    """
    try:
        tk   = yf.Ticker(sym)
        info = tk.info or {}
        hist = tk.history(period="1y", interval="1d")
        if hist.empty:
            return None

        def si(k): return _safe_float(info.get(k))

        price    = float(hist["Close"].iloc[-1])
        mktcap   = si("marketCap") or 0
        n_ana    = int(info.get("numberOfAnalystOpinions") or 0)
        rev_g    = si("revenueGrowth") or 0
        earn_g   = si("earningsGrowth") or 0
        gross_m  = si("grossMargins") or 0
        insider  = si("heldPercentInsiders") or 0
        short_p  = si("shortPercentOfFloat") or 0
        curr_r   = si("currentRatio") or 1
        de       = si("debtToEquity") or 0
        pe       = si("trailingPE")
        ps       = si("priceToSalesTrailingTwelveMonths") or 0
        fcf_y    = si("freeCashflow")
        fcf_yield = (fcf_y / mktcap) if (fcf_y and mktcap > 0) else None

        # Price momentum — we WANT flat/boring stock
        close = hist["Close"].dropna()
        mom_6m = (float(close.iloc[-1]) / float(close.iloc[-126]) - 1) if len(close) >= 126 else 0
        mom_1y = (float(close.iloc[-1]) / float(close.iloc[-252]) - 1) if len(close) >= 252 else 0
        vol_20d = float(close.pct_change().tail(20).std() * np.sqrt(252)) if len(close) >= 20 else 0.3

        # ── 1. UNDISCOVERY score (0–35) ─────────────────────────────────
        undis = 0
        # Few analysts = undiscovered
        if   n_ana == 0:  undis += 20
        elif n_ana <= 3:  undis += 15
        elif n_ana <= 6:  undis += 8
        elif n_ana <= 10: undis += 3
        # Small market cap = institutional money hasn't found it
        if   mktcap < 200e6:   undis += 10
        elif mktcap < 500e6:   undis += 8
        elif mktcap < 1e9:     undis += 5
        elif mktcap < 2e9:     undis += 2
        elif mktcap > 10e9:    undis -= 5   # too large = already found
        undis = max(0, min(35, undis))

        # ── 2. FUNDAMENTAL quality (0–30) ───────────────────────────────
        fund = 0
        # Revenue growth — core signal
        if   rev_g > 0.50:  fund += 15
        elif rev_g > 0.30:  fund += 12
        elif rev_g > 0.15:  fund += 8
        elif rev_g > 0.05:  fund += 4
        elif rev_g < -0.10: fund -= 5
        # Gross margin
        if   gross_m > 0.60: fund += 8
        elif gross_m > 0.40: fund += 5
        elif gross_m > 0.20: fund += 2
        elif gross_m < 0:    fund -= 8
        # FCF / cash strength
        if fcf_yield and fcf_yield > 0.03: fund += 5
        elif fcf_yield and fcf_yield > 0:  fund += 2
        elif curr_r > 2:                   fund += 2   # healthy current ratio
        # Debt — early companies should have low debt
        if   de < 20:   fund += 2
        elif de > 100:  fund -= 3
        fund = max(0, min(30, fund))

        # ── 3. EARLY TIMING score (0–25) ────────────────────────────────
        # We WANT the stock NOT to have already run
        timing = 0
        # Stock flat or slightly down over 6M = still early
        if   -0.10 <= mom_6m <= 0.10:  timing += 15   # sideways = perfect
        elif  0.10 <  mom_6m <= 0.25:  timing += 8    # modest gains = maybe still early
        elif  mom_6m > 0.40:           timing += 0    # already running
        elif  mom_6m < -0.20:          timing += 5    # beaten down = high risk but opportunity
        # 1Y flat = really undiscovered
        if -0.15 <= mom_1y <= 0.20:    timing += 8
        # Moderate volatility (not dead, not hyper volatile)
        if 0.20 <= vol_20d <= 0.60:    timing += 2
        timing = max(0, min(25, timing))

        # ── 4. MANAGEMENT alignment (0–10) ──────────────────────────────
        mgmt = 0
        if insider > 0.20:  mgmt += 10   # >20% insider = skin in the game
        elif insider > 0.10: mgmt += 7
        elif insider > 0.05: mgmt += 4
        elif insider > 0.02: mgmt += 2
        mgmt = max(0, min(10, mgmt))

        total = undis + fund + timing + mgmt

        # Minimum bar: must have at least some growth
        if rev_g < -0.15 and gross_m < 0:
            total = max(0, total - 20)  # penalise deteriorating businesses

        return {
            "score":     round(float(total), 1),
            "undis":     undis,
            "fund":      fund,
            "timing":    timing,
            "mgmt":      mgmt,
            "price":     round(price, 2),
            "mktcap":    mktcap,
            "n_ana":     n_ana,
            "rev_g":     rev_g,
            "gross_m":   gross_m,
            "insider":   insider,
            "mom_6m":    mom_6m,
            "mom_1y":    mom_1y,
            "vol_20d":   vol_20d,
            "ps":        ps,
            "pe":        pe,
            "fcf_yield": fcf_yield,
            "curr_r":    curr_r,
            "de":        de,
            "short_p":   short_p,
            "name":      info.get("shortName") or info.get("longName") or sym,
            "sector":    info.get("sector") or "Unknown",
            "website":   info.get("website") or "",
            "summary":   (info.get("longBusinessSummary") or "")[:400],
        }
    except Exception:
        return None


def compute_10x_potential(gem):
    """
    What needs to be true for this stock to 10x?
    Returns plain-English analysis.
    """
    mktcap = gem.get("mktcap", 0) or 0
    rev_g  = gem.get("rev_g", 0) or 0
    ps     = gem.get("ps", 0) or 0
    gross_m= gem.get("gross_m", 0) or 0
    n_ana  = gem.get("n_ana", 0)

    lines  = []

    # Market cap headroom
    if mktcap > 0:
        target_mcap = mktcap * 10
        lines.append(
            f"**Market cap needed for 10x:** ${target_mcap/1e9:.1f}B "
            f"(from current ${mktcap/1e9:.2f}B). "
            f"{'This is realistic for a mid-cap leader in an emerging space.' if target_mcap < 20e9 else 'Would require becoming a major sector leader.'}"
        )

    # Revenue growth needed
    if rev_g > 0 and ps > 0:
        # At current P/S, revenue needs to grow X× to justify 10× market cap
        rev_needed = 10 / max(ps / ps, 1)   # simplified
        lines.append(
            f"**Revenue path:** Currently growing {rev_g*100:.0f}% YoY. "
            f"At current P/S ratio, needs sustained >30% growth for 3+ years to justify 10× valuation."
        )

    # Analyst re-rating
    if n_ana <= 3:
        lines.append(
            f"**Analyst re-rating:** Only {n_ana} analyst(s) covering this stock. "
            f"When a major bank initiates coverage, institutional money flows in. "
            f"This alone can cause a 30-50% re-rating."
        )

    # Gross margin expansion
    if gross_m > 0.40:
        lines.append(
            f"**Margin leverage:** {gross_m*100:.0f}% gross margin with operating leverage — "
            f"as revenue scales, net margins expand significantly."
        )

    lines.append(
        "**Key catalyst to watch:** Major contract/partnership announcement, "
        "analyst initiation of coverage, or a major player (MSFT/Google/Amazon) investing or partnering."
    )

    return lines


@st.cache_data(ttl=3600, show_spinner=False)
def discover_industry_stocks_ai(industry_name, industry_cfg, market_hint, gemini_key):
    """Use Gemini to discover lesser-known companies in the industry."""
    if not gemini_key:
        return []
    try:
        from google import genai as genai_sdk
        client = genai_sdk.Client(api_key=gemini_key)

        seeds = industry_cfg.get("seeds_us", []) + industry_cfg.get("seeds_in", [])
        seed_str = ", ".join(seeds[:6]) if seeds else "none"

        prompt = f"""You are a research analyst specialising in undiscovered small-cap stocks.

Industry: {industry_name}
Known seed stocks: {seed_str}

Find 15 LESSER-KNOWN, UNDER-COVERED companies (not the obvious large-caps) that:
1. Are directly involved in {industry_name}
2. Have market cap BELOW $3 billion
3. Are publicly listed (provide valid stock ticker symbols)
4. Have real revenue (not just ideas/patents)
5. Are NOT well-known to most retail investors

Market preference: {"India (NSE/BSE, use .NS suffix)" if market_hint == "india"
                    else "US (NYSE/NASDAQ)" if market_hint == "us" else "Both US and India"}

Return ONLY a JSON array of objects with fields: symbol, name, reason
Example: [{{"symbol": "ARQQ", "name": "Arqit Quantum", "reason": "Quantum encryption SaaS, <5 analysts"}}]
Return ONLY the JSON array, no other text."""

        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = resp.text.strip()
        # Extract JSON
        if "[" in text:
            text = text[text.index("["):text.rindex("]")+1]
        items = json.loads(text)
        return [{"symbol": i.get("symbol","").strip().upper(),
                 "name":   i.get("name",""),
                 "reason": i.get("reason","")}
                for i in items if i.get("symbol")]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Early Stage Discovery
# ══════════════════════════════════════════════════════════════════════════════

with main_tab4:
    for _tab6_once in [True]:
        st.markdown("## 🎯 Live Trade Signals — Entry · Exit · Candlestick Monitor")
        st.caption(
            "Upload your portfolio PDF or auto-load from portfolio/watchlist. "
            "Get exact entry & exit prices. Monitor live candlestick charts with real-time exit signals."
        )

        lt2, lt1 = st.tabs(["📊 Live Candlestick Monitor", "📄 Entry / Exit Strategy"])

        # ── Shared: collect tracked tickers ───────────────────────────────────────
        _lt_pf = []
        if PORTFOLIO_FILE.exists():
            try: _lt_pf = json.loads(PORTFOLIO_FILE.read_text())
            except Exception: pass
        _lt_wl = load_watchlist()
        _lt_default_syms = list(dict.fromkeys(
            [h["symbol"] for h in _lt_pf] + [w["symbol"] for w in _lt_wl]
        ))

        # ── SUB-TAB 1: Entry / Exit Strategy ──────────────────────────────────────
        with lt1:
            st.markdown("### 📄 Select or Upload Your Portfolio")

            _lt_upload_col, _lt_sel_col = st.columns([1, 1])
            with _lt_upload_col:
                _lt_pdf = st.file_uploader(
                    "Upload portfolio PDF or screenshot (optional)",
                    type=["pdf", "png", "jpg", "jpeg"], key="lt_pdf_upload",
                    help="Portfolio statement, broker report, watchlist screenshot — PDF or PNG/JPG accepted.",
                )
            with _lt_sel_col:
                st.caption("Or manually add tickers:")
                _lt_extra = st.text_input(
                    "Additional tickers (comma separated)",
                    placeholder="AAPL, NVDA, RELIANCE.NS",
                    key="lt_extra_tickers",
                )

            # Build ticker universe
            _lt_syms = list(_lt_default_syms)

            # Reset multiselect when a different file is uploaded
            _lt_cur_fname = _lt_pdf.name if _lt_pdf else ""
            if st.session_state.get("_lt_last_file") != _lt_cur_fname:
                st.session_state["_lt_last_file"] = _lt_cur_fname
                st.session_state.pop("lt_selected_syms", None)

            if _lt_pdf:
                _lt_ftype = _lt_pdf.name.lower().rsplit(".", 1)[-1]
                if _lt_ftype in ("png", "jpg", "jpeg"):
                    _mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
                    with st.spinner("Reading image with Gemini vision…"):
                        _pdf_tickers, _img_err = _parse_image_for_tickers(_lt_pdf.read(), _mime_map.get(_lt_ftype, "image/png"))
                    _src_label = "image"
                    if _img_err:
                        st.warning(f"Image parsing: {_img_err}")
                else:
                    with st.spinner("Reading PDF…"):
                        _pdf_tickers = _parse_pdf_for_tickers(_lt_pdf.read())
                    _img_err = ""
                    _src_label = "PDF"
                if _pdf_tickers:
                    st.success(f"Found **{len(_pdf_tickers)}** tickers in {_src_label}: `{', '.join(_pdf_tickers[:15])}`")
                    _lt_syms = list(dict.fromkeys(_pdf_tickers + _lt_syms))
                elif not _img_err:
                    st.warning(f"Couldn't extract tickers from {_src_label} automatically. Add them manually below.")
            if _lt_extra:
                for _t in _lt_extra.split(","):
                    _t = _t.strip().upper()
                    if _t and _t not in _lt_syms:
                        _lt_syms.append(_t)

            if not _lt_syms:
                st.info("No stocks found. Upload a PDF/screenshot or add tickers manually above.")
            else:
                # Ensure multiselect is pre-populated when session state is empty/stale
                _lt_sel_key = "lt_selected_syms"
                _cur_sel = st.session_state.get(_lt_sel_key, [])
                # Re-seed if the stored selection has no overlap with the current symbol list
                if not _cur_sel or not any(s in _lt_syms for s in _cur_sel):
                    st.session_state[_lt_sel_key] = _lt_syms[:min(8, len(_lt_syms))]

                _lt_selected = st.multiselect(
                    "Stocks to build strategy for:",
                    options=_lt_syms,
                    key=_lt_sel_key,
                )

                _lt_risk = st.radio(
                    "Risk appetite:",
                    ["Conservative (1-2% stop)", "Moderate (3-5% stop)", "Aggressive (7-10% stop)"],
                    index=1, horizontal=True, key="lt_risk",
                )
                _lt_stop_pct = {"Conservative (1-2% stop)": 0.015,
                                "Moderate (3-5% stop)": 0.04,
                                "Aggressive (7-10% stop)": 0.085}.get(_lt_risk, 0.04)

                if st.button("🔍 Generate Entry / Exit Strategy", key="lt_run_strat", type="primary",
                             use_container_width=True):
                    if not _lt_selected:
                        st.warning("Select at least one stock from the list above before generating.")
                        break
                    st.session_state["lt_strategies"] = {}
                    _prog = st.progress(0, text="Building strategies…")
                    for _i, _sym in enumerate(_lt_selected):
                        _prog.progress((_i + 1) / max(len(_lt_selected), 1),
                                       text=f"Analysing {_sym}…")
                        try:
                            _f = fetch_stock_features(_sym)
                            if not _f:
                                continue
                            enrich_with_fundamentals(_f)
                            _mc = monte_carlo_targets(_f)
                            _px = float(_f.get("price") or 0)
                            if _px <= 0:
                                continue
                            # Entry: current price or slight pullback
                            _entry = round(_px * 0.995, 2)   # 0.5% below current
                            # Stop loss based on risk setting
                            _stop  = round(_entry * (1 - _lt_stop_pct), 2)
                            # Targets from MC
                            def _mcp(h, k, fb=0):
                                return _mc[h].get(k, fb) if _mc and h in _mc else fb
                            _t1m  = _mcp("1M", "price_base", round(_px * 1.05, 2))
                            _t3m  = _mcp("3M", "price_base", round(_px * 1.12, 2))
                            _t1y  = _mcp("1Y", "price_base", round(_px * 1.25, 2))
                            _r1m  = _mcp("1M", "ret_base", 0)
                            _r3m  = _mcp("3M", "ret_base", 0)
                            _prob = _mcp("3M", "prob_positive", 50)
                            # R/R ratio (using 3M target)
                            _rr   = round((_t3m - _entry) / (_entry - _stop), 2) if _entry > _stop else 0
                            # Recommended position size (Kelly-lite)
                            _kelly = max(0, min(25, round(_prob - 50, 0)))  # % of portfolio
                            # Avg cost from portfolio
                            _avg_cost = next((h.get("avg_price", 0) for h in _lt_pf if h["symbol"] == _sym), 0)
                            # Current P&L if in portfolio
                            _pnl_pct = round((_px - _avg_cost) / _avg_cost * 100, 1) if _avg_cost > 0 else None

                            st.session_state["lt_strategies"][_sym] = {
                                "price": _px, "entry": _entry, "stop": _stop,
                                "t1m": _t1m, "t3m": _t3m, "t1y": _t1y,
                                "r1m": _r1m, "r3m": _r3m, "prob": _prob,
                                "rr": _rr, "kelly": _kelly,
                                "avg_cost": _avg_cost, "pnl_pct": _pnl_pct,
                                "name": _f.get("name", _sym)[:30],
                                "sector": _f.get("sector", "—"),
                            }
                        except Exception:
                            pass
                    _prog.empty()
                    st.rerun()

                # ── Display strategies ─────────────────────────────────────────────
                if "lt_strategies" in st.session_state and st.session_state["lt_strategies"]:
                    _strats = st.session_state["lt_strategies"]
                    st.markdown("---")
                    st.markdown(f"### 📋 Strategy for {len(_strats)} Stocks")
                    st.caption("Entry/Exit/Stop calculated from Monte Carlo (10,000 paths). Size = Kelly-based % of portfolio.")

                    for _sym, _s in _strats.items():
                        _rr_ok = _s["rr"] >= 2
                        _prob_ok = _s["prob"] >= 55
                        _overall = "🟢 TRADE" if (_rr_ok and _prob_ok) else "🟡 WATCH" if (_rr_ok or _prob_ok) else "🔴 SKIP"

                        with st.container(border=True):
                            _lh1, _lh2 = st.columns([3, 1])
                            with _lh1:
                                st.markdown(f"**{_sym}** &nbsp; {_s['name']} &nbsp; `{_s['sector']}`")
                                if _s["pnl_pct"] is not None:
                                    _pl_icon = "📈" if _s["pnl_pct"] >= 0 else "📉"
                                    st.caption(f"Avg cost: {curr}{_s['avg_cost']:.2f}  ·  Current P&L: {_pl_icon} {_s['pnl_pct']:+.1f}%")
                            with _lh2:
                                st.metric("Signal", _overall)

                            _c1, _c2, _c3, _c4, _c5, _c6 = st.columns(6)
                            _c1.metric("Live Price",   f"{curr}{_s['price']:.2f}")
                            _c2.metric("🟢 Enter At",  f"{curr}{_s['entry']:.2f}",
                                       delta="-0.5% from live", delta_color="off")
                            _c3.metric("🔴 Stop Loss", f"{curr}{_s['stop']:.2f}",
                                       delta=f"-{_lt_stop_pct*100:.0f}%", delta_color="inverse")
                            _c4.metric("1M Target",    f"{curr}{_s['t1m']:.2f}",
                                       delta=f"{_s['r1m']:+.1f}%",
                                       delta_color="normal" if _s["r1m"] > 0 else "inverse")
                            _c5.metric("3M Target",    f"{curr}{_s['t3m']:.2f}",
                                       delta=f"{_s['r3m']:+.1f}%",
                                       delta_color="normal" if _s["r3m"] > 0 else "inverse")
                            _c6.metric("R/R Ratio",    f"{_s['rr']:.1f}:1",
                                       delta="✅ Good" if _rr_ok else "⚠️ Poor",
                                       delta_color="normal" if _rr_ok else "inverse")

                            _info_cols = st.columns([2, 2, 3])
                            with _info_cols[0]:
                                st.metric("Prob Up (3M)", f"{_s['prob']:.0f}%")
                            with _info_cols[1]:
                                st.metric("Position Size", f"{_s['kelly']:.0f}% of portfolio",
                                          delta_color="off")
                            with _info_cols[2]:
                                if _rr_ok and _prob_ok:
                                    st.success(f"Enter near {curr}{_s['entry']:.2f} · Target {curr}{_s['t3m']:.2f} · Stop {curr}{_s['stop']:.2f}")
                                elif not _rr_ok:
                                    st.warning(f"R/R {_s['rr']:.1f}:1 below 2:1 minimum — wait for better entry")
                                else:
                                    st.info(f"Probability {_s['prob']:.0f}% — monitor for clearer setup")

        # ── SUB-TAB 2: Live Candlestick Monitor ───────────────────────────────────
        with lt2:
            st.markdown("### 📊 Live Candlestick Monitor")

            # Portfolio holdings set (locked — cannot be removed from watchlist UI)
            try:
                _pf_locked = set(h["symbol"] for h in json.loads(PORTFOLIO_FILE.read_text()))
            except Exception:
                _pf_locked = set()

            # ── Controls ────────────────────────────────────────────────────────
            _STOCK_SUGGESTIONS = [
                "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","INTC","NFLX",
                "ORCL","CRM","ADBE","QCOM","AVGO","TSM","ASML","NOW","SNOW","PLTR",
                "COIN","MSTR","MELI","SHOP","SQ","PYPL","UBER","LYFT","ABNB","DASH",
                "RIVN","LCID","F","GM","TM","BABA","JD","PDD","NIO","XPEV","LI",
                "JPM","BAC","GS","MS","WFC","V","MA","AXP","BRK-B","BLK",
                "UNH","JNJ","PFE","MRNA","LLY","ABBV","BMY","GILD","CVS","CI",
                "XOM","CVX","COP","SLB","BP","SHEL","MPC","VLO","OXY","HAL",
                "SPY","QQQ","IWM","GLD","SLV","TLT","HYG","VTI","VOO","ARKK",
            ]

            _lmc1, _lmc2, _lmc3, _lmc4 = st.columns([3, 1.2, 1, 1])
            with _lmc1:
                _lm_sym_raw = st.text_input(
                    "Search any stock ticker",
                    value=st.session_state.get("lm_sym_val", _lt_default_syms[0] if _lt_default_syms else "NVDA"),
                    placeholder="Type NVDA, AAPL, TSLA, RELIANCE.NS …",
                    key="lm_sym_input",
                )
                _lm_sym = _lm_sym_raw.strip().upper() or "NVDA"
                st.session_state["lm_sym_val"] = _lm_sym
                # Suggestions
                if _lm_sym_raw and len(_lm_sym_raw) >= 1:
                    _all_suggestions = list(dict.fromkeys(_lt_default_syms + _STOCK_SUGGESTIONS))
                    _matches = [s for s in _all_suggestions if s.upper().startswith(_lm_sym.upper()) and s != _lm_sym][:8]
                    if _matches:
                        _sug_cols = st.columns(len(_matches))
                        for _si, _sm in enumerate(_matches):
                            if _sug_cols[_si].button(_sm, key=f"sug_{_sm}", use_container_width=True):
                                st.session_state["lm_sym_val"] = _sm
                                st.session_state["lm_sym_input"] = _sm
                                st.rerun()
            with _lmc2:
                _lm_interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","1h"], index=2, key="lm_interval")
            with _lmc3:
                _lm_period = {"1m":"1d","2m":"2d","5m":"5d","15m":"5d","30m":"1mo","1h":"3mo"}.get(_lm_interval,"5d")
                st.metric("Period", _lm_period)
            with _lmc4:
                if st.button("🔄 Refresh", key="lm_manual_refresh", use_container_width=True, type="primary"):
                    _fetch_live_candles.clear()
                    _fetch_finnhub_quote.clear()
                    st.rerun()
                if st.button("➕ Watchlist", key="lm_quick_add", use_container_width=True):
                    _wl = load_watchlist()
                    if not any(w["symbol"] == _lm_sym for w in _wl) and _lm_sym not in _pf_locked:
                        _wl.append({"symbol": _lm_sym, "added": str(datetime.date.today())})
                        save_watchlist(_wl)
                        st.success(f"Added {_lm_sym} to watchlist.")
                        st.rerun()
                    else:
                        st.info(f"{_lm_sym} already in your list.")

            # ── Fetch: yfinance historical bars + Finnhub live candle ────────────
            # Finnhub free tier: /quote works (real-time), /stock/candle is paywalled.
            # We build a synthetic live candle from the Finnhub quote and append it to
            # the yfinance DataFrame so signals + chart always reflect the true current price.
            # Tell ws_feed to subscribe to this symbol
            _ws_subscribe(_lm_sym)

            # Check if WebSocket feed is running and has fresh data
            _ws_live = _ws_read(_lm_sym, max_age_s=5)
            _ws_running = _ws_is_running()

            if _ws_running:
                st.caption("🟢 **WebSocket LIVE** — data streaming from Finnhub in real-time")
            else:
                st.caption("🟡 WebSocket feed not running — using yfinance (15s cache). "
                           "Run `python ws_feed.py` in a terminal for live data.")

            # Get Finnhub quote for regular session price
            _fh_quote = _fetch_finnhub_quote(_lm_sym)
            with st.spinner(f"Loading {_lm_sym}…"):
                _lm_df = _fetch_live_candles(_lm_sym, _lm_interval, _lm_period)
                if _lm_df is None:
                    st.error(f"Could not fetch data for {_lm_sym}.")

                import pandas as _pd2
                _live_candle_added = False

                # ── Option A: WebSocket live candles (best — ~1s latency) ─────────
                if _ws_live and _lm_df is not None and not _lm_df.empty:
                    _interval_min = {"1m":1,"2m":2,"5m":5,"15m":15,"30m":30,"1h":60}.get(_lm_interval, 5)
                    _ws_candle_key = str(_interval_min)
                    _ws_clist = _ws_live.get("candles", {}).get(_ws_candle_key, [])
                    if _ws_clist:
                        _ws_df = _ws_candles_to_df(_ws_clist)
                        if _ws_df is not None and not _ws_df.empty:
                            # Merge: keep yfinance history + replace recent bars with WebSocket
                            _cutoff = _ws_df.index[0]
                            _lm_df_hist = _lm_df[_lm_df.index < _cutoff]
                            if _lm_df_hist.empty:
                                _lm_df = _ws_df
                            else:
                                _lm_df = _pd2.concat([_lm_df_hist, _ws_df])
                            _live_candle_added = True

                # ── Fallback: synthetic candle from Finnhub quote (30s latency) ───
                if not _live_candle_added and _fh_quote.get("c") and _lm_df is not None and not _lm_df.empty:
                    _fh_c  = float(_fh_quote["c"])
                    _fh_h  = float(_fh_quote.get("h", _fh_c))
                    _fh_l  = float(_fh_quote.get("l", _fh_c))
                    _last_close = float(_lm_df["Close"].iloc[-1])
                    _live_ts = _pd2.Timestamp.utcnow().tz_localize(None) if _lm_df.index.tz is None \
                               else _pd2.Timestamp.utcnow()
                    _live_row = _pd2.DataFrame({
                        "Open":   [_last_close],
                        "High":   [max(_last_close, _fh_c, _fh_h)],
                        "Low":    [min(_last_close, _fh_c, _fh_l)],
                        "Close":  [_fh_c],
                        "Volume": [0],
                    }, index=[_live_ts])
                    _lm_df = _pd2.concat([_lm_df, _live_row])
                    _live_candle_added = True

            _ws_price = _ws_live.get("price") if _ws_live else None
            _lm_data_src = (
                "🟢 WebSocket LIVE (Finnhub ticks → real-time candles)" if _ws_live
                else "🟡 Finnhub quote + yfinance bars (15s cache)" if _live_candle_added
                else "⚪ yfinance only"
            )

            if _lm_df is not None and not _lm_df.empty:
                _sig  = _compute_live_signals(_lm_df)
                _hfr  = _hedge_fund_rules(_lm_df, _fh_quote)  # rule-based engine
                _res, _sup = _compute_sr_levels(_lm_df)
                _pats = _detect_candle_patterns(_lm_df)
                _tgt  = _compute_targets(_lm_df, _res, _sup)

                # Current price: during extended hours use yfinance prepost last bar
                # (Finnhub free /quote only returns regular session close, not pre/post price)
                import pandas as _pdx
                try:
                    _now_et2 = _pdx.Timestamp.now(tz="US/Eastern")
                    _mkt_open  = _now_et2.hour > 9 or (_now_et2.hour == 9 and _now_et2.minute >= 30)
                    _mkt_close = _now_et2.hour >= 16
                    _in_regular_session = _mkt_open and not _mkt_close
                except Exception:
                    _in_regular_session = True

                if _ws_price:
                    _cur_px = float(_ws_price)         # WebSocket tick — most accurate
                elif _in_regular_session and _fh_quote.get("c"):
                    _cur_px = float(_fh_quote["c"])    # Finnhub regular session
                else:
                    _cur_px = float(_lm_df["Close"].dropna().iloc[-1])  # yfinance prepost
                _atr    = _tgt.get("atr", 0)
                _fh_chg = float(_fh_quote.get("dp", 0)) if _fh_quote.get("dp") else None  # % change

                # Data source badge
                st.caption(f"**{_lm_data_src}**"
                           + (f"  ·  Day change: **{_fh_chg:+.2f}%**" if _fh_chg else "")
                           + ("  ·  Rightmost candle = live price" if _live_candle_added else ""))

                # ── Signal banner ────────────────────────────────────────────────
                _sig_map = {
                    "EXIT_NOW":     ("🔴 EXIT NOW — Overbought / Momentum collapsing", "error"),
                    "WATCH":        ("🟡 WATCH — Momentum fading, tighten stop",       "warning"),
                    "HOLD":         ("🟢 HOLD — Trend intact, ride the move",          "success"),
                    "HOLD_BUY_DIP": ("🔵 HOLD / ADD — Pullback into support",          "info"),
                }
                _sig_label, _sig_type = _sig_map.get(_sig["signal"], ("⚪ No clear signal", "info"))
                getattr(st, _sig_type)(
                    f"**{_sig_label}**"
                    + (f"  \nTriggers: {', '.join(_sig['exit_reasons'])}" if _sig.get("exit_reasons") else "")
                )

                # ── Hedge Fund Rule Engine ────────────────────────────────────────
                st.markdown("---")
                _hf_verdict = _hfr.get("verdict","HOLD")
                _hf_score   = _hfr.get("score", 0)
                _hf_max     = _hfr.get("max_score", 1)
                _hf_col = ("#00c853" if "BUY" in _hf_verdict and "SELL" not in _hf_verdict
                           else "#ff5252" if "SELL" in _hf_verdict
                           else "#ff9800")
                _hfca, _hfcb = st.columns([3,1])
                with _hfca:
                    st.markdown(f"#### 📐 Quant Rule Engine  ·  "
                                f"<span style='color:{_hf_col};font-weight:700'>{_hf_verdict}</span>  "
                                f"<span style='color:#888;font-size:0.85rem'>Score {_hf_score}/{_hf_max}</span>",
                                unsafe_allow_html=True)
                with _hfcb:
                    _hf_bar = int(abs(_hf_score) / max(abs(_hf_max),1) * 100)
                    st.progress(_hf_bar if _hf_score >= 0 else 0,
                                text=f"{'Bull' if _hf_score>=0 else 'Bear'} {_hf_bar}%")

                # Rule breakdown table
                with st.expander("📋 Rule breakdown — click to see all signals", expanded=True):
                    _rule_cols = st.columns([3, 1, 2, 5])
                    _rule_cols[0].markdown("**Rule**")
                    _rule_cols[1].markdown("**Score**")
                    _rule_cols[2].markdown("**Category**")
                    _rule_cols[3].markdown("**Reason**")
                    for _rl in _hfr.get("rules", []):
                        _rco = "#00c853" if _rl["score"]>0 else "#ff5252" if _rl["score"]<0 else "#888"
                        _rci = "🟢" if _rl["score"]>0 else "🔴" if _rl["score"]<0 else "⚪"
                        _rc1, _rc2, _rc3, _rc4 = st.columns([3, 1, 2, 5])
                        _rc1.markdown(f"{_rci} **{_rl['name']}**")
                        _rc2.markdown(f"<span style='color:{_rco};font-weight:bold'>{_rl['score']:+d}</span>",
                                      unsafe_allow_html=True)
                        _rc3.markdown(f"`{_rl['category']}`")
                        _rc4.caption(_rl["reason"])

                # ── Pattern detected ─────────────────────────────────────────────
                st.markdown("---")
                if _pats:
                    _pn, _pe, _pd = _pats[0]
                    st.caption(f"Last candle pattern: **{_pe} {_pn}** — {_pd}")

                # ── Key metrics ──────────────────────────────────────────────────
                _m1,_m2,_m3,_m4,_m5,_m6 = st.columns(6)
                _m1.metric("Price",          f"{curr}{_cur_px:.2f}")
                _m2.metric("RSI (14)",       f"{_sig.get('rsi',0):.1f}" if _sig.get("rsi") else "—",
                           delta="Overbought" if (_sig.get("rsi") or 0)>70 else "Oversold" if (_sig.get("rsi") or 0)<30 else "Neutral",
                           delta_color="inverse" if (_sig.get("rsi") or 0)>70 else "normal" if (_sig.get("rsi") or 0)<30 else "off")
                _m3.metric("MACD Hist",      f"{_sig.get('macd_hist',0):.3f}" if _sig.get("macd_hist") is not None else "—",
                           delta="Bullish" if (_sig.get("macd_hist") or 0)>0 else "Bearish",
                           delta_color="normal" if (_sig.get("macd_hist") or 0)>0 else "inverse")
                _m4.metric("ATR (14)",       f"{curr}{_atr:.2f}")
                _m5.metric("Session Stop",   f"{curr}{_tgt.get('session_stop',0):.2f}",
                           delta=f"{(_tgt.get('session_stop',_cur_px)-_cur_px)/_cur_px*100:+.1f}%", delta_color="inverse")
                _m6.metric("Vol Spike",      f"{_sig.get('vol_z',0):.1f}σ",
                           delta="High" if (_sig.get("vol_z") or 0)>2 else "Normal",
                           delta_color="inverse" if (_sig.get("vol_z") or 0)>2 else "off")

                # ── S/R + Targets panel ──────────────────────────────────────────
                _ta_col, _tb_col = st.columns(2)
                with _ta_col:
                    with st.container(border=True):
                        st.markdown("**🔴 Resistance Levels**")
                        for _i, _rv in enumerate(_res[:3]):
                            _pct = (_rv - _cur_px) / _cur_px * 100
                            st.metric(f"R{_i+1}", f"{curr}{_rv:.2f}", delta=f"+{_pct:.1f}% away", delta_color="off")
                        if not _res:
                            st.caption("No clear resistance found in window")
                with _tb_col:
                    with st.container(border=True):
                        st.markdown("**🟢 Support Levels**")
                        for _i, _sv in enumerate(_sup[:3]):
                            _pct = (_sv - _cur_px) / _cur_px * 100
                            st.metric(f"S{_i+1}", f"{curr}{_sv:.2f}", delta=f"{_pct:.1f}% away", delta_color="off")
                        if not _sup:
                            st.caption("No clear support found in window")

                # Targets row
                _tc1,_tc2,_tc3,_tc4 = st.columns(4)
                _tc1.metric("Next Resistance", f"{curr}{_tgt.get('next_resistance',0):.2f}",
                            delta=f"+{(_tgt.get('next_resistance',_cur_px)-_cur_px)/_cur_px*100:.1f}%", delta_color="off")
                _tc2.metric("Swing Stop",      f"{curr}{_tgt.get('swing_stop',0):.2f}",
                            delta=f"{(_tgt.get('swing_stop',_cur_px)-_cur_px)/_cur_px*100:+.1f}%", delta_color="inverse")
                _tc3.metric("Fib 100% Target", f"{curr}{_tgt.get('fib_100',0):.2f}",
                            delta=f"+{(_tgt.get('fib_100',_cur_px)-_cur_px)/_cur_px*100:.1f}%", delta_color="normal")
                _tc4.metric("Fib 1.618 Target",f"{curr}{_tgt.get('fib_618',0):.2f}",
                            delta=f"+{(_tgt.get('fib_618',_cur_px)-_cur_px)/_cur_px*100:.1f}%", delta_color="normal")

                # ── TradingView Professional Chart ───────────────────────────────
                import plotly.graph_objects as _go
                from plotly.subplots import make_subplots as _msp

                # Map ticker to TradingView symbol format
                def _tv_symbol(sym):
                    if sym.endswith(".NS"):
                        return f"NSE:{sym[:-3]}"
                    if sym.endswith(".BO"):
                        return f"BSE:{sym[:-3]}"
                    # Common exchange mapping for US stocks
                    _nasdaq = {"NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","INTC",
                               "NFLX","ORCL","ADBE","QCOM","AVGO","CSCO","PYPL","SBUX","COST",
                               "AVGO","MELI","SHOP","COIN","PLTR","SNOW","NOW","CRM","ADSK","PANW","CRWD"}
                    return f"NASDAQ:{sym}" if sym in _nasdaq else f"NYSE:{sym}"

                _tv_sym  = _tv_symbol(_lm_sym)
                _tv_intv = {"1m":"1","2m":"2","5m":"5","15m":"15","30m":"30","1h":"60","1d":"D"}.get(_lm_interval, "5")

                _tv_html = f"""
    <div class="tradingview-widget-container" style="height:580px;width:100%">
      <div id="tv_chart_{_lm_sym}" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/v3/bundle.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{_tv_sym}",
        "interval": "{_tv_intv}",
        "timezone": "America/New_York",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#131722",
        "enable_publishing": false,
        "allow_symbol_change": false,
        "hide_side_toolbar": false,
        "withdateranges": true,
        "save_image": true,
        "studies": [
          "RSI@tv-basicstudies",
          "MACD@tv-basicstudies",
          "BB@tv-basicstudies",
          "Volume@tv-basicstudies"
        ],
        "show_popup_button": true,
        "popup_width": "1000",
        "popup_height": "650",
        "container_id": "tv_chart_{_lm_sym}",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "range": "5D",
        "disabled_features": [],
        "enabled_features": ["pre_post_market_sessions","extended_hours"]
      }});
      </script>
    </div>
    """
                st.components.v1.html(_tv_html, height=620, scrolling=False)
                st.caption(f"TradingView chart · {_lm_sym} · Pre/post market enabled · All indicators included")

                # ── Action Summary ────────────────────────────────────────────────
                def _build_action_summary(sig, rsi_val, pats, tgt, cur_px, currency):
                    """Generate 2-line plain-English action summary from computed signals."""
                    signal   = sig.get("signal", "HOLD")
                    macd_h   = sig.get("macd_hist", 0) or 0
                    pat_name = pats[0][0] if pats else "No pattern"
                    pat_bull = any(k in pat_name for k in ("Hammer","Engulfing Bullish","Bullish","Marubozu"))
                    stop     = tgt.get("session_stop", cur_px * 0.98)
                    res      = tgt.get("next_resistance", cur_px * 1.03)
                    sup      = tgt.get("next_support", cur_px * 0.97)
                    fib      = tgt.get("fib_618", cur_px * 1.06)

                    if signal == "EXIT_NOW":
                        reasons = ", ".join(sig.get("exit_reasons", ["momentum collapsed"]))
                        line1 = f"🔴 **SELL / EXIT NOW** — {reasons}."
                        line2 = f"Exit near **{currency}{cur_px:.2f}**. Do not wait — stop at **{currency}{stop:.2f}** already breached."
                    elif signal == "WATCH":
                        reasons = ", ".join(sig.get("exit_reasons", ["momentum fading"]))
                        line1 = f"🟡 **PREPARE TO EXIT** — {reasons}. Pattern: {pat_name}."
                        line2 = f"Tighten stop to **{currency}{stop:.2f}**. Sell if price breaks below **{currency}{sup:.2f}**. Target was **{currency}{res:.2f}**."
                    elif signal == "HOLD_BUY_DIP":
                        line1 = f"🔵 **BUY THE DIP / ADD** — RSI {rsi_val:.0f} oversold, {pat_name} near support. Trend still intact."
                        line2 = f"Entry zone: **{currency}{sup:.2f}–{currency}{cur_px:.2f}**. Stop: **{currency}{stop:.2f}**. Target: **{currency}{res:.2f}** → **{currency}{fib:.2f}**."
                    else:  # HOLD
                        if rsi_val > 60 and macd_h > 0:
                            line1 = f"🟢 **HOLD — Strong trend.** RSI {rsi_val:.0f}, MACD bullish, {pat_name}. No exit signal yet."
                        elif rsi_val < 45:
                            line1 = f"🟢 **HOLD — Weak but recovering.** RSI {rsi_val:.0f}, watch for bounce off **{currency}{sup:.2f}** support."
                        else:
                            line1 = f"🟢 **HOLD — Trend intact.** RSI {rsi_val:.0f} neutral, {pat_name}. Wait for next catalyst."
                        line2 = f"Keep stop at **{currency}{stop:.2f}**. Next resistance: **{currency}{res:.2f}** (+{(res-cur_px)/cur_px*100:.1f}%). Fib target: **{currency}{fib:.2f}**."
                    return line1, line2

                # Compute RSI now (was previously in Chart 3, now computed here)
                _cl2 = _lm_df["Close"].dropna(); _dl2 = _cl2.diff()
                _rsi_ser = 100 - 100/(1 + _dl2.clip(lower=0).rolling(14).mean()/
                                      (-_dl2.clip(upper=0)).rolling(14).mean().replace(0,1e-9))
                _rsi_now = float(_rsi_ser.dropna().iloc[-1]) if not _rsi_ser.dropna().empty else 50

                _sum_line1, _sum_line2 = _build_action_summary(
                    _sig, _rsi_now, _pats, _tgt, _cur_px, curr)

                _sum_col, _ref_col = st.columns([11, 1])
                with _sum_col:
                    with st.container(border=True):
                        st.markdown(_sum_line1)
                        st.markdown(_sum_line2)
                with _ref_col:
                    if st.button("🔄", key="lm_summary_refresh", help="Refresh recommendation"):
                        st.rerun()

                # ── Personalised exit from Strategy tab ──────────────────────────
                if "lt_strategies" in st.session_state and _lm_sym in st.session_state["lt_strategies"]:
                    _s = st.session_state["lt_strategies"][_lm_sym]
                    st.markdown("#### 🚪 Your Planned Exit")
                    _xe1,_xe2,_xe3,_xe4 = st.columns(4)
                    _pnl = (_cur_px - _s["entry"]) / _s["entry"] * 100
                    _xe1.metric("P&L from Entry", f"{_pnl:+.1f}%", delta_color="normal" if _pnl>=0 else "inverse")
                    _xe2.metric("1M Target",  f"{curr}{_s['t1m']:.2f}")
                    _xe3.metric("3M Target",  f"{curr}{_s['t3m']:.2f}")
                    _xe4.metric("Hard Stop",  f"{curr}{_s['stop']:.2f}")

                    if _sig["signal"] == "EXIT_NOW":
                        st.error(f"🔴 **EXIT NOW** — {', '.join(_sig.get('exit_reasons',['Signal triggered']))}. "
                                 f"Sell near {curr}{_cur_px:.2f}.")
                    elif _sig["signal"] == "WATCH":
                        _tight_stop = max(_s["stop"], _cur_px*0.98)
                        st.warning(f"🟡 **Prepare exit.** Tighten stop to {curr}{_tight_stop:.2f}. "
                                   f"Triggers: {', '.join(_sig.get('exit_reasons',[]))}")
                    else:
                        st.success(f"🟢 **Hold.** Next exit at RSI>75 or price > {curr}{_s['t3m']:.2f}. "
                                   f"Session stop: {curr}{_tgt.get('session_stop',0):.2f}")

                st.caption(f"{_lm_data_src} · {_lm_interval} · {len(_lm_df)} bars (incl. pre/post market) · Last bar: {_lm_df.index[-1]}")

            else:
                st.warning(f"No data for **{_lm_sym}**. Check the ticker symbol and try again.")

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 7 — MARKET RISK MONITOR
    # ══════════════════════════════════════════════════════════════════════════════

    @st.cache_data(ttl=1800)
    def _fetch_market_indicators():
        """Fetch key macro/market risk indicators."""
        import yfinance as _yf2
        tickers = {
            "vix":   "^VIX",
            "sp500": "^GSPC",
            "t10y":  "^TNX",   # 10Y yield
            "t2y":   "^TYX",   # 30Y (proxy for long end)
            "t3m":   "^IRX",   # 13-week T-bill
            "gold":  "GLD",
            "dxy":   "DX-Y.NYB",
            "hyg":   "HYG",    # High yield credit
            "tlt":   "TLT",    # Long bonds
            "qqq":   "QQQ",
            "iwm":   "IWM",    # Small caps
        }
        out = {}
        for k, sym in tickers.items():
            try:
                df = _yf2.download(sym, period="1y", progress=False, auto_adjust=True)
                if hasattr(df.columns, "levels"):
                    df.columns = df.columns.get_level_values(0)
                if df is not None and not df.empty:
                    out[k] = df
            except Exception:
                pass
        return out


    @st.cache_data(ttl=1800)
    def _get_market_risk_analysis(indicators_summary: str):
        """Claude analyses macro indicators and returns crash risk assessment."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if groq_key:
                try:
                    from groq import Groq as _GQ
                    _gc = _GQ(api_key=groq_key)
                    _r  = _gc.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content": indicators_summary}],
                        max_tokens=1800, temperature=0.3,
                    )
                    return _r.choices[0].message.content
                except Exception as e:
                    return f"__ERROR__{e}"
            return "__NO_KEY__"
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role":"user","content": indicators_summary}],
            )
            return msg.content[0].text
        except Exception as e:
            return f"__ERROR__{e}"


with main_tab5:
    st.markdown("## 🚨 US Market Risk Monitor")
    st.caption(
        "Real-time macro intelligence: VIX, yield curve, credit spreads, dollar, gold, momentum. "
        "AI synthesises all signals into a crash probability assessment. Refreshes every 30 minutes."
    )

    _mr_refresh = st.button("🔄 Refresh Data", key="mr_refresh")
    if _mr_refresh:
        _fetch_market_indicators.clear()
        _get_market_risk_analysis.clear()
        st.rerun()

    with st.spinner("Fetching market indicators…"):
        _mri = _fetch_market_indicators()

    # ── Extract latest values ─────────────────────────────────────────────────
    def _mlast(key, col="Close"):
        df = _mri.get(key)
        if df is not None and not df.empty and col in df.columns:
            return float(df[col].dropna().iloc[-1])
        return None

    def _mchg(key, days=20, col="Close"):
        df = _mri.get(key)
        if df is not None and len(df) > days and col in df.columns:
            s = df[col].dropna()
            return float((s.iloc[-1] - s.iloc[-days]) / s.iloc[-days] * 100)
        return None

    _vix       = _mlast("vix")
    _sp500     = _mlast("sp500")
    _sp_1m     = _mchg("sp500", 20)
    _sp_3m     = _mchg("sp500", 60)
    _t10y      = _mlast("t10y")
    _t3m       = _mlast("t3m")
    _yc_spread = (_t10y - _t3m / 10) if _t10y and _t3m else None  # 10Y minus 3M (approx)
    _gold_chg  = _mchg("gold", 20)
    _dxy_chg   = _mchg("dxy", 20)
    _hyg_chg   = _mchg("hyg", 20)   # credit spread proxy (negative = widening = bad)
    _tlt_chg   = _mchg("tlt", 20)
    _qqq_chg   = _mchg("qqq", 20)
    _iwm_chg   = _mchg("iwm", 20)   # small caps underperforming = risk-off

    # ── Composite risk score (0-100) ──────────────────────────────────────────
    _risk_signals = []  # (label, value_str, is_risk, pts_earned, pts_max)

    def _add_risk(pts, maxp, label, value_str, is_risk):
        _risk_signals.append((label, value_str, bool(is_risk), pts if is_risk else 0, maxp))

    if _vix:
        _vix_risk = _vix > 30
        _vix_high = _vix > 40
        _add_risk(20 if _vix_high else 12 if _vix_risk else 4, 20,
                  "VIX (Fear Index)", f"{_vix:.1f}",
                  _vix > 25)

    if _yc_spread is not None:
        _yc_inv = _yc_spread < 0
        _add_risk(18 if _yc_inv else 5, 18,
                  "Yield Curve (10Y-3M)", f"{_yc_spread:+.2f}%",
                  _yc_inv)

    if _sp_3m is not None:
        _sp_neg = _sp_3m < -5
        _add_risk(15 if _sp_neg else 5 if _sp_3m < 0 else 0, 15,
                  "S&P 500 (3-month trend)", f"{_sp_3m:+.1f}%",
                  _sp_neg)

    if _hyg_chg is not None:
        _credit_bad = _hyg_chg < -3
        _add_risk(15 if _credit_bad else 5 if _hyg_chg < -1 else 0, 15,
                  "Credit Spreads (HYG)", f"{_hyg_chg:+.1f}% (neg = widening)",
                  _credit_bad)

    if _gold_chg is not None:
        _gold_surge = _gold_chg > 5
        _add_risk(10 if _gold_surge else 3 if _gold_chg > 2 else 0, 10,
                  "Gold (safe haven demand)", f"{_gold_chg:+.1f}%",
                  _gold_surge)

    if _dxy_chg is not None:
        _dxy_surge = _dxy_chg > 3
        _add_risk(8 if _dxy_surge else 2 if _dxy_chg > 1 else 0, 8,
                  "US Dollar (DXY)", f"{_dxy_chg:+.1f}%",
                  _dxy_surge)

    if _iwm_chg is not None and _qqq_chg is not None:
        _breadth_bad = (_iwm_chg - _qqq_chg) < -5
        _add_risk(8 if _breadth_bad else 2, 8,
                  "Market Breadth (IWM vs QQQ)", f"IWM {_iwm_chg:+.1f}% vs QQQ {_qqq_chg:+.1f}%",
                  _breadth_bad)

    if _tlt_chg is not None:
        _bonds_surge = _tlt_chg > 5
        _add_risk(6 if _bonds_surge else 0, 6,
                  "Long Bonds (TLT — flight to safety)", f"{_tlt_chg:+.1f}%",
                  _bonds_surge)

    _risk_pts  = sum(s[3] for s in _risk_signals)
    _risk_max  = sum(s[4] for s in _risk_signals)
    _risk_score = int(_risk_pts / max(_risk_max, 1) * 100) if _risk_max else 0

    # ── Risk level classification ─────────────────────────────────────────────
    if _risk_score >= 70:
        _risk_level = "EXTREME"
        _risk_color = "#ff1744"
        _risk_bg    = "#2a0000"
        _crash_prob_3m = "60-80%"
        _crash_prob_6m = "75-90%"
        _expected_dd   = "20-40%"
        _timeline      = "1-3 months"
    elif _risk_score >= 50:
        _risk_level = "HIGH"
        _risk_color = "#ff5252"
        _risk_bg    = "#200000"
        _crash_prob_3m = "35-55%"
        _crash_prob_6m = "50-70%"
        _expected_dd   = "15-25%"
        _timeline      = "2-6 months"
    elif _risk_score >= 30:
        _risk_level = "MODERATE"
        _risk_color = "#ff9800"
        _risk_bg    = "#1a1000"
        _crash_prob_3m = "15-30%"
        _crash_prob_6m = "25-45%"
        _expected_dd   = "10-20%"
        _timeline      = "3-9 months"
    else:
        _risk_level = "LOW"
        _risk_color = "#00c853"
        _risk_bg    = "#001a00"
        _crash_prob_3m = "5-15%"
        _crash_prob_6m = "10-20%"
        _expected_dd   = "5-15%"
        _timeline      = "6-18 months (if at all)"

    # ── Main risk gauge ───────────────────────────────────────────────────────
    import plotly.graph_objects as _go2

    _gauge_fig = _go2.Figure(_go2.Indicator(
        mode="gauge+number+delta",
        value=_risk_score,
        title={"text": "Market Crash Risk Score", "font": {"size": 18}},
        delta={"reference": 30, "increasing": {"color": "#ff5252"}, "decreasing": {"color": "#00c853"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar": {"color": _risk_color},
            "bgcolor": "#111",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30],  "color": "#001a00"},
                {"range": [30, 50], "color": "#1a1000"},
                {"range": [50, 70], "color": "#200000"},
                {"range": [70, 100],"color": "#2a0000"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 4}, "thickness": 0.8, "value": _risk_score},
        },
    ))
    _gauge_fig.update_layout(
        height=300, template="plotly_dark",
        paper_bgcolor="#0e1117", font={"color": "#fff"},
        margin=dict(l=20, r=20, t=40, b=20),
    )

    _gc1, _gc2 = st.columns([1, 1])
    with _gc1:
        st.plotly_chart(_gauge_fig, use_container_width=True)
    with _gc2:
        st.markdown(f"""
<div style="background:{_risk_bg};border:3px solid {_risk_color};border-radius:12px;
     padding:20px 24px;margin-top:20px">
  <div style="font-size:2.2rem;font-weight:900;color:{_risk_color};letter-spacing:3px">{_risk_level} RISK</div>
  <div style="color:#ccc;margin-top:10px;font-size:0.95rem">
    <b>10%+ correction (3M):</b> {_crash_prob_3m}<br>
    <b>20%+ bear market (6M):</b> {_crash_prob_6m}<br>
    <b>Expected drawdown if triggered:</b> {_expected_dd}<br>
    <b>Risk window:</b> {_timeline}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Key indicator cards ───────────────────────────────────────────────────
    st.markdown("### 📊 Key Risk Indicators")
    _ic_cols = st.columns(4)
    _ind_data = [
        ("VIX", f"{_vix:.1f}" if _vix else "—",
         "🔴 Extreme Fear" if _vix and _vix > 40 else "🟠 High Fear" if _vix and _vix > 30 else "🟡 Elevated" if _vix and _vix > 20 else "🟢 Calm",
         _vix and _vix > 30),
        ("Yield Curve 10Y-3M", f"{_yc_spread:+.2f}%" if _yc_spread else "—",
         "🔴 Inverted (Recession signal)" if _yc_spread and _yc_spread < 0 else "🟢 Normal",
         _yc_spread and _yc_spread < 0),
        ("S&P 500 (3M)", f"{_sp_3m:+.1f}%" if _sp_3m else "—",
         "🔴 Downtrend" if _sp_3m and _sp_3m < -5 else "🟡 Weak" if _sp_3m and _sp_3m < 0 else "🟢 Uptrend",
         _sp_3m and _sp_3m < -5),
        ("Credit Spreads (HYG)", f"{_hyg_chg:+.1f}%" if _hyg_chg else "—",
         "🔴 Widening fast" if _hyg_chg and _hyg_chg < -3 else "🟡 Widening" if _hyg_chg and _hyg_chg < -1 else "🟢 Stable",
         _hyg_chg and _hyg_chg < -3),
        ("Gold (20d)", f"{_gold_chg:+.1f}%" if _gold_chg else "—",
         "🟠 Safe haven rush" if _gold_chg and _gold_chg > 5 else "🟢 Normal",
         _gold_chg and _gold_chg > 5),
        ("US Dollar (20d)", f"{_dxy_chg:+.1f}%" if _dxy_chg else "—",
         "🟠 Surging (risk-off)" if _dxy_chg and _dxy_chg > 3 else "🟢 Stable",
         _dxy_chg and _dxy_chg > 3),
        ("Market Breadth", f"IWM {_iwm_chg:+.1f}% vs QQQ {_qqq_chg:+.1f}%" if _iwm_chg and _qqq_chg else "—",
         "🔴 Narrow rally (warning)" if _iwm_chg and _qqq_chg and (_iwm_chg - _qqq_chg) < -5 else "🟢 Broad",
         _iwm_chg and _qqq_chg and (_iwm_chg - _qqq_chg) < -5),
        ("Long Bonds TLT (20d)", f"{_tlt_chg:+.1f}%" if _tlt_chg else "—",
         "🟠 Flight to safety" if _tlt_chg and _tlt_chg > 5 else "🟢 Normal",
         _tlt_chg and _tlt_chg > 5),
    ]
    for _i, (_iname, _ival, _istat, _ibad) in enumerate(_ind_data):
        with _ic_cols[_i % 4]:
            with st.container(border=True):
                st.markdown(f"**{_iname}**")
                st.markdown(f"### {_ival}")
                st.caption(_istat)

    st.markdown("---")

    # ── Risk signals breakdown ────────────────────────────────────────────────
    st.markdown("### 🔍 Signal Breakdown")
    for _sname, _sval, _sbad, _spts, _smax in _risk_signals:
        _sico = "🔴" if _sbad else "🟢"
        st.markdown(f"{_sico} **{_sname}:** {_sval}  ·  Risk contribution: {_spts}/{_smax}")

    st.markdown("---")

    # ── AI Macro Intelligence ─────────────────────────────────────────────────
    st.markdown("### 🤖 AI Macro Intelligence")
    st.caption("Claude analyses all global factors: inflation, Fed policy, geopolitics, war, earnings cycle, debt levels.")

    _mr_run = st.button("🧠 Run Full Macro Analysis", key="mr_ai_run", type="primary", use_container_width=True)
    if _mr_run or st.session_state.get("mr_ai_done"):
        if _mr_run:
            _get_market_risk_analysis.clear()
            st.session_state["mr_ai_done"] = True

        _macro_prompt = f"""You are the chief macro strategist at a $50B hedge fund. Analyse the current US market risk and provide a comprehensive crash probability assessment.

CURRENT MARKET DATA:
- VIX: {_vix:.1f if _vix else 'N/A'} {'(EXTREME FEAR)' if _vix and _vix>40 else '(HIGH FEAR)' if _vix and _vix>30 else '(ELEVATED)' if _vix and _vix>20 else '(CALM)'}
- Yield Curve (10Y-3M): {f'{_yc_spread:+.2f}%' if _yc_spread else 'N/A'} {'— INVERTED (recession signal)' if _yc_spread and _yc_spread<0 else '— normal'}
- S&P 500 3-month return: {f'{_sp_3m:+.1f}%' if _sp_3m else 'N/A'}
- Credit Spreads (HYG 20d): {f'{_hyg_chg:+.1f}%' if _hyg_chg else 'N/A'}
- Gold (20d): {f'{_gold_chg:+.1f}%' if _gold_chg else 'N/A'}
- US Dollar (20d): {f'{_dxy_chg:+.1f}%' if _dxy_chg else 'N/A'}
- Long Bonds TLT (20d): {f'{_tlt_chg:+.1f}%' if _tlt_chg else 'N/A'}
- Market Breadth: IWM {f'{_iwm_chg:+.1f}%' if _iwm_chg else 'N/A'} vs QQQ {f'{_qqq_chg:+.1f}%' if _qqq_chg else 'N/A'}
- Composite Risk Score: {_risk_score}/100 ({_risk_level})

Provide a comprehensive analysis covering:

**CRASH PROBABILITY ASSESSMENT**
Give specific probabilities for:
- 10%+ S&P correction in next 3 months: X%
- 20%+ bear market in next 6 months: X%
- 30%+ crash in next 12 months: X%
Explain the methodology — what historical analogs support these numbers?

**CURRENT MACRO ENVIRONMENT**
Analyse: Fed policy & interest rates, inflation trajectory, employment, GDP growth, corporate earnings cycle, consumer health, housing market.

**GLOBAL RISK FACTORS**
Geopolitical risks (wars, trade tensions, sanctions), China slowdown, European stress, emerging market contagion, commodity supercycle.

**LEADING INDICATORS TO WATCH**
What 5 specific data points or levels would confirm a crash is imminent? Give exact thresholds (e.g. "VIX above 45", "S&P breaks 200DMA", etc.)

**HISTORICAL ANALOG**
Which past crash does the current setup most resemble — 2000 dot-com, 2008 GFC, 2020 COVID, 1987 Black Monday, or something else? How did it play out?

**MOST LIKELY SCENARIO (next 6 months)**
Not a crash prediction — but the most statistically probable path for the S&P 500.

**APPROXIMATE RISK WINDOW**
Based on historical lead times from current indicator levels, when is the elevated risk period? Give a range like "Q3 2025 - Q1 2026" with confidence level.

**INVESTOR ACTION PLAN**
What should a stock trader do right now? How should they size positions? What hedges make sense?

Be specific with numbers. Use your full knowledge of macro economics, market history, and geopolitics. Today's date context: June 2026."""

        with st.spinner("Claude is analysing global macro factors…"):
            _mr_ai = _get_market_risk_analysis(_macro_prompt)

        if _mr_ai and not _mr_ai.startswith("__ERROR__") and _mr_ai != "__NO_KEY__":
            # Escape dollar signs to prevent LaTeX rendering
            import re as _mre
            _mr_ai_clean = _mre.sub(r'\$(?=[\d\s])', r'\\$', _mr_ai)
            st.markdown(_mr_ai_clean)
        elif _mr_ai == "__NO_KEY__":
            st.warning("No AI key configured. Add `ANTHROPIC_API_KEY` to `.env`.")
        else:
            st.error(f"AI error: {(_mr_ai or '')[9:200]}")

    else:
        st.info("Click **🧠 Run Full Macro Analysis** to get Claude's assessment of global risk factors, crash probability, and what to watch.")

    # ── S&P 500 chart with risk overlay ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 S&P 500 — 1 Year with VIX Overlay")
    _sp_df = _mri.get("sp500")
    _vix_df = _mri.get("vix")
    if _sp_df is not None and not _sp_df.empty and _vix_df is not None and not _vix_df.empty:
        from plotly.subplots import make_subplots as _msp2
        _mfig = _msp2(rows=2, cols=1, shared_xaxes=True,
                      row_heights=[0.65, 0.35], vertical_spacing=0.04,
                      subplot_titles=["S&P 500", "VIX (Fear Index)"])
        _mfig.add_trace(_go2.Scatter(x=_sp_df.index, y=_sp_df["Close"],
                                     name="S&P 500", line=dict(color="#2979ff", width=2),
                                     fill="tozeroy", fillcolor="rgba(41,121,255,0.06)"), row=1, col=1)
        _mfig.add_trace(_go2.Scatter(x=_vix_df.index, y=_vix_df["Close"],
                                     name="VIX", line=dict(color="#ff5252", width=1.5)), row=2, col=1)
        _mfig.add_hline(y=30, line_color="#ff9800", line_dash="dash", line_width=1,
                        annotation_text="VIX 30 (High Fear)", row=2, col=1)
        _mfig.add_hline(y=20, line_color="#ffcc00", line_dash="dot", line_width=1,
                        annotation_text="VIX 20", row=2, col=1)
        _mfig.update_layout(height=500, template="plotly_dark", showlegend=False,
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            margin=dict(l=0, r=0, t=30, b=0))
        _mfig.update_xaxes(showgrid=False)
        _mfig.update_yaxes(showgrid=True, gridcolor="#1a1a1a")
        st.plotly_chart(_mfig, use_container_width=True)
    else:
        st.warning("Could not load chart data.")

    st.caption("⚠️ Disclaimer: Crash probabilities are based on historical indicator correlations, not guaranteed predictions. No model can predict market timing with certainty. Use as one input among many.")
with main_tab6:
    st.markdown("## 📓 Prediction Tracker")
    st.caption(
        "Add stocks to track. Click **Update** daily to generate predictions for today & tomorrow. "
        "The dashboard stores 60 days of data and tracks delta between predicted vs actual prices. "
        "**Train Model** uses all stored data to improve prediction accuracy over time."
    )

    _ptd = _pt_load()
    # Use session state so Add/Remove updates are instant without full re-run
    if "pt_watchlist" not in st.session_state:
        st.session_state["pt_watchlist"] = _ptd.get("watchlist", [])
    _pt_watchlist = st.session_state["pt_watchlist"]
    _ptd["watchlist"] = _pt_watchlist
    _pt_records   = _ptd.get("records", {})
    _pt_weights   = _ptd.get("weights", {})
    _pt_meta      = _ptd.get("model_meta", {})

    # ── Add / Remove stocks ───────────────────────────────────────────────────
    st.markdown("### 📋 Tracked Stocks")

    # Build combined ticker suggestion list (US + India)
    _pt_all_tickers = get_us_universe()
    # Filter out already-tracked ones
    _pt_suggestions = [t for t in _pt_all_tickers if t not in _pt_watchlist]

    _pt_add_col, _pt_btn_col = st.columns([3, 1])
    with _pt_add_col:
        _pt_selected = st.selectbox(
            "Search & add stock",
            options=[""] + _pt_suggestions,
            index=0,
            placeholder="Type to search — e.g. NVDA, RELIANCE.NS, AAPL…",
            key="pt_new_ticker",
            label_visibility="collapsed",
        )
        _pt_new = (_pt_selected or "").strip().upper()
    with _pt_btn_col:
        st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
        if st.button("➕ Add", key="pt_add", use_container_width=True, type="primary"):
            if _pt_new and _pt_new not in _pt_watchlist:
                _pt_watchlist.append(_pt_new)
                st.session_state["pt_watchlist"] = _pt_watchlist
                _ptd["watchlist"] = _pt_watchlist
                _pt_save(_ptd)
                st.success(f"✅ Added {_pt_new} to tracker.")

    # Show tracked stocks as removable chips
    if _pt_watchlist:
        _rm_cols = st.columns(min(len(_pt_watchlist), 8))
        for _pi, _psym in enumerate(_pt_watchlist):
            with _rm_cols[_pi % 8]:
                if st.button(f"✕ {_psym}", key=f"pt_rm_{_psym}", use_container_width=True):
                    _pt_watchlist.remove(_psym)
                    st.session_state["pt_watchlist"] = _pt_watchlist
                    _ptd["watchlist"] = _pt_watchlist
                    _pt_save(_ptd)
    else:
        st.info("No stocks tracked yet. Add some above.")

    st.divider()

    # ── Update + Train buttons ────────────────────────────────────────────────
    _pt_ub_col, _pt_tb_col, _pt_clr_col = st.columns([2, 2, 1])
    with _pt_ub_col:
        _pt_update = st.button(
            "🔄 Update — Run Today's Predictions",
            key="pt_update", type="primary", use_container_width=True,
            disabled=len(_pt_watchlist) == 0,
        )
    with _pt_tb_col:
        _pt_train = st.button(
            "🧠 Train Model — Learn from Past Predictions",
            key="pt_train", use_container_width=True,
        )
    with _pt_clr_col:
        if st.button("🗑️ Clear Data", key="pt_clear", use_container_width=True):
            _ptd["records"] = {}
            _pt_save(_ptd)
            st.success("Prediction history cleared.")
            st.rerun()

    # ── UPDATE: Run predictions ───────────────────────────────────────────────
    if _pt_update and _pt_watchlist:
        today_str = str(datetime.date.today())
        yesterday_str = str(datetime.date.today() - datetime.timedelta(days=1))

        _pt_progress = st.progress(0, text="Fetching live prices…")
        _pt_status   = st.empty()

        for _pi, _psym in enumerate(_pt_watchlist):
            _pt_progress.progress((_pi + 1) / len(_pt_watchlist),
                                  text=f"Predicting {_psym} ({_pi+1}/{len(_pt_watchlist)})…")
            _pt_status.info(f"Processing {_psym}…")

            # 1. Fetch yesterday's actual close and fill delta for yesterday's prediction
            _yest_actual = _pt_fetch_actual(_psym, yesterday_str)
            _sym_records = _pt_records.get(_psym, [])
            for _pr in _sym_records:
                if _pr.get("date") == yesterday_str and _pr.get("actual") is None and _yest_actual:
                    _pr["actual"] = _yest_actual
                    _price_at    = _pr.get("price_at_pred", _yest_actual)
                    _pr["delta_pct"] = round((_yest_actual - _pr["pred_tomorrow"]) / _pr["pred_tomorrow"] * 100, 2)
                    _pr["delta_abs"] = round(_yest_actual - _pr["pred_tomorrow"], 2)
                    _pr["direction_correct"] = (_yest_actual > _price_at) == (_pr["pred_tomorrow"] > _price_at)

            # 2. Run today's prediction
            _live_px = _yf_fast_price(_psym)
            if _live_px <= 0:
                continue
            _pred = _pt_predict_price(_psym, _live_px, curr)
            if not _pred or "error" in _pred:
                continue

            # Avoid duplicate for today
            _already_today = any(r.get("date") == today_str for r in _sym_records)
            if not _already_today:
                _sym_records.append({
                    "date":           today_str,
                    "price_at_pred":  _live_px,
                    "pred_today":     _pred["pred_today"],
                    "pred_tomorrow":  _pred["pred_tomorrow"],
                    "confidence":     _pred["confidence"],
                    "score":          _pred["score"],
                    "rsi":            _pred.get("rsi"),
                    "mom_5d":         _pred.get("mom_5d"),
                    "trend_up":       _pred.get("trend_up"),
                    "macd_bull":      _pred.get("macd_bull"),
                    "atr_pct":        _pred.get("atr_pct"),
                    "signals":        _pred.get("signals", []),
                    "actual":         None,    # filled tomorrow
                    "delta_pct":      None,
                    "delta_abs":      None,
                    "direction_correct": None,
                })

            _pt_records[_psym] = _sym_records

        # Prune old data
        _pt_records = _pt_prune(_pt_records)
        _ptd["records"]   = _pt_records
        _ptd["watchlist"] = _pt_watchlist
        _ptd["last_update"] = today_str
        _pt_save(_ptd)
        _pt_progress.empty(); _pt_status.empty()
        st.success(f"✅ Predictions updated for {len(_pt_watchlist)} stocks on {today_str}. "
                   f"Yesterday's actuals filled where available.")
        st.rerun()

    # ── TRAIN MODEL ──────────────────────────────────────────────────────────
    if _pt_train:
        with st.spinner("Analysing prediction history and training model…"):
            _new_weights, _train_summary = _pt_train_model(_ptd)
        if _new_weights:
            _ptd["weights"]    = _new_weights
            _ptd["model_meta"] = {
                "last_trained": str(datetime.date.today()),
                "n_records":    sum(len(v) for v in _ptd["records"].values()),
            }
            _pt_save(_ptd)
            st.success("✅ Model trained! Weights updated and saved.")
        st.markdown(_train_summary)

    # ── Dashboard table ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Prediction History & Delta Tracking")

    if _pt_meta.get("last_trained"):
        st.caption(f"Model last trained: {_pt_meta['last_trained']} on {_pt_meta.get('n_records',0)} data points. "
                   f"Weights: {len(_pt_weights)} signals calibrated.")

    if not _pt_records:
        st.info("No prediction data yet. Add stocks and click **Update** to start tracking.")
    else:
        # Build summary table — latest row per symbol
        _summary_rows = []
        for _psym in _pt_watchlist:
            _rows = sorted(_pt_records.get(_psym, []), key=lambda r: r["date"], reverse=True)
            if not _rows:
                continue
            _latest = _rows[0]
            _prev   = _rows[1] if len(_rows) > 1 else {}

            # Accuracy over available history
            _done = [r for r in _rows if r.get("delta_pct") is not None]
            _avg_delta = sum(abs(r["delta_pct"]) for r in _done) / len(_done) if _done else None
            _dir_acc   = sum(1 for r in _done if r.get("direction_correct")) / len(_done) * 100 if _done else None

            _summary_rows.append({
                "Symbol":         _psym,
                "Date":           _latest["date"],
                "Price at Pred":  f"${_latest.get('price_at_pred',0):.2f}",
                "Pred Today":     f"${_latest.get('pred_today',0):.2f}",
                "Pred Tomorrow":  f"${_latest.get('pred_tomorrow',0):.2f}",
                "Confidence":     f"{_latest.get('confidence',0):.0f}%",
                "Actual (yest)":  f"${_prev.get('actual',0):.2f}" if _prev.get("actual") else "—",
                "Delta %":        f"{_prev.get('delta_pct',0):+.2f}%" if _prev.get("delta_pct") is not None else "—",
                "Dir Correct":    "✅" if _prev.get("direction_correct") else ("❌" if _prev.get("direction_correct") is False else "—"),
                "Avg Delta (hist)": f"{_avg_delta:.2f}%" if _avg_delta is not None else "—",
                "Dir Accuracy":   f"{_dir_acc:.0f}%" if _dir_acc is not None else "—",
                "RSI":            f"{_latest.get('rsi',0):.0f}" if _latest.get("rsi") else "—",
                "Trend":          "↑" if _latest.get("trend_up") else "↓",
            })

        if _summary_rows:
            _pt_df = pd.DataFrame(_summary_rows)

            def _color_delta(val):
                try:
                    v = float(str(val).replace("%","").replace("+",""))
                    if v > 2:   return "color: #ff5252"
                    if v < -2:  return "color: #ff5252"
                    return "color: #00c853"
                except Exception:
                    return ""

            st.dataframe(
                _pt_df.style.map(_color_delta, subset=["Delta %"]),
                use_container_width=True, hide_index=True,
            )

        # ── Per-stock detail expanders ────────────────────────────────────────
        st.markdown("### 📈 Per-Stock Prediction History")
        for _psym in _pt_watchlist:
            _rows = sorted(_pt_records.get(_psym, []), key=lambda r: r["date"], reverse=True)
            if not _rows:
                continue
            _done = [r for r in _rows if r.get("delta_pct") is not None]
            _avg_d = sum(abs(r["delta_pct"]) for r in _done) / len(_done) if _done else None
            _dir_a = sum(1 for r in _done if r.get("direction_correct")) / len(_done) * 100 if _done else None

            _exp_lbl = (f"**{_psym}** — {len(_rows)} predictions"
                        + (f" | Avg delta: {_avg_d:.2f}%" if _avg_d else "")
                        + (f" | Direction accuracy: {_dir_a:.0f}%" if _dir_a else ""))
            with st.expander(_exp_lbl, expanded=False):
                # Signals from latest prediction
                _latest = _rows[0]
                if _latest.get("signals"):
                    st.markdown("**Signals used in latest prediction:**")
                    for _sig in _latest["signals"]:
                        st.markdown(f"  - {_sig}")

                # History table
                _hist_rows = []
                for _r in _rows[:30]:
                    _hist_rows.append({
                        "Date":        _r["date"],
                        "Price":       f"${_r.get('price_at_pred',0):.2f}",
                        "Pred Today":  f"${_r.get('pred_today',0):.2f}",
                        "Pred Tomor":  f"${_r.get('pred_tomorrow',0):.2f}",
                        "Actual":      f"${_r['actual']:.2f}" if _r.get("actual") else "Pending",
                        "Delta %":     f"{_r['delta_pct']:+.2f}%" if _r.get("delta_pct") is not None else "—",
                        "Delta $":     f"${_r['delta_abs']:+.2f}" if _r.get("delta_abs") is not None else "—",
                        "Dir ✓":       "✅" if _r.get("direction_correct") else ("❌" if _r.get("direction_correct") is False else "—"),
                        "Conf":        f"{_r.get('confidence',0):.0f}%",
                        "RSI":         f"{_r.get('rsi',0):.0f}" if _r.get("rsi") else "—",
                        "Score":       f"{_r.get('score',0):+.2f}",
                    })
                if _hist_rows:
                    st.dataframe(pd.DataFrame(_hist_rows), use_container_width=True, hide_index=True)

                # Accuracy chart — predicted vs actual over time
                _charted = [r for r in reversed(_rows) if r.get("actual")]
                if len(_charted) >= 2:
                    _ch_fig = go.Figure()
                    _ch_fig.add_trace(go.Scatter(
                        x=[r["date"] for r in _charted],
                        y=[r["pred_tomorrow"] for r in _charted],
                        name="Predicted", line=dict(color="#2979ff", dash="dash"), mode="lines+markers"
                    ))
                    _ch_fig.add_trace(go.Scatter(
                        x=[r["date"] for r in _charted],
                        y=[r["actual"] for r in _charted],
                        name="Actual", line=dict(color="#00c853"), mode="lines+markers"
                    ))
                    _ch_fig.update_layout(
                        template="plotly_dark", height=280,
                        margin=dict(t=20, b=20), showlegend=True,
                        title=f"{_psym} — Predicted vs Actual",
                    )
                    st.plotly_chart(_ch_fig, use_container_width=True)

    st.divider()
    st.caption(
        "**How the model works:** Prediction = current price ± (signal score × ATR). "
        "Signals: RSI, MACD cross, EMA20/50, volume spike, Bollinger band position, 5-day momentum, macro regime. "
        "**Train Model** analyses direction accuracy per signal and adjusts weights — signals that correctly predicted "
        "up/down direction get boosted, wrong signals get reduced. "
        "Delta = |Predicted − Actual| / Actual. Lower delta = better calibration."
    )
