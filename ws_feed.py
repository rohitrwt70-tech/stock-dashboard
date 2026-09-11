"""
Finnhub WebSocket Feed — Option A live data provider.

Run this in a separate terminal:
    python ws_feed.py

It connects to Finnhub WebSocket, aggregates trade ticks into 1m/5m candles,
and writes the latest data to ws_live.json every second.
Stock_Predictor.py reads ws_live.json to show live prices without polling.
"""

import os, json, time, threading, collections, datetime, ssl
from pathlib import Path
from dotenv import load_dotenv

# Fix macOS SSL certificate issue
try:
    import certifi
    _SSL_OPTS = {"ca_certs": certifi.where(), "cert_reqs": ssl.CERT_REQUIRED}
except ImportError:
    _SSL_OPTS = {"cert_reqs": ssl.CERT_NONE}

load_dotenv(override=True)

WS_DATA_FILE   = Path("./ws_live.json")
FINNHUB_KEY    = os.getenv("FINNHUB_API_KEY", "")
INTERVALS      = [1, 5, 15]   # minutes to aggregate into
MAX_CANDLES    = 100           # candles to keep per symbol per interval
WRITE_INTERVAL = 1.0          # seconds between file writes

# ── In-memory state ───────────────────────────────────────────────────────────
_lock        = threading.Lock()
_ticks       = collections.defaultdict(list)   # sym → list of (ts, price, volume)
_candles     = collections.defaultdict(lambda: {m: [] for m in INTERVALS})  # sym → {interval: [candle]}
_open_candle = collections.defaultdict(lambda: {m: None for m in INTERVALS}) # current partial candle
_subscribed  = set()

def _floor_ts(ts_ms: int, interval_m: int) -> int:
    """Floor a millisecond timestamp to the nearest interval boundary."""
    interval_ms = interval_m * 60 * 1000
    return (ts_ms // interval_ms) * interval_ms

def _process_tick(sym: str, price: float, volume: float, ts_ms: int):
    """Aggregate a tick into running candles for all intervals."""
    with _lock:
        _ticks[sym].append((ts_ms, price, volume))
        # Keep only last 10 minutes of raw ticks
        cutoff = ts_ms - 10 * 60 * 1000
        _ticks[sym] = [(t, p, v) for t, p, v in _ticks[sym] if t >= cutoff]

        for m in INTERVALS:
            bucket = _floor_ts(ts_ms, m)
            oc = _open_candle[sym][m]

            if oc is None or oc["t"] != bucket:
                # Close previous candle
                if oc is not None:
                    _candles[sym][m].append(oc)
                    _candles[sym][m] = _candles[sym][m][-MAX_CANDLES:]
                # Open new candle
                _open_candle[sym][m] = {
                    "t": bucket,
                    "o": price, "h": price, "l": price, "c": price,
                    "v": volume,
                }
            else:
                oc["h"] = max(oc["h"], price)
                oc["l"] = min(oc["l"], price)
                oc["c"] = price
                oc["v"] += volume

def _build_output() -> dict:
    """Snapshot current state into a serialisable dict."""
    with _lock:
        out = {}
        for sym in set(list(_ticks.keys()) + list(_candles.keys())):
            tks = _ticks.get(sym, [])
            last_price = tks[-1][1] if tks else None
            last_ts    = tks[-1][0] if tks else None

            candles_out = {}
            for m in INTERVALS:
                closed = _candles[sym][m][:]
                partial = _open_candle[sym][m]
                all_c = closed + ([partial] if partial else [])
                candles_out[str(m)] = all_c[-MAX_CANDLES:]

            out[sym] = {
                "price":       last_price,
                "ts_ms":       last_ts,
                "updated":     datetime.datetime.utcnow().isoformat(),
                "candles":     candles_out,
            }
        return out

def _writer_thread():
    """Write ws_live.json every WRITE_INTERVAL seconds."""
    while True:
        try:
            data = _build_output()
            WS_DATA_FILE.write_text(json.dumps(data))
        except Exception as e:
            print(f"[ws_feed] write error: {e}")
        time.sleep(WRITE_INTERVAL)

# ── WebSocket callbacks ───────────────────────────────────────────────────────
def _on_message(ws, message):
    try:
        msg = json.loads(message)
        if msg.get("type") != "trade":
            return
        for trade in msg.get("data", []):
            sym    = trade.get("s", "")
            price  = float(trade.get("p", 0))
            volume = float(trade.get("v", 0))
            ts_ms  = int(trade.get("t", time.time() * 1000))
            if sym and price > 0:
                _process_tick(sym, price, volume, ts_ms)
    except Exception as e:
        print(f"[ws_feed] message error: {e}")

def _on_error(ws, error):
    print(f"[ws_feed] error: {error}")

def _on_close(ws, *args):
    print("[ws_feed] connection closed — reconnecting in 5s…")

def _on_open(ws):
    print(f"[ws_feed] connected to Finnhub WebSocket")
    # Subscribe to any symbols already in the control file
    for sym in _read_control():
        _subscribed.add(sym)
    # Re-subscribe to all tracked symbols
    for sym in _subscribed:
        try:
            ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
            print(f"[ws_feed] subscribed: {sym}")
        except Exception as e:
            print(f"[ws_feed] subscribe error for {sym}: {e}")

def subscribe(ws, sym: str):
    if sym not in _subscribed:
        _subscribed.add(sym)
        try:
            if ws.sock and ws.sock.connected:
                ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                print(f"[ws_feed] subscribed: {sym}")
        except Exception:
            pass  # will retry on reconnect via _on_open

# ── Control file — Streamlit writes symbols here to subscribe ─────────────────
CONTROL_FILE = Path("./ws_control.json")

def _read_control() -> list[str]:
    try:
        if CONTROL_FILE.exists():
            data = json.loads(CONTROL_FILE.read_text())
            return data.get("symbols", [])
    except Exception:
        pass
    return []

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not FINNHUB_KEY:
        print("[ws_feed] ERROR: FINNHUB_API_KEY not set in .env")
        return

    print(f"[ws_feed] Starting — writing to {WS_DATA_FILE}")

    # Start writer thread
    t = threading.Thread(target=_writer_thread, daemon=True)
    t.start()

    import websocket as _ws

    def _run_ws():
        while True:
            try:
                ws_app = _ws.WebSocketApp(
                    f"wss://ws.finnhub.io?token={FINNHUB_KEY}",
                    on_open=_on_open,
                    on_message=_on_message,
                    on_error=_on_error,
                    on_close=_on_close,
                )

                # Poll control file in a thread to subscribe to new symbols
                def _control_poller(ws_ref):
                    while True:
                        try:
                            syms = _read_control()
                            for s in syms:
                                if s not in _subscribed:
                                    subscribe(ws_ref, s)
                        except Exception:
                            pass
                        time.sleep(2)

                ct = threading.Thread(target=_control_poller, args=(ws_app,), daemon=True)
                ct.start()

                ws_app.run_forever(ping_interval=20, ping_timeout=10, sslopt=_SSL_OPTS)
            except Exception as e:
                print(f"[ws_feed] WebSocket error: {e}")
            print("[ws_feed] reconnecting in 5s…")
            time.sleep(5)

    _run_ws()

if __name__ == "__main__":
    main()
