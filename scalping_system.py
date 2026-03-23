"""
Intraday Scalping System
========================
A robust, execution-aware intraday scalping algorithm.

Philosophy
----------
- No "best strategy" – robustness over high returns.
- Avoid indicator stacking and correlated signals.
- Focus on market regime, execution quality, and risk management.

Architecture
------------
1.  Data Layer      – yfinance (5-min & 15-min bars)
2.  Feature Layer   – ADX, ATR, Supertrend, VWAP, rolling volatility
3.  Regime Engine   – classifies each bar as Trending / Mean-Reverting / High-Vol
4.  Signal Engine   – generates entry signals only in Trending regime
5.  Risk Engine     – ATR-based SL, 1:2 RR take-profit, position sizing (0.5% risk)
6.  Execution Layer – simulates slippage + commissions; live via Alpaca paper API
7.  Backtest Engine – walk-forward simulation with realistic costs
8.  Circuit Breaker – halts trading after 3 consecutive losses

Weaknesses
----------
- Supertrend and ADX can lag on thin / fast moves.
- VWAP is session-reset; overnight gaps distort early readings.
- Slippage model is simplified (fixed bps); real slippage is dynamic.
- No order-book data → cannot model market-impact accurately.
- Parameters (ADX threshold, ATR multiplier) need periodic recalibration.

Suggested Improvements
-----------------------
- Replace rule-based signals with a lightweight ML classifier (XGBoost/LightGBM)
  trained on regime + microstructure features.
- Add order-flow imbalance (Level-2 data) for entry confirmation.
- Use adaptive position sizing based on realised volatility.
- Implement walk-forward optimisation to avoid look-ahead bias.
- Add cross-asset correlation check to avoid trading correlated instruments.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Optional Alpaca import – only required for live trading
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLIPPAGE_BPS: float = 5.0          # one-way slippage in basis points
COMMISSION_BPS: float = 2.0        # one-way commission in basis points
RISK_PER_TRADE: float = 0.005      # 0.5 % of equity risked per trade
RR_RATIO: float = 2.0              # minimum reward-to-risk ratio
ADX_TREND_THRESHOLD: float = 25.0  # ADX > 25 → trending
ADX_PERIOD: int = 14
ATR_PERIOD: int = 14
SUPERTREND_MULTIPLIER: float = 3.0
SUPERTREND_PERIOD: int = 10
VOL_ROLLING_WINDOW: int = 20       # bars for rolling volatility
VOL_HIGH_PERCENTILE: float = 80.0  # above this rolling-vol percentile → high-vol
VOLUME_MIN_MULTIPLIER: float = 1.2 # entry bar volume must be ≥ 1.2× rolling avg
VWAP_DEV_MIN: float = 0.001        # minimum VWAP deviation to consider a breakout
MAX_CONSECUTIVE_LOSSES: int = 3
LIQUID_HOURS: Tuple[int, int] = (9, 16)   # UTC hours; adjust to 14-21 UTC for NYSE regular session


# ===========================================================================
# DATA LAYER
# ===========================================================================

def fetch_data(
    symbol: str,
    period: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Download OHLCV data from yfinance and return a clean DataFrame."""
    log.info("Fetching %s data for %s (period=%s)", interval, symbol, period)
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} at interval {interval}")
    df.index = pd.to_datetime(df.index, utc=True)
    # Flatten multi-level columns produced by yfinance for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    log.info("Fetched %d bars for %s", len(df), symbol)
    return df


# ===========================================================================
# FEATURE LAYER
# ===========================================================================

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    Average Directional Index (ADX).
    Returns a Series of ADX values (0-100); values > 25 indicate trending.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = compute_atr(df, period=1)  # raw true range
    atr = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    )

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(0)


def compute_supertrend(
    df: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> pd.Series:
    """
    Supertrend indicator.
    Returns +1 for bullish (price above supertrend line) and -1 for bearish.
    """
    atr = compute_atr(df, period=period)
    hl2 = (df["High"] + df["Low"]) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend_dir = pd.Series(np.ones(len(df)), index=df.index)
    close = df["Close"].values
    upper = upper_band.values.copy()
    lower = lower_band.values.copy()
    direction = supertrend_dir.values.copy()

    for i in range(1, len(df)):
        # Adjust bands to only tighten, never widen
        lower[i] = lower[i] if lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1] else lower[i - 1]
        upper[i] = upper[i] if upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1] else upper[i - 1]

        if close[i] > upper[i - 1]:
            direction[i] = 1.0
        elif close[i] < lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

    return pd.Series(direction, index=df.index)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session VWAP reset each calendar day.
    Returns only the VWAP series; VWAP deviation is computed separately in add_features.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tpv = (typical * df["Volume"]).groupby(df.index.date).cumsum()
    cum_vol = df["Volume"].groupby(df.index.date).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return vwap


def compute_rolling_volatility(df: pd.DataFrame, window: int = VOL_ROLLING_WINDOW) -> pd.Series:
    """Annualised rolling close-to-close volatility (as a fraction)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252 * 78)  # 78 five-min bars per regular trading day (9:30–16:00)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features and attach them to the DataFrame."""
    df = df.copy()
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["Supertrend"] = compute_supertrend(df)
    df["VWAP"] = compute_vwap(df)
    df["VWAP_Dev"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
    df["RollingVol"] = compute_rolling_volatility(df)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolAvg20"].replace(0, np.nan)

    # Rolling vol percentile (rank within the trailing 100 bars)
    df["VolPercentile"] = (
        df["RollingVol"]
        .rolling(100)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )
    df.dropna(inplace=True)
    return df


# ===========================================================================
# REGIME DETECTION ENGINE
# ===========================================================================

class Regime:
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"


def detect_regime(row: pd.Series) -> str:
    """
    Classify the current bar's market regime.

    Priority:
      1. High-volatility → avoid trading
      2. Trending (ADX > threshold + Supertrend confirms) → breakout/pullback
      3. Mean-reverting → skip (no mean-reversion strategy implemented per spec)
    """
    if row["VolPercentile"] >= VOL_HIGH_PERCENTILE:
        return Regime.HIGH_VOLATILITY

    if row["ADX"] >= ADX_TREND_THRESHOLD:
        return Regime.TRENDING

    return Regime.MEAN_REVERTING


# ===========================================================================
# SIGNAL ENGINE
# ===========================================================================

@dataclass
class Signal:
    direction: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    regime: str
    timestamp: pd.Timestamp


def time_filter(ts: pd.Timestamp) -> bool:
    """Return True only during liquid trading hours defined by LIQUID_HOURS (UTC)."""
    return LIQUID_HOURS[0] <= ts.hour < LIQUID_HOURS[1]


def generate_signal(
    df: pd.DataFrame,
    idx: int,
) -> Optional[Signal]:
    """
    Evaluate the current bar and return a Signal if conditions are met.

    Entry conditions (ALL must hold):
    - Regime is Trending
    - Time-of-day filter passes
    - Supertrend direction matches the breakout direction
    - Current bar's volume ≥ 1.2× 20-bar average (volume confirmation)
    - VWAP deviation confirms breakout direction

    Entry is at the open of the NEXT bar (candle-close entry).
    Stop is ATR-based; target is 2× stop distance from entry.
    """
    row = df.iloc[idx]
    regime = detect_regime(row)

    if regime != Regime.TRENDING:
        return None

    if not time_filter(df.index[idx]):
        return None

    # Volume filter
    if row["VolRatio"] < VOLUME_MIN_MULTIPLIER:
        return None

    direction = None

    # Bullish breakout: price above VWAP, Supertrend bullish, strong volume
    if (
        row["Supertrend"] == 1.0
        and row["VWAP_Dev"] >= VWAP_DEV_MIN
        and row["Close"] > row["Open"]   # bullish candle close
    ):
        direction = "long"

    # Bearish breakout: price below VWAP, Supertrend bearish, strong volume
    elif (
        row["Supertrend"] == -1.0
        and row["VWAP_Dev"] <= -VWAP_DEV_MIN
        and row["Close"] < row["Open"]   # bearish candle close
    ):
        direction = "short"

    if direction is None:
        return None

    # Use next bar's open as entry (candle-close entry → next open execution)
    if idx + 1 >= len(df):
        return None

    entry_price = df.iloc[idx + 1]["Open"]
    atr = row["ATR"]

    if direction == "long":
        stop_loss = entry_price - atr
        take_profit = entry_price + RR_RATIO * atr
    else:
        stop_loss = entry_price + atr
        take_profit = entry_price - RR_RATIO * atr

    return Signal(
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=atr,
        regime=regime,
        timestamp=df.index[idx + 1],
    )


# ===========================================================================
# RISK ENGINE
# ===========================================================================

def position_size(equity: float, entry: float, stop: float) -> float:
    """
    Calculate share/unit quantity to risk RISK_PER_TRADE of equity.

    quantity = (equity × risk_per_trade) / |entry − stop|
    """
    risk_amount = equity * RISK_PER_TRADE
    risk_per_unit = abs(entry - stop)
    if risk_per_unit < 1e-8:
        return 0.0
    return risk_amount / risk_per_unit


# ===========================================================================
# EXECUTION LAYER (simulation)
# ===========================================================================

def apply_slippage(price: float, direction: str, is_entry: bool) -> float:
    """
    Apply fixed-bps slippage.
    Entries: adverse slippage (long pays more, short receives less).
    Exits:   adverse slippage (long receives less, short pays more).
    """
    bps = SLIPPAGE_BPS / 10_000
    if (direction == "long" and is_entry) or (direction == "short" and not is_entry):
        return price * (1 + bps)
    return price * (1 - bps)


def apply_commission(pnl: float, entry: float, exit_price: float, qty: float) -> float:
    """Deduct round-trip commission from PnL."""
    comm = COMMISSION_BPS / 10_000 * (entry + exit_price) * qty
    return pnl - comm


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    qty: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str   # "tp", "sl", "eod"
    regime: str


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "sl": t.stop_loss,
                    "tp": t.take_profit,
                    "qty": t.qty,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "exit_reason": t.exit_reason,
                    "regime": t.regime,
                }
                for t in self.trades
            ]
        )
        return df

    def metrics(self) -> dict:
        df = self.summary()
        if df.empty:
            return {}

        wins = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] <= 0]
        total = len(df)
        win_rate = len(wins) / total if total else 0
        avg_win = wins["net_pnl"].mean() if not wins.empty else 0
        avg_loss = losses["net_pnl"].mean() if not losses.empty else 0
        profit_factor = (
            wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
            if losses["net_pnl"].sum() != 0
            else float("inf")
        )
        total_net = df["net_pnl"].sum()
        max_dd = _max_drawdown(self.equity_curve)

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "total_net_pnl": round(total_net, 4),
            "max_drawdown": round(max_dd, 4),
            "avg_rr_realised": round(
                abs(avg_win / avg_loss) if avg_loss != 0 else 0, 4
            ),
        }


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()


def run_backtest(
    df: pd.DataFrame,
    initial_equity: float = 100_000.0,
) -> BacktestResult:
    """
    Walk-forward bar-by-bar backtest.

    - Signals generated on close of bar i.
    - Entry executed at open of bar i+1 with slippage.
    - Each subsequent bar checks SL / TP (using high/low of bar).
    - Circuit breaker halts after MAX_CONSECUTIVE_LOSSES consecutive losses.
    - EOD flat: any open position is closed at market on the last bar of the day.
    """
    result = BacktestResult()
    equity = initial_equity
    equity_series: list = [equity]
    timestamps: list = [df.index[0]]

    in_trade = False
    active: Optional[dict] = None
    consecutive_losses = 0
    halted = False

    for i in range(len(df) - 1):
        bar = df.iloc[i]

        # Manage open trade first
        if in_trade and active is not None:
            next_bar = df.iloc[i + 1]
            exit_price = None
            exit_reason = None

            if active["direction"] == "long":
                if next_bar["Low"] <= active["sl"]:
                    exit_price = apply_slippage(active["sl"], "long", is_entry=False)
                    exit_reason = "sl"
                elif next_bar["High"] >= active["tp"]:
                    exit_price = apply_slippage(active["tp"], "long", is_entry=False)
                    exit_reason = "tp"
            else:
                if next_bar["High"] >= active["sl"]:
                    exit_price = apply_slippage(active["sl"], "short", is_entry=False)
                    exit_reason = "sl"
                elif next_bar["Low"] <= active["tp"]:
                    exit_price = apply_slippage(active["tp"], "short", is_entry=False)
                    exit_reason = "tp"

            # End-of-day flat: close at next open if same date differs
            if exit_price is None:
                current_date = df.index[i].date()
                next_date = df.index[i + 1].date()
                if next_date != current_date:
                    exit_price = apply_slippage(next_bar["Open"], active["direction"], is_entry=False)
                    exit_reason = "eod"

            if exit_price is not None:
                direction = active["direction"]
                qty = active["qty"]
                entry = active["entry_price"]

                if direction == "long":
                    gross_pnl = (exit_price - entry) * qty
                else:
                    gross_pnl = (entry - exit_price) * qty

                net_pnl = apply_commission(gross_pnl, entry, exit_price, qty)
                equity += net_pnl

                if net_pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    log.warning(
                        "Circuit breaker triggered at %s after %d consecutive losses.",
                        df.index[i + 1],
                        consecutive_losses,
                    )
                    halted = True

                result.trades.append(
                    Trade(
                        direction=direction,
                        entry_time=active["entry_time"],
                        exit_time=df.index[i + 1],
                        entry_price=entry,
                        exit_price=exit_price,
                        stop_loss=active["sl"],
                        take_profit=active["tp"],
                        qty=qty,
                        gross_pnl=round(gross_pnl, 4),
                        net_pnl=round(net_pnl, 4),
                        exit_reason=exit_reason,
                        regime=active["regime"],
                    )
                )
                in_trade = False
                active = None

        equity_series.append(equity)
        timestamps.append(df.index[i + 1])

        # Skip new entries if halted or already in a trade
        if halted or in_trade:
            continue

        signal = generate_signal(df, i)
        if signal is None:
            continue

        qty = position_size(equity, signal.entry_price, signal.stop_loss)
        if qty <= 0:
            continue

        # Apply entry slippage
        actual_entry = apply_slippage(signal.entry_price, signal.direction, is_entry=True)

        active = {
            "direction": signal.direction,
            "entry_price": actual_entry,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "qty": qty,
            "entry_time": signal.timestamp,
            "regime": signal.regime,
        }
        in_trade = True

    result.equity_curve = pd.Series(equity_series, index=timestamps)
    return result


# ===========================================================================
# MULTI-TIMEFRAME CONFIRMATION
# ===========================================================================

def htf_trend_direction(df_15m: pd.DataFrame, ts: pd.Timestamp) -> Optional[str]:
    """
    Return the 15-min Supertrend direction at or just before `ts`.
    Used as a higher-timeframe filter for 5-min entries.
    """
    past = df_15m[df_15m.index <= ts]
    if past.empty:
        return None
    return "long" if past.iloc[-1]["Supertrend"] == 1.0 else "short"


def run_mtf_backtest(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    initial_equity: float = 100_000.0,
) -> BacktestResult:
    """
    Backtest with 15-min higher-timeframe trend filter applied to 5-min signals.
    Only trades in the direction of the 15-min Supertrend.
    """
    result = BacktestResult()
    equity = initial_equity
    equity_series: list = [equity]
    timestamps: list = [df_5m.index[0]]

    in_trade = False
    active: Optional[dict] = None
    consecutive_losses = 0
    halted = False

    for i in range(len(df_5m) - 1):
        if in_trade and active is not None:
            next_bar = df_5m.iloc[i + 1]
            exit_price = None
            exit_reason = None

            if active["direction"] == "long":
                if next_bar["Low"] <= active["sl"]:
                    exit_price = apply_slippage(active["sl"], "long", is_entry=False)
                    exit_reason = "sl"
                elif next_bar["High"] >= active["tp"]:
                    exit_price = apply_slippage(active["tp"], "long", is_entry=False)
                    exit_reason = "tp"
            else:
                if next_bar["High"] >= active["sl"]:
                    exit_price = apply_slippage(active["sl"], "short", is_entry=False)
                    exit_reason = "sl"
                elif next_bar["Low"] <= active["tp"]:
                    exit_price = apply_slippage(active["tp"], "short", is_entry=False)
                    exit_reason = "tp"

            if exit_price is None:
                current_date = df_5m.index[i].date()
                next_date = df_5m.index[i + 1].date()
                if next_date != current_date:
                    exit_price = apply_slippage(next_bar["Open"], active["direction"], is_entry=False)
                    exit_reason = "eod"

            if exit_price is not None:
                direction = active["direction"]
                qty = active["qty"]
                entry = active["entry_price"]

                if direction == "long":
                    gross_pnl = (exit_price - entry) * qty
                else:
                    gross_pnl = (entry - exit_price) * qty

                net_pnl = apply_commission(gross_pnl, entry, exit_price, qty)
                equity += net_pnl

                if net_pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    log.warning("Circuit breaker triggered at %s.", df_5m.index[i + 1])
                    halted = True

                result.trades.append(
                    Trade(
                        direction=direction,
                        entry_time=active["entry_time"],
                        exit_time=df_5m.index[i + 1],
                        entry_price=entry,
                        exit_price=exit_price,
                        stop_loss=active["sl"],
                        take_profit=active["tp"],
                        qty=qty,
                        gross_pnl=round(gross_pnl, 4),
                        net_pnl=round(net_pnl, 4),
                        exit_reason=exit_reason,
                        regime=active["regime"],
                    )
                )
                in_trade = False
                active = None

        equity_series.append(equity)
        timestamps.append(df_5m.index[i + 1])

        if halted or in_trade:
            continue

        signal = generate_signal(df_5m, i)
        if signal is None:
            continue

        # HTF filter: only take trades aligned with 15-min trend
        htf_dir = htf_trend_direction(df_15m, df_5m.index[i])
        if htf_dir is not None and htf_dir != signal.direction:
            continue

        qty = position_size(equity, signal.entry_price, signal.stop_loss)
        if qty <= 0:
            continue

        actual_entry = apply_slippage(signal.entry_price, signal.direction, is_entry=True)
        active = {
            "direction": signal.direction,
            "entry_price": actual_entry,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "qty": qty,
            "entry_time": signal.timestamp,
            "regime": signal.regime,
        }
        in_trade = True

    result.equity_curve = pd.Series(equity_series, index=timestamps)
    return result


# ===========================================================================
# ALPACA LIVE TRADING
# ===========================================================================

class AlpacaTrader:
    """
    Paper-trading execution via Alpaca REST API.

    Requires environment variables:
        ALPACA_API_KEY    – your Alpaca paper key
        ALPACA_SECRET_KEY – your Alpaca paper secret

    Usage:
        trader = AlpacaTrader()
        trader.submit_order(symbol="SPY", qty=10, side="buy")
    """

    def __init__(self) -> None:
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-trade-api is not installed. "
                "Install with: pip install alpaca-py"
            )
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )
        self.client = TradingClient(api_key, secret_key, paper=True)
        log.info("AlpacaTrader initialised (paper trading mode).")

    def get_account(self) -> dict:
        acct = self.client.get_account()
        return {
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "cash": float(acct.cash),
        }

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
        time_in_force: str = "gtc",
    ) -> dict:
        """Submit a market order via Alpaca paper API."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if time_in_force == "gtc" else TimeInForce.DAY

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty),
            side=order_side,
            time_in_force=tif,
        )
        order = self.client.submit_order(order_data=order_data)
        log.info("Order submitted: %s %s %s @ market", side.upper(), qty, symbol)
        return {"order_id": order.id, "status": order.status}

    def close_position(self, symbol: str) -> None:
        """Close all open positions for a symbol."""
        self.client.close_position(symbol)
        log.info("Position closed for %s.", symbol)


# ===========================================================================
# LIVE SCANNER (paper trading loop)
# ===========================================================================

def live_trading_loop(
    symbol: str,
    equity: float,
    poll_seconds: int = 300,  # 5-min bars → poll every 5 min
    max_trades: int = 5,
) -> None:
    """
    Simplified live scanning loop for paper trading.

    Every `poll_seconds`:
      1. Fetch fresh 5-min and 15-min data.
      2. Compute features.
      3. Evaluate the latest completed bar for a signal.
      4. Submit order via Alpaca if signal fires.

    WARNING: This is a reference skeleton. Do NOT use in production without
    additional error handling, position tracking, and connectivity checks.
    """
    if not ALPACA_AVAILABLE:
        log.error("Alpaca package not installed. Cannot run live trading.")
        return

    trader = AlpacaTrader()
    trades_today = 0
    consecutive_losses = 0
    open_position: Optional[dict] = None

    log.info("Starting live loop for %s. Polling every %ds.", symbol, poll_seconds)

    while trades_today < max_trades:
        try:
            df_5m = fetch_data(symbol, period="5d", interval="5m")
            df_15m = fetch_data(symbol, period="5d", interval="15m")
            df_5m = add_features(df_5m)
            df_15m = add_features(df_15m)

            now = datetime.now(tz=timezone.utc)
            if not time_filter(now):
                log.info("Outside liquid hours. Sleeping %ds.", poll_seconds)
                time.sleep(poll_seconds)
                continue

            # Check open position exit conditions
            if open_position is not None:
                latest_close = df_5m.iloc[-1]["Close"]
                pos = open_position
                exit_triggered = False

                if pos["direction"] == "long":
                    if latest_close <= pos["sl"] or latest_close >= pos["tp"]:
                        exit_triggered = True
                else:
                    if latest_close >= pos["sl"] or latest_close <= pos["tp"]:
                        exit_triggered = True

                if exit_triggered:
                    close_side = "sell" if pos["direction"] == "long" else "buy"
                    trader.submit_order(symbol, pos["qty"], close_side)
                    net_pnl_est = (
                        (latest_close - pos["entry"]) * pos["qty"]
                        if pos["direction"] == "long"
                        else (pos["entry"] - latest_close) * pos["qty"]
                    )
                    if net_pnl_est <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    open_position = None
                    trades_today += 1

                    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                        log.warning("Circuit breaker triggered. Stopping for the day.")
                        break

            # Look for new signal on second-to-last completed bar
            if open_position is None and consecutive_losses < MAX_CONSECUTIVE_LOSSES:
                signal = generate_signal(df_5m, len(df_5m) - 2)
                if signal is not None:
                    htf_dir = htf_trend_direction(df_15m, df_5m.index[-2])
                    if htf_dir is None or htf_dir == signal.direction:
                        acct = trader.get_account()
                        qty = position_size(acct["equity"], signal.entry_price, signal.stop_loss)
                        if qty >= 1:
                            side = "buy" if signal.direction == "long" else "sell"
                            trader.submit_order(symbol, qty, side)
                            open_position = {
                                "direction": signal.direction,
                                "entry": signal.entry_price,
                                "sl": signal.stop_loss,
                                "tp": signal.take_profit,
                                "qty": qty,
                            }

        except (ValueError, KeyError, IndexError) as exc:
            log.error("Recoverable error in live loop: %s", exc)
        except EnvironmentError as exc:
            log.critical("Authentication/connectivity error – stopping loop: %s", exc)
            break
        except Exception as exc:  # noqa: BLE001
            log.error("Unexpected error in live loop: %s", exc, exc_info=True)

        time.sleep(poll_seconds)

    log.info("Live loop ended. Trades today: %d", trades_today)


# ===========================================================================
# MAIN – BACKTEST DEMO
# ===========================================================================

def main() -> None:
    symbol = "SPY"

    # Fetch and prepare data
    df_5m_raw = fetch_data(symbol, period="60d", interval="5m")
    df_15m_raw = fetch_data(symbol, period="60d", interval="15m")

    log.info("Computing features for 5-min bars…")
    df_5m = add_features(df_5m_raw)

    log.info("Computing features for 15-min bars…")
    df_15m = add_features(df_15m_raw)

    # --- Single timeframe backtest (5-min only) ---
    log.info("Running single-timeframe (5m) backtest…")
    bt_5m = run_backtest(df_5m, initial_equity=100_000.0)
    metrics_5m = bt_5m.metrics()
    log.info("=== 5-min Backtest Metrics ===")
    for k, v in metrics_5m.items():
        log.info("  %-25s %s", k, v)

    # --- Multi-timeframe backtest (5-min signal + 15-min HTF filter) ---
    log.info("Running multi-timeframe (5m + 15m) backtest…")
    bt_mtf = run_mtf_backtest(df_5m, df_15m, initial_equity=100_000.0)
    metrics_mtf = bt_mtf.metrics()
    log.info("=== MTF Backtest Metrics ===")
    for k, v in metrics_mtf.items():
        log.info("  %-25s %s", k, v)

    # Print trade log sample
    trades_df = bt_mtf.summary()
    if not trades_df.empty:
        log.info("Sample trades (last 5):\n%s", trades_df.tail(5).to_string())
    else:
        log.info("No trades generated in MTF backtest.")

    # Equity curve stats
    eq = bt_mtf.equity_curve
    if not eq.empty:
        total_return = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0] * 100
        log.info("MTF total return: %.2f%%", total_return)


if __name__ == "__main__":
    main()
