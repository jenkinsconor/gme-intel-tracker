"""
GME Dashboard — Support/Resistance + Volume Spike Detector

Drop-in module for your existing dashboard. Works with a standard OHLCV
DataFrame indexed by datetime (or with a 'Date' column you can set as index).

What you get:
- Pivot-based Support/Resistance detection with level clustering + strength scoring
- Volume spike detection using rolling z-scores
- Optional plotting helpers to visualize S/R levels and spikes

Dependencies: pandas, numpy, matplotlib (for plotting helpers)

Author: you + ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Dict
import numpy as np
import pandas as pd


# -----------------------------
# Types
# -----------------------------
LevelType = Literal["support", "resistance", "mixed"]

@dataclass
class SRLevel:
    level: float
    strength: float
    last_touch: pd.Timestamp
    kind: LevelType
    touches: int


# -----------------------------
# Core utilities
# -----------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, (pd.DatetimeIndex,)):
        if 'Date' in df.columns:
            df = df.set_index(pd.to_datetime(df['Date']))
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")
    return df.sort_index()


def compute_pivots(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Identify swing highs/lows as simple pivots.

    A pivot high at t means price[t] is the local maximum within +/- window.
    A pivot low  at t means price[t] is the local minimum within +/- window.
    """
    df = _ensure_datetime_index(df)
    s = df[price_col]
    # rolling windows
    max_back = s.rolling(window + 1, min_periods=1).max()
    max_fwd = s[::-1].rolling(window + 1, min_periods=1).max()[::-1]
    min_back = s.rolling(window + 1, min_periods=1).min()
    min_fwd = s[::-1].rolling(window + 1, min_periods=1).min()[::-1]

    pivot_highs = (s >= max_back) & (s >= max_fwd)
    pivot_lows  = (s <= min_back) & (s <= min_fwd)

    # Remove edges where the lookahead/lookback is incomplete for cleaner signals
    pivot_highs.iloc[:window] = False
    pivot_highs.iloc[-window:] = False
    pivot_lows.iloc[:window] = False
    pivot_lows.iloc[-window:] = False

    return pivot_highs, pivot_lows


def _cluster_levels(level_values: np.ndarray, tolerance: float) -> List[Tuple[float, List[int]]]:
    """
    Cluster price levels that are within `tolerance` (absolute price units).

    Returns a list of tuples: (cluster_center_price, indices_in_cluster)
    where `indices_in_cluster` are the indices of the original level_values involved.
    """
    if len(level_values) == 0:
        return []

    sort_idx = np.argsort(level_values)
    lv = level_values[sort_idx]

    clusters: List[List[int]] = []
    current: List[int] = [sort_idx[0]]

    for i in range(1, len(lv)):
        if abs(lv[i] - lv[i-1]) <= tolerance:
            current.append(sort_idx[i])
        else:
            clusters.append(current)
            current = [sort_idx[i]]
    clusters.append(current)

    result: List[Tuple[float, List[int]]] = []
    for cl in clusters:
        prices = level_values[cl]
        center = float(np.average(prices))  # simple average as cluster center
        result.append((center, cl))
    return result


def get_support_resistance(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 5,
    tolerance_frac: float = 0.01,
    max_levels: int = 10,
    weight_recency: float = 0.9,
) -> List[SRLevel]:
    """
    Compute support/resistance levels by:
      1) Finding pivot highs/lows
      2) Clustering nearby levels within `tolerance_frac` of price
      3) Scoring clusters by touch count + recency (exponential decay)

    Parameters
    ----------
    tolerance_frac : float
        Fraction of current price used as absolute tolerance for clustering
        (e.g., 0.01 => cluster pivots within ±1% of price).
    weight_recency : float
        Exponential decay per day (0<weight<=1) applied to older touches.
    """
    df = _ensure_datetime_index(df)
    price = df[price_col]
    pivot_highs, pivot_lows = compute_pivots(df, price_col=price_col, window=window)

    pivot_prices: List[Tuple[pd.Timestamp, float, LevelType]] = []

    high_idx = pivot_highs[pivot_highs].index
    low_idx  = pivot_lows[pivot_lows].index

    for ts in high_idx:
        pivot_prices.append((ts, float(price.loc[ts]), "resistance"))
    for ts in low_idx:
        pivot_prices.append((ts, float(price.loc[ts]), "support"))


    if not pivot_prices:
        return []

    pivot_prices.sort(key=lambda x: x[1])
    levels = np.array([p[1] for p in pivot_prices])
    kinds = np.array([p[2] for p in pivot_prices], dtype=object)
    timestamps = np.array([p[0] for p in pivot_prices])

    tol = float(price.iloc[-1]) * tolerance_frac
    clusters = _cluster_levels(levels, tol)

    sr_levels: List[SRLevel] = []
    for center, idxs in clusters:
        cluster_ts = timestamps[idxs]
        cluster_kinds = kinds[idxs]
        touches = len(idxs)

        # Recency weight: newer touches count more
        # weight = sum(weight_recency ** days_ago)
        last = max(cluster_ts)
        days_ago = (df.index[-1] - cluster_ts).astype('timedelta64[D]').astype(int)
        recency_weight = (weight_recency ** days_ago).sum()

        # Mixed if both highs and lows landed near the same area
        unique_kinds = set(cluster_kinds.tolist())
        if len(unique_kinds) > 1:
            kind: LevelType = "mixed"
        else:
            kind = list(unique_kinds)[0]  # type: ignore[index]

        # Strength = touches * recency_weight (normalized later)
        raw_strength = touches + 0.5 * recency_weight
        sr_levels.append(SRLevel(level=float(center), strength=float(raw_strength), last_touch=last, kind=kind, touches=touches))

    # Normalize strength to 0..1 and return strongest first
    if sr_levels:
        mx = max(l.strength for l in sr_levels)
        for l in sr_levels:
            l.strength = float(l.strength / mx)
        sr_levels.sort(key=lambda l: (l.strength, l.last_touch), reverse=True)
        sr_levels = sr_levels[:max_levels]

    return sr_levels


# -----------------------------
# Volume spike detection
# -----------------------------

def detect_volume_spikes(
    df: pd.DataFrame,
    lookback: int = 60,
    z_threshold: float = 2.0,
    min_volume: Optional[int] = None,
) -> pd.DataFrame:
    """
    Flag dates where volume is unusually high relative to a rolling baseline.

    Returns a DataFrame with columns:
        ['Volume', 'vol_mean', 'vol_std', 'zscore', 'is_spike']
    """
    df = _ensure_datetime_index(df)
    v = df['Volume'].astype(float)
    vol_mean = v.rolling(lookback, min_periods=10).mean()
    vol_std = v.rolling(lookback, min_periods=10).std(ddof=0)
    zscore = (v - vol_mean) / (vol_std.replace(0, np.nan))

    spikes = pd.DataFrame({
        'Volume': v,
        'vol_mean': vol_mean,
        'vol_std': vol_std,
        'zscore': zscore,
    })
    spikes['is_spike'] = (spikes['zscore'] >= z_threshold)
    if min_volume is not None:
        spikes['is_spike'] &= (spikes['Volume'] >= min_volume)
    return spikes


# -----------------------------
# Optional plotting helpers
# -----------------------------

def plot_with_sr_and_volume(
    df: pd.DataFrame,
    sr_levels: List[SRLevel],
    price_col: str = "Close",
    show: bool = True,
):
    """Quick matplotlib plot for sanity checks.
    Avoids external deps; feel free to integrate into your existing charting.
    """
    import matplotlib.pyplot as plt

    df = _ensure_datetime_index(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[price_col].values, label=price_col)

    # draw S/R bands
    for lvl in sr_levels:
        alpha = 0.15 + 0.35 * lvl.strength  # stronger = more visible
        color = {
            'support': 'tab:green',
            'resistance': 'tab:red',
            'mixed': 'tab:purple'
        }.get(lvl.kind, 'gray')
        ax.axhline(lvl.level, linestyle='--', alpha=alpha, color=color)
        ax.text(df.index[-1], lvl.level, f" {lvl.kind[:1].upper()} {lvl.level:.2f}",
                va='center', fontsize=8, color=color)

    ax.set_title('Price with Support/Resistance Levels')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    if show:
        plt.show()
    return fig, ax


def plot_volume_spikes(
    spikes_df: pd.DataFrame,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(spikes_df.index, spikes_df['Volume'].values)

    # Mark spikes
    spike_idx = spikes_df.index[spikes_df['is_spike']]
    spike_vals = spikes_df.loc[spike_idx, 'Volume'].values
    ax.scatter(spike_idx, spike_vals, s=16)

    ax.set_title('Volume & Spikes')
    ax.grid(True, alpha=0.2)
    if show:
        plt.show()
    return fig, ax


# -----------------------------
# Convenience wrapper
# -----------------------------

def analyze_sr_and_volume(
    df: pd.DataFrame,
    price_col: str = "Close",
    pivot_window: int = 5,
    tolerance_frac: float = 0.01,
    max_levels: int = 10,
    lookback: int = 60,
    z_threshold: float = 2.0,
    min_volume: Optional[int] = None,
) -> Dict[str, object]:
    """All-in-one helper: returns S/R levels and volume spikes table.
    """
    sr = get_support_resistance(
        df,
        price_col=price_col,
        window=pivot_window,
        tolerance_frac=tolerance_frac,
        max_levels=max_levels,
    )
    spikes = detect_volume_spikes(
        df,
        lookback=lookback,
        z_threshold=z_threshold,
        min_volume=min_volume,
    )
    return {"sr_levels": sr, "volume_spikes": spikes}


# -----------------------------
# Example usage (remove if you import this in your dashboard)
# -----------------------------
if __name__ == "__main__":
    # Minimal smoketest with random-walk price + random volume
    rng = pd.date_range("2024-01-01", periods=250, freq="B")
    np.random.seed(7)
    price = 20 + np.cumsum(np.random.randn(len(rng)) * 0.5)
    high = price + np.random.rand(len(rng))
    low = price - np.random.rand(len(rng))
    open_ = price + np.random.randn(len(rng)) * 0.2
    vol = (1e6 + np.random.randn(len(rng)) * 2e5).clip(min=1e5)

    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': price, 'Volume': vol
    }, index=rng)

    results = analyze_sr_and_volume(
        df,
        pivot_window=5,
        tolerance_frac=0.01,
        max_levels=8,
        lookback=60,
        z_threshold=2.5,
    )

    print("Top S/R levels (strongest first):")
    for lvl in results['sr_levels']:
        print(lvl)

    # Uncomment to visualize if running locally
    # plot_with_sr_and_volume(df, results['sr_levels'])
    # plot_volume_spikes(results['volume_spikes'])

