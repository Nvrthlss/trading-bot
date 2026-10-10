"""
Chart plotting utilities using matplotlib
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_price_with_indicators(df: pd.DataFrame, title: str = "Price Chart"):
    """
    Plot price chart with moving averages on top panel,
    and RSI on the bottom panel.
    """

    fig, (ax_price, ax_rsi) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # === Top panel: price + MA ===
    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1.5)

    for col in df.columns:
        if col.startswith("sma_"):
            ax_price.plot(df.index, df[col], label=col.upper(), linewidth=1)
        elif col.startswith("ema_"):
            ax_price.plot(df.index, df[col], label=col.upper(), linewidth=1, linestyle="--")

    ax_price.set_title(title)
    ax_price.set_ylabel("Price (USDT)")
    ax_price.legend(loc="best")
    ax_price.grid(True, alpha=0.3)

    # === Bottom panel: RSI ===
    rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
    if rsi_cols:
        for col in rsi_cols:
            ax_rsi.plot(df.index, df[col], label=col.upper(), color="purple")

        ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_rsi.axhline(50, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc="best")
        ax_rsi.grid(True, alpha=0.3)

    ax_rsi.set_xlabel("Time")

    plt.tight_layout()
    plt.show()