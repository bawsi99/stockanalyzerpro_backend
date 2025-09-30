import matplotlib
matplotlib.use('Agg')  # Ensure headless-safe backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.dates as mdates
from typing import Dict, Any
# Local import to avoid circular dependency

class PatternVisualizer:
    """
    Central class for all pattern visualization logic (plotting peaks/lows, divergences, double tops/bottoms, triangles, flags, support/resistance, volume anomalies).
    """
    @staticmethod
    def plot_price_with_peaks_lows(prices: pd.Series, peaks: np.ndarray, lows: np.ndarray, title: str = "Price with Peaks and Lows"):
        plt.figure(figsize=(14, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        plt.scatter(prices.index[peaks], prices.iloc[peaks], color='red', label='Peaks', marker='^')
        plt.scatter(prices.index[lows], prices.iloc[lows], color='green', label='Lows', marker='v')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("chart_peaks_lows.png")

    @staticmethod
    def plot_divergences(prices: pd.Series, indicator: pd.Series, divergences: List[Tuple[int, int, str]], title: str = "Divergences"):
        plt.figure(figsize=(16, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        plt.plot(indicator.index, indicator, label='Indicator', color='orange', alpha=0.7)
        shown_labels = set()
        for idx1, idx2, dtype in divergences:
            color = 'red' if dtype == 'bearish' else 'green'
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color=color, linestyle='--', linewidth=2, alpha=0.8)
            plt.plot([indicator.index[idx1], indicator.index[idx2]], [indicator.iloc[idx1], indicator.iloc[idx2]], color=color, linestyle=':', linewidth=2, alpha=0.5)
            plt.annotate('\u2b07' if dtype == 'bearish' else '\u2b06', xy=(prices.index[idx2], prices.iloc[idx2]), xytext=(0, 10), textcoords="offset points", ha='center', fontsize=12, color=color)
            if dtype not in shown_labels:
                plt.plot([], [], color=color, linestyle='--', label=f'{dtype.capitalize()} Divergence')
                shown_labels.add(dtype)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("chart_divergences.png")

    @staticmethod
    def plot_double_tops_bottoms(prices: pd.Series, double_tops: list, double_bottoms: list, title: str = "Double Tops/Bottoms"):
        plt.figure(figsize=(14, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        for idx1, idx2 in double_tops:
            plt.scatter([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='red', marker='^', s=100, label='Double Top' if idx1 == double_tops[0][0] else "")
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='red', linestyle='--', alpha=0.7)
        for idx1, idx2 in double_bottoms:
            plt.scatter([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='green', marker='v', s=100, label='Double Bottom' if idx1 == double_bottoms[0][0] else "")
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='green', linestyle='--', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("chart_double_tops_bottoms.png")
        plt.close()

    @staticmethod
    def plot_triangles_flags(prices: pd.Series, triangles: list[list[int]], flags: list[list[int]], title: str = "Triangles & Flags", save_as: str | None = "chart_triangles_flags.png"):
        """
        Parameters
        ----------
        prices     : pd.Series
            Price series (index can be datetime or numeric).
        triangles  : list[list[int]]
            Each inner list contains the indices (0-based) that belong to ONE triangle.
            The first and last index of the list are used as the two anchor points.
        flags      : list[list[int]]
            Same idea as triangles but plotted as a price channel (flag/pennant).
        title      : str
            Title of the chart.
        save_as    : str | None
            File name to save the figure.  Pass None to skip saving.
        """

        # ------------------------------------------------------------------ #
        # 1.  Basic figure and price line
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(prices.index, prices.values,
                color="steelblue", lw=1.3, alpha=.8, label="Price")

        # Legend helpers
        triangle_label_used = False
        flag_label_used = False

        # ------------------------------------------------------------------ #
        # 2.  Triangles
        # ------------------------------------------------------------------ #
        for tri in triangles:
            if len(tri) < 2:
                continue          # nothing to draw

            start, end = tri[0], tri[-1]
            seg = prices.iloc[start:end + 1]

            # Upper / lower reference points
            high_idx = seg.idxmax()
            low_idx  = seg.idxmin()

            # Build x (dates) and y (prices) pairs
            x_pair = [prices.index[start], prices.index[end]]
            y_upper = [prices.loc[high_idx], prices.iloc[end]]
            y_lower = [prices.loc[low_idx],  prices.iloc[end]]

            # Plot converging trend-lines
            ax.plot(x_pair, y_upper, color="green",  lw=2,
                    label="Triangle" if not triangle_label_used else None)
            ax.plot(x_pair, y_lower, color="red",    lw=2,
                    label=None)

            triangle_label_used = True

            # Optional shading (same length as x_pair)
            ax.fill_between(x_pair, y_lower, y_upper,
                            color="gold", alpha=.20, linewidth=0)

        # ------------------------------------------------------------------ #
        # 3.  Flags / channels
        # ------------------------------------------------------------------ #
        for flg in flags:
            if len(flg) < 2:
                continue

            start, end = flg[0], flg[-1]
            seg = prices.iloc[start:end + 1]

            # Fit a simple linear regression to the segment (channel mid-line)
            x_rel = np.arange(len(seg))
            m, b = np.polyfit(x_rel, seg.values, 1)
            mid_line = m * x_rel + b

            # Channel width – use one standard deviation of the segment
            spread = seg.std(ddof=0)
            upper = mid_line + spread
            lower = mid_line - spread

            # Plot channel
            ax.plot(seg.index, upper, color="purple", ls="--", lw=1.5,
                    label="Flag" if not flag_label_used else None)
            ax.plot(seg.index, lower, color="purple", ls="--", lw=1.5)

            # Shade channel
            ax.fill_between(seg.index, lower, upper,
                            color="plum", alpha=.15, linewidth=0)

            flag_label_used = True

        # ------------------------------------------------------------------ #
        # 4.  Cosmetics
        # ------------------------------------------------------------------ #
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=.3)
        ax.legend(loc="best")

        fig.tight_layout()

        if save_as:
            fig.savefig(save_as, dpi=300)

        plt.close(fig)    # keep notebooks tidy
        return fig

    @staticmethod
    def plot_support_resistance(prices: pd.Series, support: list, resistance: list, title: str = "Support & Resistance"):
        plt.figure(figsize=(14, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        for level in support:
            plt.axhline(y=level, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Support' if support.index(level) == 0 else "")
        for level in resistance:
            plt.axhline(y=level, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Resistance' if resistance.index(level) == 0 else "")
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("chart_support_resistance.png")
        plt.close()


# Helper for candlestick plotting
def plot_candlesticks(ax, data, width=0.6, width2=0.1):
    up = data[data['close'] >= data['open']]
    down = data[data['close'] < data['open']]
    ax.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green')
    ax.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green')
    ax.bar(up.index, up['low'] - up['open'], width2, bottom=up['open'], color='green')
    ax.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red')
    ax.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red')
    ax.bar(down.index, down['low'] - down['close'], width2, bottom=down['close'], color='red')
    return ax

def plot_volume_bars(ax, data, width=0.8):
    for i in range(len(data)):
        if i > 0 and data['close'].iloc[i] >= data['close'].iloc[i-1]:
            ax.bar(data.index[i], data['volume'].iloc[i], width=width, color='green', alpha=0.5)
        else:
            ax.bar(data.index[i], data['volume'].iloc[i], width=width, color='red', alpha=0.5)
    return ax

# Centralized constants
BOLLINGER_BAND_NARROW = 0.1
BOLLINGER_BAND_WIDE = 0.3
VOLUME_HIGH_RATIO = 1.5
VOLUME_LOW_RATIO = 0.5
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_STRONG = 25

class ChartVisualizer:
    """
    Centralized class for creating all technical analysis and pattern charts.
    """
    @staticmethod
    def plot_comparison_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str, stock_symbol: str = 'Stock'):
        """
        Create and save a multi-panel technical analysis comparison chart.
        """
        fig = plt.figure(figsize=(12, 20))
        gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        up = data[data['close'] >= data['open']]
        down = data[data['close'] < data['open']]
        width = 0.6
        width2 = 0.1
        ax1.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green')
        ax1.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green')
        ax1.bar(up.index, up['low'] - up['open'], width2, bottom=up['open'], color='green')
        ax1.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red')
        ax1.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red')
        ax1.bar(down.index, down['low'] - down['close'], width2, bottom=down['close'], color='red')
        from ml.indicators.technical_indicators import TechnicalIndicators
        sma_20 = TechnicalIndicators.calculate_sma(data, 'close', 20)
        sma_50 = TechnicalIndicators.calculate_sma(data, 'close', 50)
        sma_200 = TechnicalIndicators.calculate_sma(data, 'close', 200)
        ax1.plot(data.index, sma_20, label='SMA 20', color='blue', linewidth=1)
        ax1.plot(data.index, sma_50, label='SMA 50', color='orange', linewidth=1)
        ax1.plot(data.index, sma_200, label='SMA 200', color='purple', linewidth=1)
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(data)
        ax1.plot(data.index, upper_band, 'r--', label='Upper BB', linewidth=1, alpha=0.7)
        ax1.plot(data.index, lower_band, 'r--', label='Lower BB', linewidth=1, alpha=0.7)
        ax1.fill_between(data.index, upper_band, lower_band, color='gray', alpha=0.05)
        ax1.set_title(f'{stock_symbol} Technical Analysis Comparison')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        for i in range(len(data)):
            if i > 0 and data['close'].iloc[i] >= data['close'].iloc[i-1]:
                ax2.bar(data.index[i], data['volume'].iloc[i], width=0.8, color='green', alpha=0.5)
            else:
                ax2.bar(data.index[i], data['volume'].iloc[i], width=0.8, color='red', alpha=0.5)
        volume_ma = data['volume'].rolling(window=20).mean()
        ax2.plot(data.index, volume_ma, color='blue', linewidth=1, label='Volume MA (20)')
        if 'volume' in indicators and 'obv' in indicators['volume']:
            obv = TechnicalIndicators.calculate_obv(data)
            # Handle division by zero or NaN
            if pd.isna(obv.mean()) or obv.mean() == 0:
                obv_normalized = obv  # Use original OBV if mean is zero/NaN
            else:
                obv_normalized = obv * (data['volume'].mean() / obv.mean())
            ax2.plot(data.index, obv_normalized, color='purple', linewidth=1, label='OBV (normalized)')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(data)
        ax3.plot(data.index, macd_line, label='MACD (12, 26)', color='blue', linewidth=1)
        ax3.plot(data.index, signal_line, label='Signal (9)', color='red', linewidth=1)
        for i in range(len(data)):
            if i < len(histogram):
                if histogram.iloc[i] >= 0:
                    ax3.bar(data.index[i], histogram.iloc[i], width=0.8, color='green', alpha=0.5)
                else:
                    ax3.bar(data.index[i], histogram.iloc[i], width=0.8, color='red', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        rsi = TechnicalIndicators.calculate_rsi(data)
        ax4.plot(data.index, rsi, label='RSI (14)', color='blue', linewidth=1)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax4.axhline(y=50, color='k', linestyle='--', alpha=0.2)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel('RSI')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
        stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic_oscillator(data)
        ax5.plot(data.index, stoch_k, label='%K', color='blue', linewidth=1)
        ax5.plot(data.index, stoch_d, label='%D', color='red', linewidth=1)
        ax5.axhline(y=80, color='r', linestyle='--', alpha=0.5)
        ax5.axhline(y=20, color='g', linestyle='--', alpha=0.5)
        ax5.axhline(y=50, color='k', linestyle='--', alpha=0.2)
        ax5.set_ylim(0, 100)
        ax5.set_ylabel('Stochastic')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper left')
        ax6 = fig.add_subplot(gs[5, 0], sharex=ax1)
        adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(data)
        ax6.plot(data.index, adx, label='ADX', color='black', linewidth=1)
        ax6.plot(data.index, plus_di, label='+DI', color='green', linewidth=1)
        ax6.plot(data.index, minus_di, label='-DI', color='red', linewidth=1)
        ax6.axhline(y=25, color='k', linestyle='--', alpha=0.3)
        ax6.set_ylim(0, 60)
        ax6.set_ylabel('ADX')
        ax6.set_xlabel('Date')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper left')
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_divergence_chart(prices: pd.Series, indicator: pd.Series, divergences: List, save_path: str, title: str = 'Divergences'):
        """
        Create and save a chart annotated with price/indicator divergences.
        """
        plt.figure(figsize=(16, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        plt.plot(indicator.index, indicator, label='Indicator', color='orange', alpha=0.7)
        shown_labels = set()
        for idx1, idx2, dtype in divergences:
            color = 'red' if dtype == 'bearish' else 'green'
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color=color, linestyle='--', linewidth=2, alpha=0.8)
            plt.plot([indicator.index[idx1], indicator.index[idx2]], [indicator.iloc[idx1], indicator.iloc[idx2]], color=color, linestyle=':', linewidth=2, alpha=0.5)
            plt.annotate('\u2b07' if dtype == 'bearish' else '\u2b06', xy=(prices.index[idx2], prices.iloc[idx2]), xytext=(0, 10), textcoords="offset points", ha='center', fontsize=12, color=color)
            if dtype not in shown_labels:
                plt.plot([], [], color=color, linestyle='--', label=f'{dtype.capitalize()} Divergence')
                shown_labels.add(dtype)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_double_tops_bottoms_chart(prices: pd.Series, double_tops: List, double_bottoms: List, save_path: str, title: str = 'Double Tops/Bottoms'):
        """
        Create and save a chart annotated with double tops and double bottoms.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        for idx1, idx2 in double_tops:
            plt.scatter([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='red', marker='^', s=100, label='Double Top' if idx1 == double_tops[0][0] else "")
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='red', linestyle='--', alpha=0.7)
        for idx1, idx2 in double_bottoms:
            plt.scatter([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='green', marker='v', s=100, label='Double Bottom' if idx1 == double_bottoms[0][0] else "")
            plt.plot([prices.index[idx1], prices.index[idx2]], [prices.iloc[idx1], prices.iloc[idx2]], color='green', linestyle='--', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_support_resistance_chart(prices: pd.Series, support: List, resistance: List, save_path: str, title: str = 'Support & Resistance'):
        """
        Create and save a chart annotated with support and resistance levels.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(prices.index, prices, label='Price', color='blue')
        for level in support:
            plt.axhline(y=level, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Support' if support.index(level) == 0 else "")
        for level in resistance:
            plt.axhline(y=level, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Resistance' if resistance.index(level) == 0 else "")
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_triangles_flags_chart(prices: pd.Series, triangles: List, flags: List, save_path: str, title: str = 'Triangles & Flags'):
        """
        Create and save a chart annotated with triangles and flags.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(prices.index, prices.values, color="steelblue", lw=1.3, alpha=.8, label="Price")
        triangle_label_used = False
        flag_label_used = False
        for tri in triangles:
            if len(tri) < 2:
                continue
            start, end = tri[0], tri[-1]
            seg = prices.iloc[start:end + 1]
            high_idx = seg.idxmax()
            low_idx  = seg.idxmin()
            x_pair = [prices.index[start], prices.index[end]]
            y_upper = [prices.loc[high_idx], prices.iloc[end]]
            y_lower = [prices.loc[low_idx],  prices.iloc[end]]
            ax.plot(x_pair, y_upper, color="green",  lw=2, label="Triangle" if not triangle_label_used else None)
            ax.plot(x_pair, y_lower, color="red",    lw=2, label=None)
            triangle_label_used = True
            ax.fill_between(x_pair, y_lower, y_upper, color="gold", alpha=.20, linewidth=0)
        for flg in flags:
            if len(flg) < 2:
                continue
            start, end = flg[0], flg[-1]
            seg = prices.iloc[start:end + 1]
            x_rel = np.arange(len(seg))
            m, b = np.polyfit(x_rel, seg.values, 1)
            mid_line = m * x_rel + b
            spread = seg.std(ddof=0)
            upper = mid_line + spread
            lower = mid_line - spread
            ax.plot(seg.index, upper, color="purple", ls="--", lw=1.5, label="Flag" if not flag_label_used else None)
            ax.plot(seg.index, lower, color="purple", ls="--", lw=1.5)
            ax.fill_between(seg.index, lower, upper, color="plum", alpha=.15, linewidth=0)
            flag_label_used = True
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=.3)
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_volume_anomalies_chart(volume: pd.Series, anomalies: List, save_path: str, title: str = 'Volume Anomalies'):
        """
        Plot volume anomalies with enhanced price context.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
        
        # Plot volume
        ax1.plot(volume.index, volume, label='Volume', color='blue', alpha=0.7)
        ax1.set_ylabel('Volume')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Mark anomalies
        if anomalies:
            for i, anomaly in enumerate(anomalies):
                if isinstance(anomaly, dict) and 'date' in anomaly:
                    # New format with date and volume info
                    date = pd.to_datetime(anomaly['date'])
                    vol = anomaly.get('volume', volume.get(date, 0))
                    ax1.scatter(date, vol, color='red', marker='o', s=100, alpha=0.8, label='Anomaly' if i == 0 else "")
                    ax1.annotate(f"Anomaly\n{anomaly.get('volume_ratio', 0):.1f}x", 
                               xy=(date, vol), xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                elif isinstance(anomaly, pd.Timestamp):
                    # Handle datetime indices directly
                    vol = volume.get(anomaly, 0)
                    ax1.scatter(anomaly, vol, color='red', marker='o', s=100, alpha=0.8, label='Anomaly' if i == 0 else "")
                    ax1.annotate(f"Anomaly", 
                               xy=(anomaly, vol), xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                else:
                    # Handle integer indices (legacy format)
                    try:
                        ax1.scatter(volume.index[anomaly], volume.iloc[anomaly], color='red', marker='o', s=80, label='Anomaly' if i == 0 else "")
                    except (IndexError, TypeError):
                        # Skip invalid indices
                        continue
        
        # Plot volume moving average
        volume_ma = volume.rolling(window=20).mean()
        ax1.plot(volume.index, volume_ma, label='20-day MA', color='orange', alpha=0.8)
        
        ax1.legend()
        
        # Plot volume ratio (current volume / moving average)
        volume_ratio = volume / volume_ma
        ax2.plot(volume.index, volume_ratio, label='Volume Ratio', color='purple', alpha=0.7)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Normal Volume')
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='2x Volume')
        ax2.set_ylabel('Volume Ratio')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_price_volume_correlation(data: pd.DataFrame, anomalies: List, save_path: str, title: str = 'Price-Volume Correlation'):
        """
        Plot price and volume correlation with anomaly markers.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 1, 1])
        
        # Plot price
        ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1.5)
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Mark anomalies on price chart
        if anomalies:
            for anomaly in anomalies:
                if isinstance(anomaly, dict) and 'date' in anomaly:
                    date = pd.to_datetime(anomaly['date'])
                    price = anomaly.get('price', data['close'].get(date, 0))
                    ax1.scatter(date, price, color='red', marker='o', s=100, alpha=0.8, label='Volume Anomaly' if anomaly == anomalies[0] else "")
                    ax1.annotate(f"Vol: {anomaly.get('volume_ratio', 0):.1f}x\nPrice: {price:.2f}", 
                               xy=(date, price), xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.legend()
        
        # Plot volume
        ax2.plot(data.index, data['volume'], label='Volume', color='green', alpha=0.7)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Mark anomalies on volume chart
        if anomalies:
            for anomaly in anomalies:
                if isinstance(anomaly, dict) and 'date' in anomaly:
                    date = pd.to_datetime(anomaly['date'])
                    vol = anomaly.get('volume', data['volume'].get(date, 0))
                    ax2.scatter(date, vol, color='red', marker='o', s=100, alpha=0.8)
        
        ax2.legend()
        
        # Plot correlation
        # Calculate rolling correlation between price changes and volume changes
        price_changes = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        
        # Rolling correlation (20-day window)
        correlation = price_changes.rolling(window=20).corr(volume_changes)
        ax3.plot(data.index, correlation, label='Price-Volume Correlation (20d)', color='purple', alpha=0.8)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Correlation')
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Positive Correlation')
        ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Negative Correlation')
        ax3.set_ylabel('Correlation')
        ax3.set_xlabel('Date')
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_candlestick_with_volume(data: pd.DataFrame, anomalies: List, save_path: str, title: str = 'Price & Volume Analysis'):
        """
        Plot candlestick chart with volume bars and anomaly markers.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        # Plot candlesticks
        up = data[data['close'] >= data['open']]
        down = data[data['close'] < data['open']]
        width = 0.6
        width2 = 0.1
        
        # Up candlesticks
        ax1.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.8)
        ax1.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green')
        ax1.bar(up.index, up['low'] - up['open'], width2, bottom=up['open'], color='green')
        
        # Down candlesticks
        ax1.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red', alpha=0.8)
        ax1.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red')
        ax1.bar(down.index, down['low'] - down['close'], width2, bottom=down['close'], color='red')
        
        # Mark anomalies on price chart
        if anomalies:
            for anomaly in anomalies:
                if isinstance(anomaly, dict) and 'date' in anomaly:
                    date = pd.to_datetime(anomaly['date'])
                    price = anomaly.get('price', data['close'].get(date, 0))
                    ax1.scatter(date, price, color='orange', marker='*', s=200, alpha=0.9, label='Volume Anomaly' if anomaly == anomalies[0] else "")
                    ax1.annotate(f"Vol: {anomaly.get('volume_ratio', 0):.1f}x", 
                               xy=(date, price), xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot volume bars
        colors = ['green' if close >= open else 'red' for close, open in zip(data['close'], data['open'])]
        ax2.bar(data.index, data['volume'], color=colors, alpha=0.7, width=0.8)
        
        # Mark anomalies on volume chart
        if anomalies:
            for anomaly in anomalies:
                if isinstance(anomaly, dict) and 'date' in anomaly:
                    date = pd.to_datetime(anomaly['date'])
                    vol = anomaly.get('volume', data['volume'].get(date, 0))
                    ax2.scatter(date, vol, color='orange', marker='*', s=100, alpha=0.9)
        
        # Add volume moving average
        volume_ma = data['volume'].rolling(window=20).mean()
        ax2.plot(data.index, volume_ma, label='20-day Volume MA', color='blue', alpha=0.8)
        
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 

    @staticmethod
    def plot_head_and_shoulders_pattern(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Head and Shoulders patterns.
        
        Args:
            data: Price data
            patterns: List of H&S patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        
        for i, pattern in enumerate(patterns[:5]):  # Limit to 5 patterns for clarity
            color = colors[i % len(colors)]
            
            # Plot shoulders and head
            left_shoulder = pattern['left_shoulder']
            head = pattern['head']
            right_shoulder = pattern['right_shoulder']
            neckline = pattern['neckline']
            
            # Plot key points
            ax.scatter(left_shoulder['index'], left_shoulder['price'], 
                      color=color, s=100, marker='o', alpha=0.8, label=f'Left Shoulder {i+1}')
            ax.scatter(head['index'], head['price'], 
                      color=color, s=150, marker='^', alpha=0.8, label=f'Head {i+1}')
            ax.scatter(right_shoulder['index'], right_shoulder['price'], 
                      color=color, s=100, marker='o', alpha=0.8, label=f'Right Shoulder {i+1}')
            
            # Plot neckline
            neckline_start = left_shoulder['index']
            neckline_end = right_shoulder['index']
            ax.plot([neckline_start, neckline_end], [neckline['level'], neckline['level']], 
                   color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'Neckline {i+1}')
            
            # Add pattern info
            ax.annotate(f'H&S {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=(head['index'], head['price']), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Head and Shoulders Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_inverse_head_and_shoulders_pattern(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Inverse Head and Shoulders patterns.
        
        Args:
            data: Price data
            patterns: List of inverse H&S patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['green', 'lime', 'teal', 'olive', 'darkgreen']
        
        for i, pattern in enumerate(patterns[:5]):  # Limit to 5 patterns for clarity
            color = colors[i % len(colors)]
            
            # Plot shoulders and head
            left_shoulder = pattern['left_shoulder']
            head = pattern['head']
            right_shoulder = pattern['right_shoulder']
            neckline = pattern['neckline']
            
            # Plot key points
            ax.scatter(left_shoulder['index'], left_shoulder['price'], 
                      color=color, s=100, marker='o', alpha=0.8, label=f'Left Shoulder {i+1}')
            ax.scatter(head['index'], head['price'], 
                      color=color, s=150, marker='v', alpha=0.8, label=f'Head {i+1}')
            ax.scatter(right_shoulder['index'], right_shoulder['price'], 
                      color=color, s=100, marker='o', alpha=0.8, label=f'Right Shoulder {i+1}')
            
            # Plot neckline
            neckline_start = left_shoulder['index']
            neckline_end = right_shoulder['index']
            ax.plot([neckline_start, neckline_end], [neckline['level'], neckline['level']], 
                   color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'Neckline {i+1}')
            
            # Add pattern info
            ax.annotate(f'Inv H&S {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=(head['index'], head['price']), 
                       xytext=(10, -10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Inverse Head and Shoulders Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_cup_and_handle_pattern(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Cup and Handle patterns.
        
        Args:
            data: Price data
            patterns: List of Cup and Handle patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['blue', 'cyan', 'navy', 'skyblue', 'royalblue']
        
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for clarity
            color = colors[i % len(colors)]
            
            cup = pattern['cup']
            handle = pattern['handle']
            
            # Plot cup
            cup_start = cup['start_index']
            cup_end = cup['end_index']
            cup_data = data.iloc[cup_start:cup_end + 1]
            ax.plot(cup_data.index, cup_data['close'], color=color, linewidth=3, alpha=0.8, label=f'Cup {i+1}')
            
            # Plot handle
            handle_start = handle['start_index']
            handle_end = handle['end_index']
            handle_data = data.iloc[handle_start:handle_end + 1]
            ax.plot(handle_data.index, handle_data['close'], color=color, linewidth=2, alpha=0.8, label=f'Handle {i+1}')
            
            # Plot breakout level
            ax.axhline(y=pattern['breakout_level'], color=color, linestyle='--', alpha=0.6, label=f'Breakout {i+1}')
            
            # Add pattern info
            ax.annotate(f'C&H {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=(handle_end, handle['end_price']), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Cup and Handle Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_multi_timeframe_analysis(data: pd.DataFrame, mtf_analysis: Dict, save_path: str = None):
        """
        Plot multi-timeframe analysis.
        
        Args:
            data: Price data
            mtf_analysis: Multi-timeframe analysis results
            save_path: Path to save the plot
        """
        if not mtf_analysis or "error" in mtf_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Multi-Timeframe Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Short-term analysis
        ax1 = axes[0, 0]
        short_term = mtf_analysis.get("short_term", {})
        if short_term:
            periods = list(short_term.keys())
            bullish_periods = [short_term[p].get("bullish", 0) for p in periods]
            bearish_periods = [short_term[p].get("bearish", 0) for p in periods]
            
            x = np.arange(len(periods))
            width = 0.35
            
            ax1.bar(x - width/2, bullish_periods, width, label='Bullish', color='green', alpha=0.7)
            ax1.bar(x + width/2, bearish_periods, width, label='Bearish', color='red', alpha=0.7)
            
            ax1.set_xlabel('Time Periods')
            ax1.set_ylabel('Signal Strength')
            ax1.set_title(f'Short-term Analysis\nAI Confidence: {short_term.get("ai_confidence", 0):.0f}%')
            ax1.set_xticks(x)
            ax1.set_xticklabels(periods, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add AI analysis info
            ax1.text(0.02, 0.98, f'AI Trend: {short_term.get("ai_trend", "Unknown")}\nConfidence: {short_term.get("ai_confidence", 0):.0f}%',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Medium-term analysis
        ax2 = axes[0, 1]
        medium_term = mtf_analysis.get("medium_term", {})
        if medium_term:
            periods = list(medium_term.keys())
            bullish_periods = [medium_term[p].get("bullish", 0) for p in periods]
            bearish_periods = [medium_term[p].get("bearish", 0) for p in periods]
            
            x = np.arange(len(periods))
            width = 0.35
            
            ax2.bar(x - width/2, bullish_periods, width, label='Bullish', color='green', alpha=0.7)
            ax2.bar(x + width/2, bearish_periods, width, label='Bearish', color='red', alpha=0.7)
            
            ax2.set_xlabel('Time Periods')
            ax2.set_ylabel('Signal Strength')
            ax2.set_title(f'Medium-term Analysis\nAI Confidence: {medium_term.get("ai_confidence", 0):.0f}%')
            ax2.set_xticks(x)
            ax2.set_xticklabels(periods, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add AI analysis info
            ax2.text(0.02, 0.98, f'AI Trend: {medium_term.get("ai_trend", "Unknown")}\nConfidence: {medium_term.get("ai_confidence", 0):.0f}%',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Long-term analysis
        ax3 = axes[1, 0]
        long_term = mtf_analysis.get("long_term", {})
        if long_term:
            periods = list(long_term.keys())
            bullish_periods = [long_term[p].get("bullish", 0) for p in periods]
            bearish_periods = [long_term[p].get("bearish", 0) for p in periods]
            
            x = np.arange(len(periods))
            width = 0.35
            
            ax3.bar(x - width/2, bullish_periods, width, label='Bullish', color='green', alpha=0.7)
            ax3.bar(x + width/2, bearish_periods, width, label='Bearish', color='red', alpha=0.7)
            
            ax3.set_xlabel('Time Periods')
            ax3.set_ylabel('Signal Strength')
            ax3.set_title(f'Long-term Analysis\nAI Confidence: {long_term.get("ai_confidence", 0):.0f}%')
            ax3.set_xticks(x)
            ax3.set_xticklabels(periods, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add AI analysis info
            ax3.text(0.02, 0.98, f'AI Trend: {long_term.get("ai_trend", "Unknown")}\nConfidence: {long_term.get("ai_confidence", 0):.0f}%',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 4: Overall AI analysis
        ax4 = axes[1, 1]
        
        if "overall_ai_analysis" in mtf_analysis:
            overall = mtf_analysis["overall_ai_analysis"]
            
            # Create a summary chart
            categories = ['Short-term', 'Medium-term', 'Long-term']
            ai_confidence = [
                mtf_analysis.get("short_term", {}).get("ai_confidence", 0),
                mtf_analysis.get("medium_term", {}).get("ai_confidence", 0),
                mtf_analysis.get("long_term", {}).get("ai_confidence", 0)
            ]
            
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            ax4.bar(categories, ai_confidence, color=colors, alpha=0.7)
            
            ax4.set_ylabel('AI Confidence (%)')
            ax4.set_title(f'Overall AI Analysis\nPrimary Trend: {overall.get("primary_trend", "Unknown")}')
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(ai_confidence):
                ax4.text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom')
            
            # Add overall analysis info
            ax4.text(0.02, 0.98, f'Primary Trend: {overall.get("primary_trend", "Unknown")}\nConfidence: {overall.get("confidence", 0):.0f}%',
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 

    @staticmethod
    def plot_triple_top_pattern(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Triple Top patterns.
        
        Args:
            data: Price data
            patterns: List of Triple Top patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['red', 'darkred', 'crimson', 'firebrick', 'indianred']
        
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for clarity
            color = colors[i % len(colors)]
            
            peaks = pattern['peaks']
            valleys = pattern['valleys']
            support_level = pattern['support_level']
            
            # Plot peaks
            for j, peak in enumerate(peaks):
                ax.scatter(peak['index'], peak['price'], 
                          color=color, s=100, marker='v', alpha=0.8, 
                          label=f'Peak {j+1} {i+1}' if j == 0 else "")
            
            # Plot support level
            start_idx = peaks[0]['index']
            end_idx = peaks[-1]['index']
            ax.plot([start_idx, end_idx], [support_level, support_level], 
                   color=color, linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Support {i+1}' if i == 0 else "")
            
            # Add pattern info
            ax.annotate(f'Triple Top {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=(peaks[1]['index'], peaks[1]['price']), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Triple Top Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_triple_bottom_pattern(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Triple Bottom patterns.
        
        Args:
            data: Price data
            patterns: List of Triple Bottom patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['green', 'darkgreen', 'forestgreen', 'limegreen', 'seagreen']
        
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for clarity
            color = colors[i % len(colors)]
            
            lows = pattern['lows']
            peaks = pattern['peaks']
            resistance_level = pattern['resistance_level']
            
            # Plot lows
            for j, low in enumerate(lows):
                ax.scatter(low['index'], low['price'], 
                          color=color, s=100, marker='^', alpha=0.8, 
                          label=f'Low {j+1} {i+1}' if j == 0 else "")
            
            # Plot resistance level
            start_idx = lows[0]['index']
            end_idx = lows[-1]['index']
            ax.plot([start_idx, end_idx], [resistance_level, resistance_level], 
                   color=color, linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Resistance {i+1}' if i == 0 else "")
            
            # Add pattern info
            ax.annotate(f'Triple Bottom {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=(lows[1]['index'], lows[1]['price']), 
                       xytext=(10, -10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Triple Bottom Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_wedge_patterns(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Wedge patterns.
        
        Args:
            data: Price data
            patterns: List of Wedge patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['purple', 'orange', 'brown', 'pink', 'olive']
        
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for clarity
            color = colors[i % len(colors)]
            
            start_idx = pattern['start_index']
            end_idx = pattern['end_index']
            wedge_type = pattern['type']
            slope_highs = pattern['slope_highs']
            slope_lows = pattern['slope_lows']
            swing_points = pattern['swing_points']
            
            # Plot swing points
            highs = swing_points['highs']
            lows = swing_points['lows']
            
            for high in highs:
                ax.scatter(high['index'], high['price'], 
                          color=color, s=80, marker='v', alpha=0.8)
            
            for low in lows:
                ax.scatter(low['index'], low['price'], 
                          color=color, s=80, marker='^', alpha=0.8)
            
            # Plot trend lines
            x_range = np.array([start_idx, end_idx])
            
            # Upper line (highs)
            if len(highs) >= 2:
                high_x = np.array([h['index'] for h in highs])
                high_y = np.array([h['price'] for h in highs])
                high_slope, high_intercept = np.polyfit(high_x, high_y, 1)
                upper_line = high_slope * x_range + high_intercept
                ax.plot(x_range, upper_line, color=color, linestyle='--', linewidth=2, alpha=0.8)
            
            # Lower line (lows)
            if len(lows) >= 2:
                low_x = np.array([l['index'] for l in lows])
                low_y = np.array([l['price'] for l in lows])
                low_slope, low_intercept = np.polyfit(low_x, low_y, 1)
                lower_line = low_slope * x_range + low_intercept
                ax.plot(x_range, lower_line, color=color, linestyle='--', linewidth=2, alpha=0.8)
            
            # Add pattern info
            ax.annotate(f'{wedge_type.replace("_", " ").title()} {i+1}\nQ:{pattern["quality_score"]:.0f}', 
                       xy=((start_idx + end_idx) / 2, data['close'].iloc[start_idx:end_idx].mean()), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Wedge Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_channel_patterns(data: pd.DataFrame, patterns: List[Dict], save_path: str = None):
        """
        Plot Channel patterns.
        
        Args:
            data: Price data
            patterns: List of Channel patterns
            save_path: Path to save the plot
        """
        if not patterns:
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price data
        ax.plot(data.index, data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        colors = ['teal', 'gold', 'coral', 'slateblue', 'chocolate']
        
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for clarity
            color = colors[i % len(colors)]
            
            start_idx = pattern['start_index']
            end_idx = pattern['end_index']
            channel_type = pattern['type']
            slope_highs = pattern['slope_highs']
            slope_lows = pattern['slope_lows']
            swing_points = pattern['swing_points']
            
            # Plot swing points
            highs = swing_points['highs']
            lows = swing_points['lows']
            
            for high in highs:
                ax.scatter(high['index'], high['price'], 
                          color=color, s=80, marker='v', alpha=0.8)
            
            for low in lows:
                ax.scatter(low['index'], low['price'], 
                          color=color, s=80, marker='^', alpha=0.8)
            
            # Plot channel lines
            x_range = np.array([start_idx, end_idx])
            
            # Upper line (highs)
            if len(highs) >= 2:
                high_x = np.array([h['index'] for h in highs])
                high_y = np.array([h['price'] for h in highs])
                high_slope, high_intercept = np.polyfit(high_x, high_y, 1)
                upper_line = high_slope * x_range + high_intercept
                ax.plot(x_range, upper_line, color=color, linestyle='-', linewidth=2, alpha=0.8)
            
            # Lower line (lows)
            if len(lows) >= 2:
                low_x = np.array([l['index'] for l in lows])
                low_y = np.array([l['price'] for l in lows])
                low_slope, low_intercept = np.polyfit(low_x, low_y, 1)
                lower_line = low_slope * x_range + low_intercept
                ax.plot(x_range, lower_line, color=color, linestyle='-', linewidth=2, alpha=0.8)
            
            # Fill channel area
            if len(highs) >= 2 and len(lows) >= 2:
                ax.fill_between(x_range, lower_line, upper_line, 
                               color=color, alpha=0.1)
            
            # Add pattern info
            ax.annotate(f'{channel_type.replace("_", " ").title()} {i+1}\nQ:{pattern["quality_score"]:.0f}\nTouches: {pattern["touches"]}', 
                       xy=((start_idx + end_idx) / 2, data['close'].iloc[start_idx:end_idx].mean()), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=8)
        
        ax.set_title('Channel Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 

    @staticmethod
    def plot_comprehensive_technical_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str = None, stock_symbol: str = 'Stock'):
        """
        Create comprehensive technical overview chart combining price, indicators, and support/resistance.
        Returns matplotlib figure object. Optionally saves to file if save_path is provided.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Main price chart with indicators
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Compute and plot moving averages and Bollinger Bands directly from data
        from ml.indicators.technical_indicators import TechnicalIndicators
        sma_20_series = TechnicalIndicators.calculate_sma(data, 'close', 20)
        sma_50_series = TechnicalIndicators.calculate_sma(data, 'close', 50)
        sma_200_series = TechnicalIndicators.calculate_sma(data, 'close', 200)
        ema_20_series = TechnicalIndicators.calculate_ema(data, 'close', 20)
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(data)

        ax1.plot(data.index, sma_20_series, label='SMA 20', color='orange', alpha=0.7)
        ax1.plot(data.index, sma_50_series, label='SMA 50', color='red', alpha=0.7)
        ax1.plot(data.index, sma_200_series, label='SMA 200', color='purple', alpha=0.7)
        ax1.plot(data.index, upper_band, label='BB Upper', color='gray', linestyle='--', alpha=0.6)
        ax1.plot(data.index, lower_band, label='BB Lower', color='gray', linestyle='--', alpha=0.6)
        ax1.fill_between(data.index, upper_band, lower_band, alpha=0.1, color='gray')
        
        # Add support/resistance levels
        from ml.indicators.technical_indicators import TechnicalIndicators
        support, resistance = TechnicalIndicators.detect_support_resistance(data)
        for level in support:
            ax1.axhline(y=level, color='green', linestyle='--', alpha=0.6, label='Support' if level == support[0] else "")
        for level in resistance:
            ax1.axhline(y=level, color='red', linestyle='--', alpha=0.6, label='Resistance' if level == resistance[0] else "")
        
        ax1.set_title(f'{stock_symbol} - Comprehensive Technical Overview', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2 = axes[1]
        ax2.bar(data.index, data['volume'], color='lightblue', alpha=0.7, label='Volume')
        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # RSI chart
        ax3 = axes[2]
        rsi_series = TechnicalIndicators.calculate_rsi(data)
        if rsi_series is not None and len(rsi_series) == len(data.index):
            ax3.plot(data.index, rsi_series, label='RSI', color='purple', linewidth=1.5)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.6, label='Overbought')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.6, label='Oversold')
            ax3.set_ylabel('RSI')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # MACD chart
        ax4 = axes[3]
        macd_line_series, signal_line_series, histogram_series = TechnicalIndicators.calculate_macd(data)
        if macd_line_series is not None and signal_line_series is not None:
            ax4.plot(data.index, macd_line_series, label='MACD', color='blue', linewidth=1.5)
            ax4.plot(data.index, signal_line_series, label='Signal', color='red', linewidth=1.5)
            if histogram_series is not None:
                ax4.bar(data.index, histogram_series, color='gray', alpha=0.5, label='Histogram')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_ylabel('MACD')
            ax4.set_xlabel('Date')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file only if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return the figure object instead of closing it
        return fig

    @staticmethod
    def plot_comprehensive_pattern_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str = None, stock_symbol: str = 'Stock'):
        """
        Create comprehensive pattern analysis chart showing all reversal and continuation patterns.
        Returns matplotlib figure object. Optionally saves to file if save_path is provided.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Main price chart with patterns
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Detect and plot all patterns
        from patterns.recognition import PatternRecognition
        
        # Divergences
        from ml.indicators.technical_indicators import TechnicalIndicators
        rsi = TechnicalIndicators.calculate_rsi(data)
        divergences = PatternRecognition.detect_divergence(data['close'], rsi)
        for idx1, idx2, dtype in divergences:
            # Validate indices
            if (isinstance(idx1, (int, np.integer)) and isinstance(idx2, (int, np.integer)) and
                0 <= idx1 < len(data) and 0 <= idx2 < len(data)):
                color = 'red' if dtype == 'bearish' else 'green'
                ax1.plot([data.index[idx1], data.index[idx2]], [data['close'].iloc[idx1], data['close'].iloc[idx2]], 
                        color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax1.annotate('↓' if dtype == 'bearish' else '↑', 
                            xy=(data.index[idx2], data['close'].iloc[idx2]), 
                            xytext=(0, 10), textcoords="offset points", ha='center', fontsize=12, color=color)
        
        # Double tops/bottoms
        double_tops = PatternRecognition.detect_double_top(data['close'])
        double_bottoms = PatternRecognition.detect_double_bottom(data['close'])
        
        for idx1, idx2 in double_tops:
            # Validate indices
            if (isinstance(idx1, (int, np.integer)) and isinstance(idx2, (int, np.integer)) and
                0 <= idx1 < len(data) and 0 <= idx2 < len(data)):
                ax1.scatter([data.index[idx1], data.index[idx2]], [data['close'].iloc[idx1], data['close'].iloc[idx2]], 
                           color='red', marker='^', s=100, label='Double Top' if idx1 == double_tops[0][0] else "")
                ax1.plot([data.index[idx1], data.index[idx2]], [data['close'].iloc[idx1], data['close'].iloc[idx2]], 
                        color='red', linestyle='--', alpha=0.7)
        
        for idx1, idx2 in double_bottoms:
            # Validate indices
            if (isinstance(idx1, (int, np.integer)) and isinstance(idx2, (int, np.integer)) and
                0 <= idx1 < len(data) and 0 <= idx2 < len(data)):
                ax1.scatter([data.index[idx1], data.index[idx2]], [data['close'].iloc[idx1], data['close'].iloc[idx2]], 
                           color='green', marker='v', s=100, label='Double Bottom' if idx1 == double_bottoms[0][0] else "")
                ax1.plot([data.index[idx1], data.index[idx2]], [data['close'].iloc[idx1], data['close'].iloc[idx2]], 
                        color='green', linestyle='--', alpha=0.7)
        
        # Triangles and flags
        triangles = PatternRecognition.detect_triangle(data['close'])
        flags = PatternRecognition.detect_flag(data['close'])
        
        for tri in triangles:
            if len(tri) >= 2:
                # Validate indices
                if (isinstance(tri[0], (int, np.integer)) and isinstance(tri[-1], (int, np.integer)) and
                    0 <= tri[0] < len(data) and 0 <= tri[-1] < len(data)):
                    ax1.plot([data.index[tri[0]], data.index[tri[-1]]], [data['close'].iloc[tri[0]], data['close'].iloc[tri[-1]]], 
                            color='orange', linestyle='-', linewidth=2, alpha=0.8, label='Triangle' if tri == triangles[0] else "")
        
        for flag in flags:
            if len(flag) >= 2:
                # Validate indices
                if (isinstance(flag[0], (int, np.integer)) and isinstance(flag[-1], (int, np.integer)) and
                    0 <= flag[0] < len(data) and 0 <= flag[-1] < len(data)):
                    ax1.plot([data.index[flag[0]], data.index[flag[-1]]], [data['close'].iloc[flag[0]], data['close'].iloc[flag[-1]]], 
                            color='purple', linestyle='-', linewidth=2, alpha=0.8, label='Flag' if flag == flags[0] else "")
        
        ax1.set_title(f'{stock_symbol} - Comprehensive Pattern Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI for pattern confirmation
        ax2.plot(data.index, rsi, label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.6, label='Overbought')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.6, label='Oversold')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file only if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return the figure object instead of closing it
        return fig

    @staticmethod
    def plot_comprehensive_volume_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str = None, stock_symbol: str = 'Stock'):
        """
        Create comprehensive volume analysis chart showing all volume patterns and correlations.
        Returns matplotlib figure object. Optionally saves to file if save_path is provided.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        try:
            # Main price and volume chart
            ax1 = axes[0]
            ax1_twin = ax1.twinx()
            
            # Price line
            ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Volume bars
            colors = ['green' if close >= open else 'red' for close, open in zip(data['close'], data['open'])]
            ax1_twin.bar(data.index, data['volume'], color=colors, alpha=0.7, width=0.8, label='Volume')
            ax1_twin.set_ylabel('Volume', color='lightblue')
            ax1_twin.tick_params(axis='y', labelcolor='lightblue')
            
            ax1.set_title(f'{stock_symbol} - Comprehensive Volume Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Volume analysis
            ax2 = axes[1]
            volume_ma = data['volume'].rolling(window=20).mean()
            ax2.plot(data.index, data['volume'], label='Volume', color='lightblue', alpha=0.7, linewidth=1)
            ax2.plot(data.index, volume_ma, label='Volume MA 20', color='red', linewidth=1.5)
            ax2.set_ylabel('Volume')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Volume ratio analysis
            ax3 = axes[2]
            volume_ratio = data['volume'] / volume_ma
            ax3.plot(data.index, volume_ratio, label='Volume Ratio', color='green', linewidth=1.5)
            ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Average')
            ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.6, label='High Volume')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Low Volume')
            ax3.set_ylabel('Volume Ratio')
            ax3.set_xlabel('Date')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to file only if save_path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            # If anything fails, create a basic fallback chart
            plt.close(fig)
            return ChartVisualizer._create_basic_volume_chart(data, save_path, stock_symbol)
    
    @staticmethod
    def plot_enhanced_volume_chart_with_agents(data: pd.DataFrame, 
                                             indicators: Dict[str, Any], 
                                             volume_agents_result: Dict[str, Any] = None,
                                             save_path: str = None, 
                                             stock_symbol: str = 'Stock'):
        """
        Create enhanced volume analysis chart that integrates volume agents insights with robust fallback.
        
        Args:
            data: Stock price and volume data
            indicators: Technical indicators
            volume_agents_result: Result from volume agents analysis (optional)
            save_path: Path to save chart (optional)
            stock_symbol: Stock symbol for chart title
            
        Returns:
            matplotlib figure object
        """
        try:
            # Determine chart complexity based on volume agents availability
            has_volume_agents = (
                volume_agents_result is not None and 
                volume_agents_result.get('success', False) and
                volume_agents_result.get('volume_analysis') is not None
            )
            
            if has_volume_agents:
                return ChartVisualizer._create_agent_enhanced_volume_chart(
                    data, indicators, volume_agents_result, save_path, stock_symbol
                )
            else:
                # Fall back to traditional comprehensive volume chart
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Volume agents unavailable for {stock_symbol}, using traditional volume chart")
                return ChartVisualizer.plot_comprehensive_volume_chart(
                    data, indicators, save_path, stock_symbol
                )
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Enhanced volume chart generation failed for {stock_symbol}: {e}")
            
            # Ultimate fallback to basic volume chart
            return ChartVisualizer._create_basic_volume_chart(data, save_path, stock_symbol)
    
    @staticmethod
    def _create_agent_enhanced_volume_chart(data: pd.DataFrame, 
                                          indicators: Dict[str, Any],
                                          volume_agents_result: Dict[str, Any],
                                          save_path: str = None, 
                                          stock_symbol: str = 'Stock'):
        """
        Create volume chart enhanced with volume agents insights
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        volume_analysis = volume_agents_result.get('volume_analysis', {})
        consensus_analysis = volume_agents_result.get('consensus_analysis', {})
        
        # Main price and volume chart with agent insights
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Price line
        ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1.5)
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Volume bars with agent-specific coloring
        volume_colors = ChartVisualizer._get_agent_volume_colors(data, volume_analysis)
        ax1_twin.bar(data.index, data['volume'], color=volume_colors, alpha=0.7, label='Volume')
        ax1_twin.set_ylabel('Volume', color='lightblue')
        ax1_twin.tick_params(axis='y', labelcolor='lightblue')
        
        # Add agent-specific annotations
        ChartVisualizer._add_volume_agent_annotations(ax1, ax1_twin, data, volume_analysis)
        
        ax1.set_title(f'{stock_symbol} - Volume Analysis with AI Agents', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Volume trend analysis from agents
        ax2 = axes[1]
        volume_ma = data['volume'].rolling(window=20).mean()
        ax2.plot(data.index, data['volume'], label='Volume', color='lightblue', alpha=0.7, linewidth=1)
        ax2.plot(data.index, volume_ma, label='Volume MA 20', color='red', linewidth=1.5)
        
        # Add agent consensus signals
        consensus_signals = consensus_analysis.get('consensus_signals', {})
        if consensus_signals.get('primary_signal') != 'neutral':
            signal = consensus_signals['primary_signal']
            confidence = consensus_analysis.get('overall_confidence', 0)
            ax2.axhline(y=volume_ma.iloc[-1], color='green' if signal == 'bullish' else 'red', 
                       linestyle='--', alpha=0.8, 
                       label=f'Agent Signal: {signal.title()} ({confidence:.1%} confidence)')
        
        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Volume ratio with agent thresholds
        ax3 = axes[2]
        volume_ratio = data['volume'] / volume_ma
        ax3.plot(data.index, volume_ratio, label='Volume Ratio', color='green', linewidth=1.5)
        ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Average')
        
        # Add agent-specific thresholds if available
        agent_status = volume_analysis.get('agent_status', {})
        if agent_status.get('successful_agents', 0) > 0:
            ax3.axhline(y=1.5, color='orange', linestyle='--', alpha=0.6, label='High Volume (Agents)')
            ax3.axhline(y=2.5, color='red', linestyle='--', alpha=0.6, label='Anomaly Threshold')
        else:
            # Fallback to traditional thresholds
            ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.6, label='High Volume')
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Low Volume')
        
        ax3.set_ylabel('Volume Ratio')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Agent performance and reliability indicator
        ax4 = axes[3]
        
        # Show agent success rates and confidence over time (simplified visualization)
        agent_details = agent_status.get('agent_details', {})
        successful_agents = [name for name, details in agent_details.items() if details.get('success', False)]
        failed_agents = [name for name, details in agent_details.items() if not details.get('success', False)]
        
        # Create a simple bar chart showing agent performance
        if successful_agents or failed_agents:
            agents = successful_agents + failed_agents
            success_values = [1] * len(successful_agents) + [0] * len(failed_agents)
            colors = ['green'] * len(successful_agents) + ['red'] * len(failed_agents)
            
            bars = ax4.bar(range(len(agents)), success_values, color=colors, alpha=0.7)
            ax4.set_xticks(range(len(agents)))
            ax4.set_xticklabels([agent.replace('_', ' ').title() for agent in agents], rotation=45)
            ax4.set_ylabel('Agent Status')
            ax4.set_ylim(0, 1.2)
            ax4.set_title('Volume Agents Performance')
            
            # Add success rate annotation
            success_rate = len(successful_agents) / len(agents) if agents else 0
            ax4.text(0.02, 0.95, f'Success Rate: {success_rate:.1%}', 
                    transform=ax4.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No Volume Agents Data Available\nUsing Traditional Analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        ax4.set_xlabel('Volume Analysis Agents')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file only if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def _create_basic_volume_chart(data: pd.DataFrame, save_path: str = None, stock_symbol: str = 'Stock'):
        """
        Create a basic volume chart as ultimate fallback
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Basic price chart
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        ax1.set_title(f'{stock_symbol} - Basic Volume Analysis (Fallback Mode)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Basic volume chart
        ax2.bar(data.index, data['volume'], color='lightblue', alpha=0.7, label='Volume')
        try:
            volume_ma = data['volume'].rolling(window=20).mean()
            ax2.plot(data.index, volume_ma, color='red', linewidth=1.5, label='20-day MA')
        except:
            pass  # Skip moving average if calculation fails
        
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def _get_agent_volume_colors(data: pd.DataFrame, volume_analysis: Dict[str, Any]) -> list:
        """
        Get volume bar colors based on volume agents analysis
        """
        try:
            volume_summary = volume_analysis.get('volume_summary', {})
            key_findings = volume_analysis.get('key_findings', [])
            
            # Default color scheme
            colors = ['lightblue'] * len(data)
            
            # Color based on volume analysis insights
            if 'anomalies' in str(key_findings).lower():
                # Highlight potential anomaly periods in red
                volume_ma = data['volume'].rolling(window=20).mean()
                volume_ratio = data['volume'] / volume_ma
                
                for i, ratio in enumerate(volume_ratio):
                    if ratio > 2.5:  # High volume anomaly
                        colors[i] = 'red'
                    elif ratio > 1.8:  # Elevated volume
                        colors[i] = 'orange'
                    elif ratio < 0.3:  # Low volume
                        colors[i] = 'gray'
            
            return colors
            
        except Exception as e:
            # Fallback to default coloring
            return ['lightblue'] * len(data)
    
    @staticmethod
    def _add_volume_agent_annotations(ax1, ax1_twin, data: pd.DataFrame, volume_analysis: Dict[str, Any]):
        """
        Add annotations based on volume agents insights
        """
        try:
            key_findings = volume_analysis.get('key_findings', [])
            
            # Add annotations for significant findings
            for i, finding in enumerate(key_findings[:3]):  # Limit to top 3 findings
                if 'anomaly' in finding.lower() or 'institutional' in finding.lower():
                    # Find the most recent high volume point to annotate
                    volume_ma = data['volume'].rolling(window=20).mean()
                    volume_ratio = data['volume'] / volume_ma
                    
                    # Find recent high volume points
                    high_volume_indices = volume_ratio[volume_ratio > 2.0].index
                    if len(high_volume_indices) > 0:
                        # Use the most recent high volume point
                        annotation_date = high_volume_indices[-1]
                        annotation_volume = data.loc[annotation_date, 'volume']
                        annotation_price = data.loc[annotation_date, 'close']
                        
                        # Add annotation
                        ax1.annotate(
                            finding[:30] + '...' if len(finding) > 30 else finding,
                            xy=(annotation_date, annotation_price),
                            xytext=(10, 20 + i * 15),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                            fontsize=8
                        )
                        break  # Only add one annotation to avoid clutter
                        
        except Exception as e:
            # Skip annotations if they fail
            pass
        
        # Price and volume correlation
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Price line
        ax1.plot(data.index, data['close'], label='Price', color='blue', linewidth=1.5)
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Volume bars
        ax1_twin.bar(data.index, data['volume'], color='lightblue', alpha=0.7, label='Volume')
        ax1_twin.set_ylabel('Volume', color='lightblue')
        ax1_twin.tick_params(axis='y', labelcolor='lightblue')
        
        # Volume anomalies
        from patterns.recognition import PatternRecognition
        anomalies = PatternRecognition.detect_volume_anomalies(data['volume'])
        for anomaly in anomalies:
            # Ensure anomaly is a valid integer index
            if isinstance(anomaly, (int, np.integer)) and 0 <= anomaly < len(data):
                ax1_twin.bar(data.index[anomaly], data['volume'].iloc[anomaly], color='red', alpha=0.8)
        
        ax1.set_title(f'{stock_symbol} - Comprehensive Volume Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Volume trend
        ax2 = axes[1]
        volume_ma = data['volume'].rolling(window=20).mean()
        ax2.plot(data.index, data['volume'], label='Volume', color='lightblue', alpha=0.7)
        ax2.plot(data.index, volume_ma, label='Volume MA 20', color='red', linewidth=1.5)
        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Volume ratio
        ax3 = axes[2]
        volume_ratio = data['volume'] / volume_ma
        ax3.plot(data.index, volume_ratio, label='Volume Ratio', color='green', linewidth=1.5)
        ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Average')
        ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.6, label='High Volume')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Low Volume')
        ax3.set_ylabel('Volume Ratio')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file only if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return the figure object instead of closing it
        return fig

    @staticmethod
    def plot_multi_period_ma_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str = None, stock_symbol: str = 'Stock'):
        """
        Create multi-period moving average chart showing SMA 20/50/200 on a single timeframe.
        
        NOTE: This is NOT a true multi-timeframe analysis chart. It only shows different
        period moving averages on ONE timeframe. For true MTF analysis, see
        agents.mtf_analysis.visualization.MTFVisualizer.
        
        Returns matplotlib figure object. Optionally saves to file if save_path is provided.
        """
    
    @staticmethod
    def plot_mtf_comparison_chart(data: pd.DataFrame, indicators: Dict[str, Any], save_path: str = None, stock_symbol: str = 'Stock'):
        """
        DEPRECATED: Use plot_multi_period_ma_chart() instead.
        This function name is misleading - it doesn't show multi-timeframe data.
        
        For true multi-timeframe visualization, use:
        agents.mtf_analysis.visualization.MTFVisualizer.create_mtf_comparison_chart()
        """
        import warnings
        warnings.warn(
            "plot_mtf_comparison_chart is deprecated and misleadingly named. "
            "Use plot_multi_period_ma_chart() for multi-period MAs, or "
            "MTFVisualizer.create_mtf_comparison_chart() for true MTF analysis.",
            DeprecationWarning,
            stacklevel=2
        )
        return ChartVisualizer.plot_multi_period_ma_chart(data, indicators, save_path, stock_symbol)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Main price chart with multiple timeframes
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        
        # Add different timeframe moving averages computed from data (ensures full-length series)
        from ml.indicators.technical_indicators import TechnicalIndicators
        sma_20_series = TechnicalIndicators.calculate_sma(data, 'close', 20)
        sma_50_series = TechnicalIndicators.calculate_sma(data, 'close', 50)
        sma_200_series = TechnicalIndicators.calculate_sma(data, 'close', 200)
        ax1.plot(data.index, sma_20_series, label='SMA 20 (Short-term)', color='orange', alpha=0.7)
        ax1.plot(data.index, sma_50_series, label='SMA 50 (Medium-term)', color='red', alpha=0.7)
        ax1.plot(data.index, sma_200_series, label='SMA 200 (Long-term)', color='purple', alpha=0.7)
        
        # Add trend lines for different timeframes
        # Short-term trend (last 20 days)
        if len(data) >= 20:
            short_trend = np.polyfit(range(20), data['close'].iloc[-20:], 1)
            short_trend_line = np.poly1d(short_trend)(range(20))
            ax1.plot(data.index[-20:], short_trend_line, color='orange', linestyle='--', alpha=0.8, label='Short-term Trend')
        
        # Medium-term trend (last 50 days)
        if len(data) >= 50:
            medium_trend = np.polyfit(range(50), data['close'].iloc[-50:], 1)
            medium_trend_line = np.poly1d(medium_trend)(range(50))
            ax1.plot(data.index[-50:], medium_trend_line, color='red', linestyle='--', alpha=0.8, label='Medium-term Trend')
        
        # Long-term trend (last 200 days)
        if len(data) >= 200:
            long_trend = np.polyfit(range(200), data['close'].iloc[-200:], 1)
            long_trend_line = np.poly1d(long_trend)(range(200))
            ax1.plot(data.index[-200:], long_trend_line, color='purple', linestyle='--', alpha=0.8, label='Long-term Trend')
        
        ax1.set_title(f'{stock_symbol} - Multi-Period Moving Averages (SMA 20/50/200)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume across timeframes
        ax2 = axes[1]
        ax2.bar(data.index, data['volume'], color='lightblue', alpha=0.7, label='Volume')
        
        # Volume moving averages for different timeframes
        if len(data) >= 20:
            volume_ma_20 = data['volume'].rolling(window=20).mean()
            ax2.plot(data.index, volume_ma_20, label='Volume MA 20', color='orange', linewidth=1.5)
        if len(data) >= 50:
            volume_ma_50 = data['volume'].rolling(window=50).mean()
            ax2.plot(data.index, volume_ma_50, label='Volume MA 50', color='red', linewidth=1.5)
        
        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # RSI for momentum across timeframes
        ax3 = axes[2]
        try:
            from ml.indicators.technical_indicators import TechnicalIndicators
            rsi_series = TechnicalIndicators.calculate_rsi(data)
        except Exception:
            rsi_series = None
        if rsi_series is not None and len(rsi_series) == len(data.index):
            ax3.plot(data.index, rsi_series, label='RSI 14', color='purple', linewidth=1.5)
        # Draw reference lines and cosmetics even if RSI failed to compute
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.6, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.6, label='Oversold')
        ax3.set_ylabel('RSI')
        ax3.set_xlabel('Date')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file only if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return the figure object instead of closing it
        return fig 