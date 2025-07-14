import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.dates as mdates
from typing import Dict, Any

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
        from technical_indicators import TechnicalIndicators
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