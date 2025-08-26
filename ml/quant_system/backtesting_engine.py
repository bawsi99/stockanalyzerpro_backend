"""
Backtesting Engine for Quantitative Trading System

This module provides comprehensive backtesting capabilities for:
1. Strategy backtesting with realistic market conditions
2. Performance analysis and reporting
3. Risk-adjusted metrics calculation
4. Trade analysis and optimization
5. Walk-forward analysis
6. Monte Carlo simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Capital and fees
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage
    
    # Position sizing
    position_size: float = 0.1  # 10% of capital per trade
    max_positions: int = 10  # Maximum concurrent positions
    
    # Risk management
    stop_loss: float = 0.02  # 2% stop loss
    take_profit: float = 0.04  # 4% take profit
    trailing_stop: bool = True
    trailing_stop_distance: float = 0.01  # 1% trailing stop
    
    # Backtesting parameters
    start_date: datetime = None
    end_date: datetime = None
    benchmark: str = None  # Benchmark for comparison
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime(2020, 1, 1)
        if self.end_date is None:
            self.end_date = datetime(2024, 1, 1)

class Trade:
    """Represents a single trade."""
    
    def __init__(self, symbol: str, entry_date: datetime, entry_price: float, 
                 direction: str = 'long', shares: int = 0):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction
        self.shares = shares
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.status = 'open'  # 'open', 'closed', 'stopped'
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.entry_commission = 0.0
        self.exit_commission = 0.0
    
    def update_trailing_stop(self, current_price: float, trailing_distance: float):
        """Update trailing stop."""
        if self.direction == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop = current_price * (1 - trailing_distance)
        else:  # short
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                self.trailing_stop = current_price * (1 + trailing_distance)
    
    def check_exit_conditions(self, current_price: float, current_date: datetime) -> bool:
        """Check if trade should be closed."""
        if self.status != 'open':
            return False
        
        # Check stop loss
        if self.stop_loss is not None:
            if (self.direction == 'long' and current_price <= self.stop_loss) or \
               (self.direction == 'short' and current_price >= self.stop_loss):
                self.exit_date = current_date
                self.exit_price = self.stop_loss
                self.status = 'stopped'
                return True
        
        # Check take profit
        if self.take_profit is not None:
            if (self.direction == 'long' and current_price >= self.take_profit) or \
               (self.direction == 'short' and current_price <= self.take_profit):
                self.exit_date = current_date
                self.exit_price = self.take_profit
                self.status = 'closed'
                return True
        
        # Check trailing stop
        if self.trailing_stop is not None:
            if (self.direction == 'long' and current_price <= self.trailing_stop) or \
               (self.direction == 'short' and current_price >= self.trailing_stop):
                self.exit_date = current_date
                self.exit_price = self.trailing_stop
                self.status = 'stopped'
                return True
        
        return False
    
    def close_trade(self, exit_date: datetime, exit_price: float):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.status = 'closed'
        
        # Calculate P&L
        if self.direction == 'long':
            self.pnl = (exit_price - self.entry_price) * self.shares
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.shares
        
        self.pnl_pct = self.pnl / (self.entry_price * self.shares)

class Portfolio:
    """Manages portfolio state during backtesting."""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.0, slippage_rate: float = 0.0, trailing_stop_enabled: bool = True, trailing_stop_distance: float = 0.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # symbol -> Trade
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_distance = trailing_stop_distance
        
    def add_trade(self, trade: Trade):
        """Add a new trade to the portfolio."""
        if trade.symbol in self.positions:
            logger.warning(f"Trade for {trade.symbol} already exists")
            return False
        
        # Apply slippage to entry and compute commission
        if trade.direction == 'long':
            effective_entry_price = trade.entry_price * (1 + self.slippage_rate)
        else:
            effective_entry_price = trade.entry_price * (1 - self.slippage_rate)
        position_value = effective_entry_price * trade.shares
        entry_commission = position_value * self.commission_rate
        
        # Check if we have enough capital
        if position_value + entry_commission > self.capital:
            logger.warning(f"Insufficient capital for trade: {trade.symbol}")
            return False
        
        # Deduct capital (including commission)
        self.capital -= (position_value + entry_commission)

        # Persist effective entry and commission
        trade.entry_price = effective_entry_price
        trade.entry_commission = entry_commission

        # Initialize trailing stop if enabled
        if self.trailing_stop_enabled and self.trailing_stop_distance > 0:
            if trade.direction == 'long':
                trade.trailing_stop = trade.entry_price * (1 - self.trailing_stop_distance)
            else:
                trade.trailing_stop = trade.entry_price * (1 + self.trailing_stop_distance)
        self.positions[trade.symbol] = trade
        
        return True
    
    def close_trade(self, symbol: str, exit_date: datetime, exit_price: float):
        """Close a trade."""
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        # Apply slippage to exit and compute commission
        if trade.direction == 'long':
            effective_exit_price = exit_price * (1 - self.slippage_rate)
        else:
            effective_exit_price = exit_price * (1 + self.slippage_rate)

        exit_gross_value = effective_exit_price * trade.shares
        exit_commission = exit_gross_value * self.commission_rate

        # Set trade exit details
        trade.close_trade(exit_date, effective_exit_price)
        trade.exit_commission = exit_commission
        
        # Add back capital net of commission
        self.capital += (exit_gross_value - exit_commission)

        # Recalculate P&L net of commissions
        gross_pnl = (trade.exit_price - trade.entry_price) * trade.shares if trade.direction == 'long' \
            else (trade.entry_price - trade.exit_price) * trade.shares
        net_pnl = gross_pnl - trade.entry_commission - trade.exit_commission
        trade.pnl = net_pnl
        trade.pnl_pct = net_pnl / (trade.entry_price * trade.shares) if trade.shares > 0 else 0.0
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        return True
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Get total portfolio value."""
        total_value = self.capital
        
        for symbol, trade in self.positions.items():
            if symbol in current_prices:
                position_value = current_prices[symbol] * trade.shares
                total_value += position_value
        
        return total_value
    
    def update_equity_curve(self, current_prices: Dict[str, float], date: datetime):
        """Update equity curve."""
        total_value = self.get_total_value(current_prices)
        self.equity_curve.append({
            'date': date,
            'total_value': total_value,
            'capital': self.capital,
            'positions_value': total_value - self.capital
        })
    
    def calculate_daily_returns(self):
        """Calculate daily returns from equity curve."""
        if len(self.equity_curve) < 2:
            return
        
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1]['total_value']
            curr_value = self.equity_curve[i]['total_value']
            daily_return = (curr_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(
            self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            trailing_stop_enabled=self.config.trailing_stop,
            trailing_stop_distance=self.config.trailing_stop_distance,
        )
        self.data = {}
        self.benchmark_data = None
        self.results = {}
    
    def load_data(self, data: Dict[str, pd.DataFrame]):
        """Load data for backtesting."""
        self.data = data
        logger.info(f"Loaded data for {len(data)} symbols")
    
    def load_benchmark(self, benchmark_data: pd.DataFrame):
        """Load benchmark data."""
        self.benchmark_data = benchmark_data
        logger.info("Loaded benchmark data")
    
    def run_backtest(self, strategy_function: Callable) -> Dict[str, Any]:
        """Run backtest with given strategy."""
        logger.info("Starting backtest...")
        
        # Get all dates
        all_dates = set()
        for symbol, df in self.data.items():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        
        # Filter dates by config
        start_idx = 0
        end_idx = len(all_dates)
        
        if self.config.start_date:
            start_idx = max(0, np.searchsorted(all_dates, self.config.start_date))
        if self.config.end_date:
            end_idx = min(len(all_dates), np.searchsorted(all_dates, self.config.end_date))
        
        backtest_dates = all_dates[start_idx:end_idx]
        if not backtest_dates:
            logger.warning("No dates available in the selected backtest range")
            self.results = {}
            return self.results
        logger.info(f"Running backtest from {backtest_dates[0]} to {backtest_dates[-1]}")
        
        # Main backtest loop
        for date in backtest_dates:
            # Get current prices
            current_prices = {}
            for symbol, df in self.data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'close']
            
            # Update trailing stops
            if self.config.trailing_stop:
                for symbol, trade in self.portfolio.positions.items():
                    if symbol in current_prices:
                        trade.update_trailing_stop(current_prices[symbol], 
                                                   self.config.trailing_stop_distance)
            
            # Check exit conditions
            for symbol, trade in list(self.portfolio.positions.items()):
                if symbol in current_prices:
                    if trade.check_exit_conditions(current_prices[symbol], date):
                        # Use the exit price determined by the triggered condition
                        self.portfolio.close_trade(symbol, date, trade.exit_price)
            
            # Run strategy
            if len(self.portfolio.positions) < self.config.max_positions:
                strategy_function(self, date, current_prices)
            
            # Update equity curve
            self.portfolio.update_equity_curve(current_prices, date)
        
        # Calculate results
        self.calculate_results()
        
        logger.info("Backtest completed")
        return self.results
    
    def calculate_results(self):
        """Calculate backtest results."""
        if not self.portfolio.equity_curve:
            logger.warning("No equity curve data available")
            return
        
        # Calculate daily returns
        self.portfolio.calculate_daily_returns()
        
        # Basic metrics
        initial_value = self.portfolio.initial_capital
        final_value = self.portfolio.equity_curve[-1]['total_value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate metrics
        daily_returns = pd.Series(self.portfolio.daily_returns)
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        equity_series = pd.Series([e['total_value'] for e in self.portfolio.equity_curve])
        cumulative_returns = (equity_series / equity_series.iloc[0]) - 1
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(drawdown.min())
        
        # Trade statistics
        total_trades = len(self.portfolio.closed_trades)
        winning_trades = len([t for t in self.portfolio.closed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in self.portfolio.closed_trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.portfolio.closed_trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Store results
        num_days = max(1, len(self.portfolio.equity_curve))
        try:
            cagr = (final_value / initial_value) ** (252 / num_days) - 1
        except Exception:
            cagr = 0.0
        self.results = {
            'total_return': total_return,
            'annualized_return': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_value': final_value,
            'equity_curve': self.portfolio.equity_curve,
            'closed_trades': self.portfolio.closed_trades
        }
    
    def generate_report(self) -> str:
        """Generate backtest report."""
        if not self.results:
            return "No results available"
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST RESULTS")
        report.append("=" * 60)
        report.append(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        report.append(f"Final Value: ${self.results['final_value']:,.2f}")
        report.append(f"Total Return: {self.results['total_return']:.2%}")
        report.append(f"Annualized Return: {self.results['annualized_return']:.2%}")
        report.append(f"Volatility: {self.results['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {self.results['max_drawdown']:.2%}")
        report.append("")
        report.append("TRADE STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {self.results['total_trades']}")
        report.append(f"Winning Trades: {self.results['winning_trades']}")
        report.append(f"Losing Trades: {self.results['losing_trades']}")
        report.append(f"Win Rate: {self.results['win_rate']:.2%}")
        report.append(f"Average Win: ${self.results['avg_win']:,.2f}")
        report.append(f"Average Loss: ${self.results['avg_loss']:,.2f}")
        report.append(f"Profit Factor: {self.results['profit_factor']:.2f}")
        
        return "\n".join(report)
    
    def plot_results(self):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.portfolio.equity_curve:
                logger.warning("No equity curve data to plot")
                return
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot equity curve
            dates = [e['date'] for e in self.portfolio.equity_curve]
            values = [e['total_value'] for e in self.portfolio.equity_curve]
            
            ax1.plot(dates, values, label='Portfolio Value', linewidth=2)
            ax1.axhline(y=self.portfolio.initial_capital, color='r', linestyle='--', 
                       label='Initial Capital')
            ax1.set_title('Portfolio Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot drawdown
            equity_series = pd.Series(values, index=dates)
            cumulative_returns = (equity_series / equity_series.iloc[0]) - 1
            running_max = cumulative_returns.expanding().max()
            drawdown = cumulative_returns - running_max
            
            ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(dates, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")

class StrategyTemplate:
    """Template for implementing trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from data."""
        # This should be implemented by specific strategies
        raise NotImplementedError
    
    def execute_strategy(self, engine: BacktestEngine, date: datetime, 
                        current_prices: Dict[str, float]):
        """Execute strategy logic."""
        # This should be implemented by specific strategies
        raise NotImplementedError

class SimpleMovingAverageStrategy(StrategyTemplate):
    """Simple moving average crossover strategy."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("Simple Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on moving average crossover."""
        df = data.copy()
        
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1  # Buy signal
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1  # Sell signal
        
        return df
    
    def execute_strategy(self, engine: BacktestEngine, date: datetime, 
                        current_prices: Dict[str, float]):
        """Execute the moving average strategy."""
        for symbol, df in engine.data.items():
            if symbol in current_prices and date in df.index:
                # Ensure signals exist; compute if missing
                if 'signal' not in df.columns:
                    df = self.generate_signals(df)
                    engine.data[symbol] = df
                # Get current signal
                if date in df.index:
                    signal = df.loc[date, 'signal'] if 'signal' in df.columns else 0
                    
                    # Execute trades based on signal
                    if signal == 1 and symbol not in engine.portfolio.positions:
                        # Buy signal
                        price = current_prices[symbol]
                        shares = int(engine.portfolio.capital * engine.config.position_size / price)
                        
                        if shares > 0:
                            trade = Trade(symbol, date, price, 'long', shares)
                            trade.stop_loss = price * (1 - engine.config.stop_loss)
                            trade.take_profit = price * (1 + engine.config.take_profit)
                            
                            if engine.portfolio.add_trade(trade):
                                logger.info(f"Opened long position in {symbol} at {price}")
                    
                    elif signal == -1 and symbol in engine.portfolio.positions:
                        # Sell signal
                        trade = engine.portfolio.positions[symbol]
                        if trade.direction == 'long':
                            engine.portfolio.close_trade(symbol, date, current_prices[symbol])
                            logger.info(f"Closed long position in {symbol} at {current_prices[symbol]}")

class WalkForwardAnalyzer:
    """Perform walk-forward analysis."""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = []
    
    def run_walk_forward(self, data: Dict[str, pd.DataFrame], 
                        strategy_class: type, 
                        train_period: int = 252, 
                        test_period: int = 63,
                        step_size: int = 21) -> List[Dict[str, Any]]:
        """Run walk-forward analysis."""
        logger.info("Starting walk-forward analysis...")
        
        # Get all dates
        all_dates = set()
        for symbol, df in data.items():
            all_dates.update(df.index)
        
        all_dates = sorted(list(all_dates))
        
        results = []
        
        for i in range(0, len(all_dates) - train_period - test_period, step_size):
            # Define train and test periods
            train_start = all_dates[i]
            train_end = all_dates[i + train_period]
            test_start = all_dates[i + train_period]
            test_end = all_dates[i + train_period + test_period]
            
            logger.info(f"Walk-forward period {i//step_size + 1}: {train_start} to {test_end}")
            
            # Create strategy instance
            strategy = strategy_class()
            
            # Run backtest on test period
            engine = BacktestEngine(self.config)
            
            # Filter data for test period
            test_data = {}
            for symbol, df in data.items():
                test_data[symbol] = df[test_start:test_end]
            
            engine.load_data(test_data)
            
            # Run backtest
            engine.run_backtest(strategy.execute_strategy)
            
            # Store results
            period_result = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'results': engine.results
            }
            
            results.append(period_result)
        
        self.results = results
        logger.info(f"Walk-forward analysis completed: {len(results)} periods")
        
        return results
    
    def generate_walk_forward_report(self) -> str:
        """Generate walk-forward analysis report."""
        if not self.results:
            return "No walk-forward results available"
        
        # Calculate average metrics
        total_returns = [r['results']['total_return'] for r in self.results]
        sharpe_ratios = [r['results']['sharpe_ratio'] for r in self.results]
        max_drawdowns = [r['results']['max_drawdown'] for r in self.results]
        win_rates = [r['results']['win_rate'] for r in self.results]
        
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD ANALYSIS RESULTS")
        report.append("=" * 60)
        report.append(f"Number of Periods: {len(self.results)}")
        report.append(f"Average Total Return: {np.mean(total_returns):.2%}")
        report.append(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
        report.append(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
        report.append(f"Average Win Rate: {np.mean(win_rates):.2%}")
        report.append(f"Return Std Dev: {np.std(total_returns):.2%}")
        report.append(f"Sharpe Std Dev: {np.std(sharpe_ratios):.2f}")
        
        return "\n".join(report)
