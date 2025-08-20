#!/usr/bin/env python3
"""
Validation Script for Step 1.4: Risk Management

This script validates the risk management implementation by testing:
1. Position sizing calculations
2. Stop-loss and take-profit management
3. Risk metrics calculation (VaR, CVaR, Sharpe ratio, etc.)
4. Portfolio risk management
5. Drawdown analysis
6. Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the quant_system directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from risk_management import RiskConfig, PositionSizer, StopLossManager, RiskMetrics, PortfolioRiskManager, RiskAdjustedPerformance

def test_position_sizing():
    """Test position sizing functionality."""
    print("🔍 Testing Position Sizing")
    print("=" * 50)
    
    # Create risk config
    config = RiskConfig(
        initial_capital=100000.0,
        max_position_size=0.1,
        max_portfolio_risk=0.02
    )
    
    position_sizer = PositionSizer(config)
    
    # Test basic position sizing
    capital = 100000.0
    entry_price = 100.0
    stop_loss_price = 98.0  # 2% stop loss
    
    position_result = position_sizer.calculate_position_size(capital, entry_price, stop_loss_price)
    
    if position_result:
        print(f"✅ Position sizing calculated successfully")
        print(f"   Position size: ${position_result['position_size']:.2f}")
        print(f"   Shares: {position_result['shares']:.0f}")
        print(f"   Risk amount: ${position_result['risk_amount']:.2f}")
        print(f"   Risk percentage: {position_result['risk_percentage']:.2%}")
        
        # Check that risk is within limits
        if position_result['risk_percentage'] <= config.max_portfolio_risk:
            print("✅ Risk within portfolio limits")
        else:
            print("❌ Risk exceeds portfolio limits")
            return False
        
        return True
    else:
        print("❌ Position sizing calculation failed")
        return False

def test_kelly_criterion():
    """Test Kelly Criterion position sizing."""
    print("\n🔍 Testing Kelly Criterion")
    print("=" * 50)
    
    config = RiskConfig()
    position_sizer = PositionSizer(config)
    
    # Test Kelly Criterion
    win_rate = 0.6
    avg_win = 0.03  # 3% average win
    avg_loss = 0.02  # 2% average loss
    capital = 100000.0
    
    kelly_result = position_sizer.calculate_kelly_position_size(win_rate, avg_win, avg_loss, capital)
    
    if kelly_result:
        print(f"✅ Kelly Criterion calculated successfully")
        print(f"   Kelly fraction: {kelly_result['kelly_fraction']:.4f}")
        print(f"   Position size: ${kelly_result['position_size']:.2f}")
        
        # Check that Kelly fraction is reasonable
        if 0 <= kelly_result['kelly_fraction'] <= config.max_position_size:
            print("✅ Kelly fraction within reasonable bounds")
        else:
            print("❌ Kelly fraction out of bounds")
            return False
        
        return True
    else:
        print("❌ Kelly Criterion calculation failed")
        return False

def test_stop_loss_management():
    """Test stop-loss and take-profit management."""
    print("\n🔍 Testing Stop-Loss Management")
    print("=" * 50)
    
    config = RiskConfig()
    stop_loss_manager = StopLossManager(config)
    
    # Test stop-loss calculation
    entry_price = 100.0
    direction = 'long'
    
    # Test percentage-based stop loss
    stop_loss = stop_loss_manager.calculate_stop_loss(entry_price, direction)
    expected_stop_loss = entry_price * (1 - config.default_stop_loss)
    
    print(f"✅ Stop-loss calculated: ${stop_loss:.2f}")
    print(f"   Expected stop-loss: ${expected_stop_loss:.2f}")
    
    if abs(stop_loss - expected_stop_loss) < 0.01:
        print("✅ Stop-loss calculation correct")
    else:
        print("❌ Stop-loss calculation incorrect")
        return False
    
    # Test take-profit calculation
    take_profit = stop_loss_manager.calculate_take_profit(entry_price, direction)
    expected_take_profit = entry_price * (1 + config.default_take_profit)
    
    print(f"✅ Take-profit calculated: ${take_profit:.2f}")
    print(f"   Expected take-profit: ${expected_take_profit:.2f}")
    
    if abs(take_profit - expected_take_profit) < 0.01:
        print("✅ Take-profit calculation correct")
    else:
        print("❌ Take-profit calculation incorrect")
        return False
    
    # Test stop-loss hit detection
    current_price = 97.0  # Below stop loss
    stop_loss_hit = stop_loss_manager.check_stop_loss_hit(current_price, stop_loss, direction)
    
    if stop_loss_hit:
        print("✅ Stop-loss hit detection working")
    else:
        print("❌ Stop-loss hit detection failed")
        return False
    
    return True

def test_risk_metrics():
    """Test risk metrics calculation."""
    print("\n🔍 Testing Risk Metrics")
    print("=" * 50)
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    print(f"✅ Test data created: {len(returns)} daily returns")
    
    config = RiskConfig()
    risk_metrics = RiskMetrics(config)
    
    # Test VaR calculation
    var = risk_metrics.calculate_var(returns)
    print(f"✅ VaR calculated: {var:.4f}")
    
    if var > 0:
        print("✅ VaR calculation successful")
    else:
        print("❌ VaR calculation failed")
        return False
    
    # Test CVaR calculation
    cvar = risk_metrics.calculate_cvar(returns)
    print(f"✅ CVaR calculated: {cvar:.4f}")
    
    if cvar > 0:
        print("✅ CVaR calculation successful")
    else:
        print("❌ CVaR calculation failed")
        return False
    
    # Test Sharpe ratio calculation
    sharpe = risk_metrics.calculate_sharpe_ratio(returns)
    print(f"✅ Sharpe ratio calculated: {sharpe:.4f}")
    
    # Test Sortino ratio calculation
    sortino = risk_metrics.calculate_sortino_ratio(returns)
    print(f"✅ Sortino ratio calculated: {sortino:.4f}")
    
    # Test maximum drawdown calculation
    prices = (1 + returns).cumprod()
    drawdown_result = risk_metrics.calculate_max_drawdown(prices)
    
    print(f"✅ Maximum drawdown calculated: {drawdown_result['max_drawdown']:.4f}")
    print(f"   Drawdown duration: {drawdown_result['drawdown_duration']} periods")
    
    if drawdown_result['max_drawdown'] >= 0:
        print("✅ Drawdown calculation successful")
    else:
        print("❌ Drawdown calculation failed")
        return False
    
    return True

def test_portfolio_risk_management():
    """Test portfolio risk management."""
    print("\n🔍 Testing Portfolio Risk Management")
    print("=" * 50)
    
    # Create sample portfolio data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # Create returns for multiple assets
    returns_data = {
        'Asset_A': np.random.normal(0.001, 0.02, 252),
        'Asset_B': np.random.normal(0.001, 0.025, 252),
        'Asset_C': np.random.normal(0.001, 0.03, 252)
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    print(f"✅ Portfolio data created: {returns_df.shape}")
    
    config = RiskConfig()
    portfolio_manager = PortfolioRiskManager(config)
    
    # Test position limits
    current_positions = {'Asset_A': 0.3, 'Asset_B': 0.2}
    new_position = {'position_size': 0.1}
    
    position_allowed = portfolio_manager.check_position_limits(current_positions, new_position)
    
    if position_allowed:
        print("✅ Position limits check passed")
    else:
        print("❌ Position limits check failed")
        return False
    
    # Test portfolio weights calculation
    positions = {'Asset_A': 30000, 'Asset_B': 20000, 'Asset_C': 10000}
    weights = portfolio_manager.calculate_portfolio_weights(positions)
    
    print(f"✅ Portfolio weights calculated: {weights}")
    
    # Check that weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) < 0.001:
        print("✅ Portfolio weights sum to 1.0")
    else:
        print(f"❌ Portfolio weights sum to {weight_sum:.4f}")
        return False
    
    # Test portfolio metrics calculation
    portfolio_metrics = portfolio_manager.calculate_portfolio_metrics(positions, returns_df)
    
    if portfolio_metrics:
        print("✅ Portfolio metrics calculated successfully")
        for metric, value in portfolio_metrics.items():
            print(f"   {metric}: {value:.4f}")
    else:
        print("❌ Portfolio metrics calculation failed")
        return False
    
    return True

def test_risk_adjusted_performance():
    """Test risk-adjusted performance metrics."""
    print("\n🔍 Testing Risk-Adjusted Performance")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # Portfolio returns
    portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    # Market returns (benchmark)
    market_returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
    
    print(f"✅ Test data created: {len(portfolio_returns)} periods")
    
    config = RiskConfig()
    performance = RiskAdjustedPerformance(config)
    
    # Test information ratio
    info_ratio = performance.calculate_information_ratio(portfolio_returns, market_returns)
    print(f"✅ Information ratio calculated: {info_ratio:.4f}")
    
    # Test Calmar ratio
    calmar_ratio = performance.calculate_calmar_ratio(portfolio_returns)
    print(f"✅ Calmar ratio calculated: {calmar_ratio:.4f}")
    
    # Test Treynor ratio
    treynor_ratio = performance.calculate_treynor_ratio(portfolio_returns, market_returns)
    print(f"✅ Treynor ratio calculated: {treynor_ratio:.4f}")
    
    # Test Jensen's alpha
    jensen_alpha = performance.calculate_jensen_alpha(portfolio_returns, market_returns)
    print(f"✅ Jensen's alpha calculated: {jensen_alpha:.4f}")
    
    # Test beta calculation
    beta = performance.risk_metrics.calculate_beta(portfolio_returns, market_returns)
    print(f"✅ Beta calculated: {beta:.4f}")
    
    print("✅ All risk-adjusted performance metrics calculated successfully")
    return True

def test_correlation_analysis():
    """Test correlation analysis."""
    print("\n🔍 Testing Correlation Analysis")
    print("=" * 50)
    
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # Create correlated returns
    returns_data = {
        'Asset_A': np.random.normal(0.001, 0.02, 252),
        'Asset_B': np.random.normal(0.001, 0.025, 252),
        'Asset_C': np.random.normal(0.001, 0.03, 252)
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    print(f"✅ Test data created: {returns_df.shape}")
    
    config = RiskConfig()
    risk_metrics = RiskMetrics(config)
    
    # Calculate correlation matrix
    correlation_matrix = risk_metrics.calculate_correlation_matrix(returns_df)
    
    if not correlation_matrix.empty:
        print("✅ Correlation matrix calculated successfully")
        print(f"   Matrix shape: {correlation_matrix.shape}")
        
        # Check that diagonal elements are 1
        diagonal_ones = all(abs(correlation_matrix.iloc[i, i] - 1.0) < 0.001 
                           for i in range(len(correlation_matrix)))
        
        if diagonal_ones:
            print("✅ Correlation matrix diagonal elements are 1.0")
        else:
            print("❌ Correlation matrix diagonal elements are not 1.0")
            return False
        
        # Check that matrix is symmetric
        is_symmetric = correlation_matrix.equals(correlation_matrix.T)
        
        if is_symmetric:
            print("✅ Correlation matrix is symmetric")
        else:
            print("❌ Correlation matrix is not symmetric")
            return False
        
        return True
    else:
        print("❌ Correlation matrix calculation failed")
        return False

def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 STEP 1.4 VALIDATION: Risk Management")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Position Sizing", test_position_sizing()))
    test_results.append(("Kelly Criterion", test_kelly_criterion()))
    test_results.append(("Stop-Loss Management", test_stop_loss_management()))
    test_results.append(("Risk Metrics", test_risk_metrics()))
    test_results.append(("Portfolio Risk Management", test_portfolio_risk_management()))
    test_results.append(("Risk-Adjusted Performance", test_risk_adjusted_performance()))
    test_results.append(("Correlation Analysis", test_correlation_analysis()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 STEP 1.4 VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ Risk management is working correctly")
        print("✅ Ready to proceed to Step 1.5: Backtesting Engine")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
