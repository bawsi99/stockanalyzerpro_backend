#!/usr/bin/env python3
"""
Enhanced Sector Classifier that builds on the existing sector classification system.
This module adds instrument filtering and improved categorization while maintaining backward compatibility.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set
import json
from pathlib import Path

# Import existing components
from agents.sector.classifier import SectorClassifier
from core.instrument_filter import instrument_filter
from analysis.datasets import analysis_datasets

class EnhancedSectorClassifier(SectorClassifier):
    """
    Enhanced sector classifier with instrument filtering and improved categorization.
    Maintains backward compatibility with existing SectorClassifier.
    """
    
    def __init__(self, sector_folder: str = "sector_category"):
        """
        Initialize the enhanced sector classifier
        
        Args:
            sector_folder: Path to the folder containing sector JSON files
        """
        # Initialize parent class
        super().__init__(sector_folder)
        
        # Enhanced features
        self.major_stocks = set()
        self.filtered_equity_stocks = pd.DataFrame()
        self.instrument_breakdown = {}
        
        # Load major stocks
        self._load_major_stocks()
    
    def _load_major_stocks(self):
        """Load and filter major stocks for analysis"""
        try:
            # Load instruments from CSV
            instruments_df = instrument_filter.load_instruments_from_csv()
            
            if not instruments_df.empty:
                # Get instrument breakdown
                self.instrument_breakdown = instrument_filter.get_instrument_breakdown(instruments_df)
                
                # Filter equity stocks
                self.filtered_equity_stocks = instrument_filter.filter_equity_stocks(instruments_df)
                
                # Get major stocks
                major_stocks_df = instrument_filter.get_major_stocks_criteria(self.filtered_equity_stocks)
                self.major_stocks = set(major_stocks_df['tradingsymbol'].tolist())
                
                logging.info(f"Loaded {len(self.major_stocks)} major stocks from {len(self.filtered_equity_stocks)} equity stocks")
                
        except Exception as e:
            logging.error(f"Error loading major stocks: {e}")
    
    def get_stock_sector(self, symbol: str) -> Optional[str]:
        """
        Enhanced sector classification - only for major stocks
        
        Args:
            symbol: Stock symbol
            
        Returns:
            str: Sector name or None if not found or not a major stock
        """
        symbol_upper = symbol.upper()
        
        # Only classify major stocks
        if symbol_upper not in self.major_stocks:
            logging.debug(f"Stock {symbol_upper} not in major stocks list")
            return None
        
        # Use parent class method for actual sector classification
        return super().get_stock_sector(symbol)
    
    def get_major_stocks(self) -> Set[str]:
        """
        Get set of major stocks for analysis
        
        Returns:
            Set[str]: Set of major stock symbols
        """
        return self.major_stocks.copy()
    
    def get_filtered_equity_stocks(self) -> pd.DataFrame:
        """
        Get filtered equity stocks dataframe
        
        Returns:
            DataFrame: Filtered equity stocks
        """
        return self.filtered_equity_stocks.copy()
    
    def get_instrument_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of instrument types
        
        Returns:
            Dict: Count of each instrument type
        """
        return self.instrument_breakdown.copy()
    
    def create_analysis_datasets(self, output_dir: str = "analysis_datasets"):
        """
        Create and export analysis datasets
        
        Args:
            output_dir: Output directory for datasets
        """
        if self.filtered_equity_stocks.empty:
            logging.warning("No filtered equity stocks available for dataset creation")
            return
        
        # Create and export datasets
        analysis_datasets.export_datasets(self.filtered_equity_stocks, output_dir)
    
    def get_dataset_summary(self) -> Dict[str, Dict]:
        """
        Get summary of analysis datasets
        
        Returns:
            Dict: Summary of all datasets
        """
        if self.filtered_equity_stocks.empty:
            return {}
        
        return analysis_datasets.get_dataset_summary(self.filtered_equity_stocks)
    
    def get_trading_stocks(self) -> List[str]:
        """
        Get stocks suitable for trading analysis
        
        Returns:
            List[str]: List of trading stocks
        """
        if self.filtered_equity_stocks.empty:
            return []
        
        trading_dataset = analysis_datasets.create_trading_dataset(self.filtered_equity_stocks)
        return trading_dataset.get('liquid_stocks', [])
    
    def get_portfolio_stocks(self) -> List[str]:
        """
        Get stocks suitable for portfolio analysis
        
        Returns:
            List[str]: List of portfolio stocks
        """
        if self.filtered_equity_stocks.empty:
            return []
        
        portfolio_dataset = analysis_datasets.create_portfolio_dataset(self.filtered_equity_stocks)
        return portfolio_dataset.get('core_holdings', [])
    
    def get_sector_stocks_enhanced(self, sector: str) -> List[str]:
        """
        Get stocks in a sector (enhanced version)
        
        Args:
            sector: Sector name
            
        Returns:
            List[str]: List of stocks in the sector
        """
        # First try existing sector classification
        existing_stocks = super().get_sector_stocks(sector)
        
        # Filter to only major stocks
        major_stocks_in_sector = [stock for stock in existing_stocks if stock in self.major_stocks]
        
        logging.info(f"Sector {sector}: {len(major_stocks_in_sector)} major stocks out of {len(existing_stocks)} total")
        
        return major_stocks_in_sector
    
    def get_sector_performance_data(self, sector: str) -> Dict[str, str]:
        """
        Get sector performance tracking data
        
        Args:
            sector: Sector name
            
        Returns:
            Dict: Sector performance data
        """
        primary_index = self.get_primary_sector_index(sector)
        sector_stocks = self.get_sector_stocks_enhanced(sector)
        
        return {
            'sector': sector,
            'primary_index': primary_index,
            'stock_count': len(sector_stocks),
            'stocks': sector_stocks,
            'display_name': self.get_sector_display_name(sector)
        }
    
    def analyze_portfolio_sectors(self, portfolio_stocks: List[str]) -> Dict[str, Dict]:
        """
        Analyze sector allocation of a portfolio
        
        Args:
            portfolio_stocks: List of stock symbols in portfolio
            
        Returns:
            Dict: Portfolio sector analysis
        """
        sector_allocation = {}
        unclassified_stocks = []
        
        for stock in portfolio_stocks:
            sector = self.get_stock_sector(stock)
            if sector:
                if sector not in sector_allocation:
                    sector_allocation[sector] = {
                        'stocks': [],
                        'count': 0,
                        'percentage': 0
                    }
                sector_allocation[sector]['stocks'].append(stock)
                sector_allocation[sector]['count'] += 1
            else:
                unclassified_stocks.append(stock)
        
        # Calculate percentages
        total_stocks = len(portfolio_stocks)
        for sector_data in sector_allocation.values():
            sector_data['percentage'] = (sector_data['count'] / total_stocks) * 100
        
        return {
            'sector_allocation': sector_allocation,
            'unclassified_stocks': unclassified_stocks,
            'total_stocks': total_stocks,
            'classified_stocks': total_stocks - len(unclassified_stocks)
        }
    
    def get_sector_rotation_analysis(self) -> Dict[str, Dict]:
        """
        Get data for sector rotation analysis
        
        Returns:
            Dict: Sector rotation analysis data
        """
        sectors = self.get_all_sectors()
        sector_data = {}
        
        for sector_info in sectors:
            sector_code = sector_info['code']
            sector_data[sector_code] = self.get_sector_performance_data(sector_code)
        
        return {
            'sectors': sector_data,
            'total_sectors': len(sectors),
            'analysis_type': 'sector_rotation'
        }
    
    def export_enhanced_data(self, output_dir: str = "enhanced_sector_data"):
        """
        Export enhanced sector data
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export major stocks
        major_stocks_file = output_path / "major_stocks.json"
        with open(major_stocks_file, 'w') as f:
            json.dump({
                'major_stocks': list(self.major_stocks),
                'total_count': len(self.major_stocks),
                'created_at': pd.Timestamp.now().isoformat()
            }, f, indent=2)
        
        # Export instrument breakdown
        breakdown_file = output_path / "instrument_breakdown.json"
        with open(breakdown_file, 'w') as f:
            json.dump(self.instrument_breakdown, f, indent=2)
        
        # Export filtered equity stocks
        if not self.filtered_equity_stocks.empty:
            stocks_file = output_path / "filtered_equity_stocks.csv"
            self.filtered_equity_stocks.to_csv(stocks_file, index=False)
        
        # Export sector performance data
        sector_performance = {}
        for sector_info in self.get_all_sectors():
            sector_code = sector_info['code']
            sector_performance[sector_code] = self.get_sector_performance_data(sector_code)
        
        performance_file = output_path / "sector_performance.json"
        with open(performance_file, 'w') as f:
            json.dump(sector_performance, f, indent=2)
        
        logging.info(f"Exported enhanced sector data to {output_path}")
    
    def get_system_summary(self) -> Dict[str, any]:
        """
        Get comprehensive system summary including enhanced features
        
        Returns:
            Dict: Complete system summary
        """
        summary = {
            'base_sectors': len(self.sector_mappings),
            'total_stocks': len(self.stock_to_sector),
            'major_stocks': len(self.major_stocks),
            'filtered_equity_stocks': len(self.filtered_equity_stocks),
            'instrument_breakdown': self.instrument_breakdown,
            'enhanced_features': {
                'major_stocks_filtering': True,
                'instrument_filtering': True,
                'analysis_datasets': True,
                'portfolio_analysis': True,
                'sector_rotation': True
            }
        }
        
        return summary
    
    def get_real_time_sector_performance(self, sector: str) -> Dict[str, any]:
        """
        Get real-time sector performance data (requires market data integration)
        
        Args:
            sector: Sector name
            
        Returns:
            Dict: Real-time sector performance data
        """
        try:
            # This would integrate with real market data providers
            # For now, return structured data format
            sector_stocks = self.get_sector_stocks_enhanced(sector)
            primary_index = self.get_primary_sector_index(sector)
            
            return {
                'sector': sector,
                'primary_index': primary_index,
                'stock_count': len(sector_stocks),
                'stocks': sector_stocks,
                'display_name': self.get_sector_display_name(sector),
                'last_updated': pd.Timestamp.now().isoformat(),
                'performance_metrics': {
                    'sector_return_1d': None,  # Would be calculated from real data
                    'sector_return_1w': None,
                    'sector_return_1m': None,
                    'sector_return_3m': None,
                    'sector_return_1y': None,
                    'volatility': None,
                    'beta': None,
                    'sharpe_ratio': None
                },
                'top_performers': [],  # Would be populated from real data
                'bottom_performers': [],  # Would be populated from real data
                'sector_trend': 'neutral'  # bullish, bearish, neutral
            }
        except Exception as e:
            logging.error(f"Error getting real-time sector performance for {sector}: {e}")
            return None
    
    def get_sector_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate sector correlation matrix (requires historical data)
        
        Returns:
            DataFrame: Sector correlation matrix
        """
        try:
            sectors = list(self.sector_mappings.keys())
            correlation_matrix = pd.DataFrame(index=sectors, columns=sectors)
            
            # This would be populated with real correlation data
            # For now, return empty matrix with proper structure
            for sector1 in sectors:
                for sector2 in sectors:
                    if sector1 == sector2:
                        correlation_matrix.loc[sector1, sector2] = 1.0
                    else:
                        correlation_matrix.loc[sector1, sector2] = None  # Would be calculated
            
            return correlation_matrix
        except Exception as e:
            logging.error(f"Error calculating sector correlation matrix: {e}")
            return pd.DataFrame()
    
    def get_sector_risk_metrics(self, sector: str) -> Dict[str, any]:
        """
        Calculate sector-specific risk metrics
        
        Args:
            sector: Sector name
            
        Returns:
            Dict: Risk metrics for the sector
        """
        try:
            sector_stocks = self.get_sector_stocks_enhanced(sector)
            
            return {
                'sector': sector,
                'stock_count': len(sector_stocks),
                'risk_metrics': {
                    'sector_volatility': None,  # Would be calculated from historical data
                    'sector_beta': None,
                    'sector_var_95': None,  # Value at Risk 95%
                    'sector_var_99': None,  # Value at Risk 99%
                    'sector_max_drawdown': None,
                    'sector_sharpe_ratio': None,
                    'sector_sortino_ratio': None,
                    'sector_information_ratio': None
                },
                'concentration_risk': {
                    'top_5_weight': None,  # Weight of top 5 stocks
                    'top_10_weight': None,  # Weight of top 10 stocks
                    'herfindahl_index': None  # Concentration index
                },
                'liquidity_metrics': {
                    'avg_daily_volume': None,
                    'avg_turnover_ratio': None,
                    'bid_ask_spread': None
                }
            }
        except Exception as e:
            logging.error(f"Error calculating risk metrics for sector {sector}: {e}")
            return None

# Global instance
enhanced_sector_classifier = EnhancedSectorClassifier() 