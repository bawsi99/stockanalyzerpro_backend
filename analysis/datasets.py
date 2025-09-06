#!/usr/bin/env python3
"""
Analysis Datasets for creating purpose-specific stock datasets.
This module creates different datasets for trading, portfolio, and sector analysis.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set
import json
from pathlib import Path

class AnalysisDatasets:
    """
    Creates purpose-specific datasets for different analysis types
    """
    
    def __init__(self):
        self.nifty_indices = {
            'NIFTY 50': {
                'description': 'Top 50 stocks by market cap',
                'analysis_purpose': 'Core portfolio, market benchmark'
            },
            'NIFTY NEXT 50': {
                'description': 'Next 50 stocks by market cap',
                'analysis_purpose': 'Mid-cap exposure, growth stocks'
            },
            'NIFTY BANK': {
                'description': 'Banking sector stocks',
                'analysis_purpose': 'Banking sector analysis'
            },
            'NIFTY IT': {
                'description': 'IT sector stocks',
                'analysis_purpose': 'IT sector analysis'
            },
            'NIFTY PHARMA': {
                'description': 'Pharmaceutical stocks',
                'analysis_purpose': 'Pharma sector analysis'
            },
            'NIFTY AUTO': {
                'description': 'Automobile stocks',
                'analysis_purpose': 'Auto sector analysis'
            },
            'NIFTY FMCG': {
                'description': 'FMCG stocks',
                'analysis_purpose': 'FMCG sector analysis'
            },
            'NIFTY ENERGY': {
                'description': 'Energy sector stocks',
                'analysis_purpose': 'Energy sector analysis'
            },
            'NIFTY METAL': {
                'description': 'Metals & Mining stocks',
                'analysis_purpose': 'Metal sector analysis'
            },
            'NIFTY REALTY': {
                'description': 'Real Estate stocks',
                'analysis_purpose': 'Realty sector analysis'
            },
            'NIFTY OIL AND GAS': {
                'description': 'Oil & Gas stocks',
                'analysis_purpose': 'Oil & Gas sector analysis'
            },
            'NIFTY HEALTHCARE': {
                'description': 'Healthcare stocks',
                'analysis_purpose': 'Healthcare sector analysis'
            },
            'NIFTY CONSR DURBL': {
                'description': 'Consumer Durables stocks',
                'analysis_purpose': 'Consumer Durables analysis'
            },
            'NIFTY MEDIA': {
                'description': 'Media & Entertainment stocks',
                'analysis_purpose': 'Media sector analysis'
            },
            'NIFTY INFRA': {
                'description': 'Infrastructure stocks',
                'analysis_purpose': 'Infrastructure analysis'
            },
            'NIFTY CONSUMPTION': {
                'description': 'Consumption stocks',
                'analysis_purpose': 'Consumption analysis'
            },
            'NIFTY SERV SECTOR': {
                'description': 'Services sector stocks',
                'analysis_purpose': 'Services sector analysis'
            }
        }
    
    def create_trading_dataset(self, equity_stocks_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create dataset optimized for trading analysis
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            
        Returns:
            Dict: Trading-focused stock lists
        """
        if equity_stocks_df.empty:
            return {}
        
        trading_dataset = {
            'liquid_stocks': self._get_liquid_stocks(equity_stocks_df),
            'sector_leaders': self._get_sector_leaders(equity_stocks_df),
            'momentum_stocks': self._get_momentum_stocks(equity_stocks_df),
            'high_volume_stocks': self._get_high_volume_stocks(equity_stocks_df)
        }
        
        logging.info("Created trading dataset:")
        for category, stocks in trading_dataset.items():
            logging.info(f"  {category}: {len(stocks)} stocks")
        
        return trading_dataset
    
    def create_portfolio_dataset(self, equity_stocks_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create dataset optimized for portfolio analysis
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            
        Returns:
            Dict: Portfolio-focused stock lists
        """
        if equity_stocks_df.empty:
            return {}
        
        portfolio_dataset = {
            'core_holdings': self._get_core_holdings(equity_stocks_df),
            'diversified_stocks': self._get_diversified_stocks(equity_stocks_df),
            'large_cap': self._get_large_cap_stocks(equity_stocks_df),
            'mid_cap': self._get_mid_cap_stocks(equity_stocks_df),
            'small_cap': self._get_small_cap_stocks(equity_stocks_df)
        }
        
        logging.info("Created portfolio dataset:")
        for category, stocks in portfolio_dataset.items():
            logging.info(f"  {category}: {len(stocks)} stocks")
        
        return portfolio_dataset
    
    def create_sector_dataset(self, equity_stocks_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create dataset optimized for sector analysis
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            
        Returns:
            Dict: Sector-focused stock lists
        """
        if equity_stocks_df.empty:
            return {}
        
        sector_dataset = {}
        
        # Create sector lists based on NIFTY indices
        for index_name in self.nifty_indices.keys():
            sector_stocks = self._get_sector_stocks(equity_stocks_df, index_name)
            if sector_stocks:
                sector_dataset[index_name.lower().replace(' ', '_')] = sector_stocks
        
        logging.info("Created sector dataset:")
        for sector, stocks in sector_dataset.items():
            logging.info(f"  {sector}: {len(stocks)} stocks")
        
        return sector_dataset
    
    def _get_liquid_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get highly liquid stocks for day trading"""
        # Focus on major stocks with good liquidity
        liquid_stocks = equity_stocks_df[
            # Remove illiquid suffixes
            (~equity_stocks_df['tradingsymbol'].str.contains('-BE|-SM|-ST|-BZ', case=False, na=False)) &
            # Reasonable symbol length
            (equity_stocks_df['tradingsymbol'].str.len() >= 3) &
            (equity_stocks_df['tradingsymbol'].str.len() <= 15)
        ]['tradingsymbol'].tolist()
        
        return liquid_stocks[:500]  # Top 500 liquid stocks
    
    def _get_sector_leaders(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get sector leader stocks"""
        # This would ideally be based on market cap and sector leadership
        # For now, return a subset of liquid stocks
        return self._get_liquid_stocks(equity_stocks_df)[:100]
    
    def _get_momentum_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get stocks suitable for momentum strategies"""
        # This would ideally be based on beta and volatility
        # For now, return a subset of liquid stocks
        return self._get_liquid_stocks(equity_stocks_df)[:200]
    
    def _get_high_volume_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get high volume stocks"""
        # This would ideally be based on actual volume data
        # For now, return a subset of liquid stocks
        return self._get_liquid_stocks(equity_stocks_df)[:300]
    
    def _get_core_holdings(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get core portfolio holdings (NIFTY 50 equivalent)"""
        # This would ideally be based on actual NIFTY 50 constituents
        # For now, return top liquid stocks
        return self._get_liquid_stocks(equity_stocks_df)[:50]
    
    def _get_diversified_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get diversified stock selection"""
        # This would ideally be based on sector diversification
        # For now, return a larger set of liquid stocks
        return self._get_liquid_stocks(equity_stocks_df)[:200]
    
    def _get_large_cap_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get large cap stocks"""
        # This would ideally be based on market cap
        # For now, return top stocks
        return self._get_liquid_stocks(equity_stocks_df)[:100]
    
    def _get_mid_cap_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get mid cap stocks"""
        # This would ideally be based on market cap
        # For now, return middle range stocks
        liquid_stocks = self._get_liquid_stocks(equity_stocks_df)
        return liquid_stocks[100:300]
    
    def _get_small_cap_stocks(self, equity_stocks_df: pd.DataFrame) -> List[str]:
        """Get small cap stocks"""
        # This would ideally be based on market cap
        # For now, return remaining stocks
        liquid_stocks = self._get_liquid_stocks(equity_stocks_df)
        return liquid_stocks[300:500]
    
    def _get_sector_stocks(self, equity_stocks_df: pd.DataFrame, index_name: str) -> List[str]:
        """Get stocks for a specific sector/index"""
        # This would ideally be based on actual index constituents
        # For now, return a subset based on sector keywords
        sector_keywords = self._get_sector_keywords(index_name)
        
        if not sector_keywords:
            return []
        
        sector_stocks = []
        for _, row in equity_stocks_df.iterrows():
            symbol = str(row['tradingsymbol']).upper()
            name = str(row['name']).upper()
            
            if any(keyword in symbol or keyword in name for keyword in sector_keywords):
                sector_stocks.append(row['tradingsymbol'])
        
        return sector_stocks[:50]  # Limit to top 50 per sector
    
    def _get_sector_keywords(self, index_name: str) -> List[str]:
        """Get keywords for sector identification"""
        sector_keywords = {
            'NIFTY BANK': ['BANK', 'FINANCE', 'CREDIT', 'INSURANCE'],
            'NIFTY IT': ['TECH', 'SOFTWARE', 'IT', 'DIGITAL'],
            'NIFTY PHARMA': ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE'],
            'NIFTY AUTO': ['AUTO', 'MOTOR', 'VEHICLE', 'TYRE'],
            'NIFTY FMCG': ['FOOD', 'BEVERAGE', 'CONSUMER', 'FMCG'],
            'NIFTY ENERGY': ['POWER', 'ENERGY', 'ELECTRICITY'],
            'NIFTY METAL': ['STEEL', 'METAL', 'MINING', 'ALUMINIUM'],
            'NIFTY REALTY': ['REAL', 'ESTATE', 'PROPERTY', 'CONSTRUCTION'],
            'NIFTY OIL AND GAS': ['OIL', 'GAS', 'PETROLEUM', 'REFINERY'],
            'NIFTY HEALTHCARE': ['HEALTHCARE', 'HOSPITAL', 'MEDICAL'],
            'NIFTY CONSR DURBL': ['ELECTRONICS', 'APPLIANCE', 'DURABLE'],
            'NIFTY MEDIA': ['MEDIA', 'ENTERTAINMENT', 'BROADCASTING'],
            'NIFTY INFRA': ['INFRASTRUCTURE', 'INFRA', 'ENGINEERING'],
            'NIFTY CONSUMPTION': ['CONSUMER', 'RETAIL', 'SHOPPING'],
            'NIFTY SERV SECTOR': ['TELECOM', 'COMMUNICATION', 'TRANSPORT']
        }
        
        return sector_keywords.get(index_name, [])
    
    def export_datasets(self, equity_stocks_df: pd.DataFrame, output_dir: str = "analysis_datasets"):
        """
        Export all analysis datasets to JSON files
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            output_dir: Output directory for datasets
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create all datasets
        trading_dataset = self.create_trading_dataset(equity_stocks_df)
        portfolio_dataset = self.create_portfolio_dataset(equity_stocks_df)
        sector_dataset = self.create_sector_dataset(equity_stocks_df)
        
        # Export datasets
        datasets = {
            'trading': trading_dataset,
            'portfolio': portfolio_dataset,
            'sector': sector_dataset,
            'metadata': {
                'total_stocks': len(equity_stocks_df),
                'created_at': pd.Timestamp.now().isoformat(),
                'description': 'Analysis-ready stock datasets'
            }
        }
        
        # Save to JSON file
        output_file = output_path / "analysis_datasets.json"
        with open(output_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        logging.info(f"Exported analysis datasets to {output_file}")
        
        # Also save individual datasets
        for dataset_name, dataset_data in datasets.items():
            if dataset_name != 'metadata':
                individual_file = output_path / f"{dataset_name}_dataset.json"
                with open(individual_file, 'w') as f:
                    json.dump(dataset_data, f, indent=2)
                logging.info(f"Exported {dataset_name} dataset to {individual_file}")
    
    def get_dataset_summary(self, equity_stocks_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get summary of all datasets
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            
        Returns:
            Dict: Summary of all datasets
        """
        trading_dataset = self.create_trading_dataset(equity_stocks_df)
        portfolio_dataset = self.create_portfolio_dataset(equity_stocks_df)
        sector_dataset = self.create_sector_dataset(equity_stocks_df)
        
        summary = {
            'trading_datasets': {
                name: len(stocks) for name, stocks in trading_dataset.items()
            },
            'portfolio_datasets': {
                name: len(stocks) for name, stocks in portfolio_dataset.items()
            },
            'sector_datasets': {
                name: len(stocks) for name, stocks in sector_dataset.items()
            },
            'total_equity_stocks': len(equity_stocks_df)
        }
        
        return summary

# Global instance
analysis_datasets = AnalysisDatasets() 