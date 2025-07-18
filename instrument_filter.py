#!/usr/bin/env python3
"""
Instrument Filter for separating different types of financial instruments.
This module filters equity stocks from bonds, SDLs, indices, and other instruments.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import re

class InstrumentFilter:
    """
    Filters and categorizes financial instruments from zerodha_instruments.csv
    """
    
    def __init__(self):
        self.instrument_categories = {
            "INDICES": {
                "description": "Market indices for benchmarking",
                "patterns": ["NIFTY", "SENSEX", "INDIA VIX", "BEES"],
                "analysis_purpose": "Market sentiment, sector performance"
            },
            
            "EQUITY_STOCKS": {
                "description": "Common stocks for trading",
                "patterns": ["EQ", "NSE", "no special suffixes"],
                "analysis_purpose": "Stock analysis, sector rotation"
            },
            
            "GOVERNMENT_BONDS": {
                "description": "Government securities and SDLs",
                "patterns": ["-SG", "SDL"],
                "analysis_purpose": "Fixed income, yield analysis"
            },
            
            "CORPORATE_BONDS": {
                "description": "Corporate debt instruments",
                "patterns": ["-N1", "-N2", "-N3", "-N4", "-N5", "-N6", "-N7", "-N8", "-N9", "-N0"],
                "analysis_purpose": "Credit analysis, yield spreads"
            },
            
            "GOLD_BONDS": {
                "description": "Sovereign Gold Bonds",
                "patterns": ["-GB", "SGB"],
                "analysis_purpose": "Gold exposure, inflation hedge"
            },
            
            "INFRASTRUCTURE_BONDS": {
                "description": "Infrastructure project bonds",
                "patterns": ["NHAI", "IREDA", "PFCL", "IRFC", "RECL", "NTPC"],
                "analysis_purpose": "Infrastructure investment"
            },
            
            "OTHER_DEBT": {
                "description": "Other debt instruments",
                "patterns": ["-SG", "-GB", "-N"],
                "analysis_purpose": "Fixed income analysis"
            }
        }
    
    def load_instruments_from_csv(self, csv_path: str = "zerodha_instruments.csv") -> pd.DataFrame:
        """
        Load instruments from CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame: Raw instruments data
        """
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} instruments from {csv_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading instruments from {csv_path}: {e}")
            return pd.DataFrame()
    
    def classify_instrument_type(self, symbol: str, name: str, instrument_type: str, segment: str) -> str:
        """
        Classify instrument by type
        
        Args:
            symbol: Trading symbol
            name: Instrument name
            instrument_type: Instrument type from CSV
            segment: Market segment
            
        Returns:
            str: Instrument category
        """
        symbol_upper = str(symbol).upper()
        name_upper = str(name).upper()
        
        # Check for indices first
        if self._is_index(symbol_upper, name_upper):
            return "INDICES"
        
        # Check for gold bonds
        if self._is_gold_bond(symbol_upper):
            return "GOLD_BONDS"
        
        # Check for government bonds (SDL)
        if self._is_government_bond(symbol_upper, name_upper):
            return "GOVERNMENT_BONDS"
        
        # Check for infrastructure bonds
        if self._is_infrastructure_bond(symbol_upper, name_upper):
            return "INFRASTRUCTURE_BONDS"
        
        # Check for corporate bonds
        if self._is_corporate_bond(symbol_upper):
            return "CORPORATE_BONDS"
        
        # Check for other debt instruments
        if self._is_other_debt(symbol_upper):
            return "OTHER_DEBT"
        
        # Default to equity stocks
        return "EQUITY_STOCKS"
    
    def _is_index(self, symbol: str, name: str) -> bool:
        """Check if instrument is an index"""
        index_keywords = ["NIFTY", "SENSEX", "INDIA VIX", "BEES"]
        return any(keyword in symbol or keyword in name for keyword in index_keywords)
    
    def _is_gold_bond(self, symbol: str) -> bool:
        """Check if instrument is a gold bond"""
        return "-GB" in symbol or "SGB" in symbol
    
    def _is_government_bond(self, symbol: str, name: str) -> bool:
        """Check if instrument is a government bond (SDL)"""
        return "-SG" in symbol or "SDL" in name
    
    def _is_infrastructure_bond(self, symbol: str, name: str) -> bool:
        """Check if instrument is an infrastructure bond"""
        infra_keywords = ["NHAI", "IREDA", "PFCL", "IRFC", "RECL", "NTPC"]
        return any(keyword in symbol or keyword in name for keyword in infra_keywords)
    
    def _is_corporate_bond(self, symbol: str) -> bool:
        """Check if instrument is a corporate bond"""
        bond_patterns = [f"-N{i}" for i in range(10)]  # -N0 to -N9
        return any(pattern in symbol for pattern in bond_patterns)
    
    def _is_other_debt(self, symbol: str) -> bool:
        """Check if instrument is other debt"""
        debt_patterns = ["-SG", "-GB", "-N"]
        return any(pattern in symbol for pattern in debt_patterns)
    
    def filter_equity_stocks(self, instruments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter only equity stocks from instruments dataframe
        
        Args:
            instruments_df: Raw instruments dataframe
            
        Returns:
            DataFrame: Filtered equity stocks only
        """
        if instruments_df.empty:
            return pd.DataFrame()
        
        # Add instrument category column
        instruments_df['instrument_category'] = instruments_df.apply(
            lambda row: self.classify_instrument_type(
                row['tradingsymbol'], 
                row['name'], 
                row['instrument_type'], 
                row['segment']
            ), 
            axis=1
        )
        
        # Filter for equity stocks only
        equity_stocks = instruments_df[
            (instruments_df['instrument_category'] == 'EQUITY_STOCKS') &
            (instruments_df['instrument_type'] == 'EQ') &
            (instruments_df['exchange'] == 'NSE') &
            (instruments_df['tradingsymbol'].notna()) &
            (instruments_df['name'].notna()) &
            (~instruments_df['tradingsymbol'].str.contains('NIFTY|SENSEX|INDIA VIX|BEES', case=False, na=False)) &
            (instruments_df['tradingsymbol'].str.len() <= 20)  # Most stock symbols are short
        ].copy()
        
        logging.info(f"Filtered {len(equity_stocks)} equity stocks from {len(instruments_df)} total instruments")
        
        return equity_stocks
    
    def get_instrument_breakdown(self, instruments_df: pd.DataFrame) -> Dict[str, int]:
        """
        Get breakdown of instrument types
        
        Args:
            instruments_df: Raw instruments dataframe
            
        Returns:
            Dict: Count of each instrument type
        """
        if instruments_df.empty:
            return {}
        
        # Add instrument category if not present
        if 'instrument_category' not in instruments_df.columns:
            instruments_df['instrument_category'] = instruments_df.apply(
                lambda row: self.classify_instrument_type(
                    row['tradingsymbol'], 
                    row['name'], 
                    row['instrument_type'], 
                    row['segment']
                ), 
                axis=1
            )
        
        breakdown = instruments_df['instrument_category'].value_counts().to_dict()
        
        logging.info("Instrument breakdown:")
        for category, count in breakdown.items():
            logging.info(f"  {category}: {count}")
        
        return breakdown
    
    def get_major_stocks_criteria(self, equity_stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply criteria to identify major liquid stocks
        
        Args:
            equity_stocks_df: Filtered equity stocks dataframe
            
        Returns:
            DataFrame: Major stocks only
        """
        if equity_stocks_df.empty:
            return pd.DataFrame()
        
        # Filter criteria for major stocks
        major_stocks = equity_stocks_df[
            # Remove stocks with suffixes indicating illiquidity
            (~equity_stocks_df['tradingsymbol'].str.contains('-BE|-SM|-ST|-BZ', case=False, na=False)) &
            # Remove very short symbols (likely not stocks)
            (equity_stocks_df['tradingsymbol'].str.len() >= 3) &
            # Remove stocks with numbers at start (likely bonds)
            (~equity_stocks_df['tradingsymbol'].str.match(r'^\d+', na=False))
        ].copy()
        
        logging.info(f"Identified {len(major_stocks)} major stocks from {len(equity_stocks_df)} equity stocks")
        
        return major_stocks
    
    def export_filtered_data(self, instruments_df: pd.DataFrame, output_path: str = "filtered_equity_stocks.csv"):
        """
        Export filtered equity stocks to CSV
        
        Args:
            instruments_df: Filtered equity stocks dataframe
            output_path: Output file path
        """
        try:
            instruments_df.to_csv(output_path, index=False)
            logging.info(f"Exported {len(instruments_df)} equity stocks to {output_path}")
        except Exception as e:
            logging.error(f"Error exporting to {output_path}: {e}")

# Global instance
instrument_filter = InstrumentFilter() 