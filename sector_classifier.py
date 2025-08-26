import pandas as pd
import logging
from typing import Dict, Optional, List
import json
import os
from pathlib import Path
# Redis cache functionality will be used instead of local cache
import time
from functools import wraps

class RateLimiter:
    """Simple rate limiter for API calls and data operations."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_call(self) -> bool:
        """Check if a call can be made within rate limits."""
        now = time.time()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.can_call():
            time.sleep(0.1)  # Wait 100ms before retrying

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logging.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

class SectorClassifier:
    """
    Classifies stocks into sectors and provides sector-specific index mappings.
    Now reads sector data from JSON files in the sector_category folder.
    Implements singleton pattern for optimization.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to prevent duplicate data loading."""
        if cls._instance is None:
            cls._instance = super(SectorClassifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, sector_folder: str = "sector_category"):
        """
        Initialize the SectorClassifier with data from JSON files.
        
        Args:
            sector_folder: Path to the folder containing sector JSON files
        """
        # Skip initialization if already initialized (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.sector_folder = Path(sector_folder)
        self.sector_mappings = {}
        self.stock_to_sector = {}
        self.rate_limiter = RateLimiter(max_calls=1000, time_window=60)  # 1000 calls per minute
        
        # Load all sector data from JSON files
        self._load_sector_data()
        
        # Create reverse mapping for quick lookup
        self._build_stock_to_sector_mapping()
        
        # Mark as initialized
        self._initialized = True
    
    def _load_sector_data(self):
        """Load sector data from all JSON files in the sector folder."""
        if not self.sector_folder.exists():
            logging.error(f"Sector folder not found: {self.sector_folder}")
            return
        
        for json_file in self.sector_folder.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    sector_data = json.load(f)
                
                sector_code = sector_data.get('sector_code')
                if sector_code:
                    self.sector_mappings[sector_code] = {
                        'indices': sector_data.get('indices', []),
                        'primary_index': sector_data.get('primary_index'),
                        'display_name': sector_data.get('display_name'),
                        'stocks': sector_data.get('stocks', []),
                        'description': sector_data.get('description', '')
                    }
                    logging.info(f"Loaded sector data for {sector_code}: {len(sector_data.get('stocks', []))} stocks")
                else:
                    logging.warning(f"No sector_code found in {json_file}")
                    
            except Exception as e:
                logging.error(f"Error loading sector data from {json_file}: {e}")
        
        logging.info(f"Loaded {len(self.sector_mappings)} sectors from JSON files")
    
    def _build_stock_to_sector_mapping(self):
        """Build reverse mapping from stock symbols to sectors."""
        for sector, data in self.sector_mappings.items():
            for stock in data['stocks']:
                self.stock_to_sector[stock] = sector
        
        logging.info(f"Built stock-to-sector mapping for {len(self.stock_to_sector)} stocks")
    
    def get_stock_sector(self, symbol: str) -> Optional[str]:
        """
        Get the sector for a given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            str: Sector name or None if not found
        """
        return self.stock_to_sector.get(symbol.upper())
    
    def get_sector_display_name(self, sector: str) -> Optional[str]:
        """
        Get the display name for a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            str: Display name or None if not found
        """
        return self.sector_mappings.get(sector, {}).get('display_name')
    
    def get_sector_indices(self, sector: str) -> List[str]:
        """
        Get all available indices for a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List[str]: List of index symbols
        """
        return self.sector_mappings.get(sector, {}).get('indices', [])
    
    def get_primary_sector_index(self, sector: str) -> Optional[str]:
        """
        Get the primary index for a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            str: Primary index symbol or None
        """
        return self.sector_mappings.get(sector, {}).get('primary_index')
    
    def get_all_sectors(self) -> List[Dict[str, str]]:
        """
        Get list of all available sectors with display names.
        
        Returns:
            List[Dict[str, str]]: List of sectors with code and display name
        """
        return [
            {
                'code': sector,
                'name': data['display_name'],
                'primary_index': data['primary_index']
            }
            for sector, data in self.sector_mappings.items()
        ]
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get all stocks in a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List[str]: List of stock symbols
        """
        return self.sector_mappings.get(sector, {}).get('stocks', [])
    
    def add_stock_to_sector(self, stock_symbol: str, sector: str) -> bool:
        """
        Add a stock to a sector. This will update both the sector data and the reverse mapping.
        
        Args:
            stock_symbol: Stock symbol to add
            sector: Sector to add the stock to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.sector_mappings:
            logging.error(f"Sector {sector} not found")
            return False
        
        stock_symbol = stock_symbol.upper()
        if stock_symbol not in self.sector_mappings[sector]['stocks']:
            self.sector_mappings[sector]['stocks'].append(stock_symbol)
            self.stock_to_sector[stock_symbol] = sector
            logging.info(f"Added {stock_symbol} to sector {sector}")
            return True
        else:
            logging.warning(f"Stock {stock_symbol} already exists in sector {sector}")
            return False
    
    def remove_stock_from_sector(self, stock_symbol: str, sector: str) -> bool:
        """
        Remove a stock from a sector.
        
        Args:
            stock_symbol: Stock symbol to remove
            sector: Sector to remove the stock from
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.sector_mappings:
            logging.error(f"Sector {sector} not found")
            return False
        
        stock_symbol = stock_symbol.upper()
        if stock_symbol in self.sector_mappings[sector]['stocks']:
            self.sector_mappings[sector]['stocks'].remove(stock_symbol)
            if stock_symbol in self.stock_to_sector:
                del self.stock_to_sector[stock_symbol]
            logging.info(f"Removed {stock_symbol} from sector {sector}")
            return True
        else:
            logging.warning(f"Stock {stock_symbol} not found in sector {sector}")
            return False
    
    def save_sector_data(self, sector: str) -> bool:
        """
        Save sector data back to its JSON file.
        
        Args:
            sector: Sector to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if sector not in self.sector_mappings:
            logging.error(f"Sector {sector} not found")
            return False
        
        try:
            sector_data = self.sector_mappings[sector]
            json_data = {
                'sector_code': sector,
                'display_name': sector_data['display_name'],
                'indices': sector_data['indices'],
                'primary_index': sector_data['primary_index'],
                'stocks': sector_data['stocks'],
                'description': sector_data.get('description', '')
            }
            
            json_file = self.sector_folder / f"{sector.lower()}.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logging.info(f"Saved sector data for {sector} to {json_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving sector data for {sector}: {e}")
            return False
    
    def reload_sector_data(self):
        """Reload all sector data from JSON files."""
        self.sector_mappings.clear()
        self.stock_to_sector.clear()
        self._load_sector_data()
        self._build_stock_to_sector_mapping()
        logging.info("Reloaded all sector data from JSON files")
    
    def validate_data_integrity(self) -> Dict[str, List[str]]:
        """
        Comprehensive data validation to ensure sector data integrity.
        
        Returns:
            Dict[str, List[str]]: Dictionary with validation results and issues
        """
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check for duplicate stocks across sectors
        stock_counts = {}
        for stock, sector in self.stock_to_sector.items():
            if stock in stock_counts:
                stock_counts[stock].append(sector)
            else:
                stock_counts[stock] = [sector]
        
        for stock, sectors in stock_counts.items():
            if len(sectors) > 1:
                issues['errors'].append(f"Stock '{stock}' appears in multiple sectors: {', '.join(sectors)}")
        
        # Check for empty sectors
        for sector_code, data in self.sector_mappings.items():
            if not data['stocks']:
                issues['warnings'].append(f"Sector '{sector_code}' has no stocks")
        
        # Check for missing required fields
        for sector_code, data in self.sector_mappings.items():
            required_fields = ['display_name', 'primary_index', 'indices']
            for field in required_fields:
                if not data.get(field):
                    issues['errors'].append(f"Sector '{sector_code}' missing required field: {field}")
        
        # Check for invalid stock symbols
        invalid_symbols = []
        for stock in self.stock_to_sector.keys():
            if len(stock) < 2 or len(stock) > 20:
                invalid_symbols.append(stock)
            elif not stock.replace('-', '').replace('_', '').isalnum():
                invalid_symbols.append(stock)
        
        if invalid_symbols:
            issues['warnings'].append(f"Found {len(invalid_symbols)} potentially invalid stock symbols")
        
        # Check for sector consistency
        total_stocks = len(self.stock_to_sector)
        total_in_sectors = sum(len(data['stocks']) for data in self.sector_mappings.values())
        
        if total_stocks != total_in_sectors:
            issues['errors'].append(f"Stock count mismatch: {total_stocks} in mapping vs {total_in_sectors} in sectors")
        
        # Add summary info
        issues['info'].extend([
            f"Total sectors: {len(self.sector_mappings)}",
            f"Total stocks: {total_stocks}",
            f"Average stocks per sector: {total_stocks / len(self.sector_mappings):.1f}" if self.sector_mappings else "0"
        ])
        
        return issues
    
    def get_sector_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about the sector classification system.
        
        Returns:
            Dict: Statistics about sectors and stocks
        """
        stats = {
            'total_sectors': len(self.sector_mappings),
            'total_stocks': len(self.stock_to_sector),
            'sector_details': {},
            'largest_sectors': [],
            'smallest_sectors': []
        }
        
        # Calculate sector details
        sector_sizes = []
        for sector_code, data in self.sector_mappings.items():
            stock_count = len(data['stocks'])
            sector_sizes.append((sector_code, stock_count))
            
            stats['sector_details'][sector_code] = {
                'name': data['display_name'],
                'stock_count': stock_count,
                'primary_index': data['primary_index'],
                'indices': data['indices']
            }
        
        # Sort sectors by size
        sector_sizes.sort(key=lambda x: x[1], reverse=True)
        stats['largest_sectors'] = sector_sizes[:5]
        stats['smallest_sectors'] = sector_sizes[-5:]
        
        return stats
    
    def export_sector_mappings(self, filepath: str = "sector_mappings.json"):
        """
        Export sector mappings to JSON file for frontend use.
        
        Args:
            filepath: Path to save the JSON file
        """
        export_data = {
            'sectors': self.sector_mappings,
            'stock_to_sector': self.stock_to_sector,
            'sector_list': self.get_all_sectors(),
            'metadata': {
                'total_sectors': len(self.sector_mappings),
                'total_stocks': len(self.stock_to_sector),
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"Sector mappings exported to {filepath}")

# Global instance
sector_classifier = SectorClassifier() 