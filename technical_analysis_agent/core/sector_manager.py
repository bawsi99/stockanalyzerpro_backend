#!/usr/bin/env python3
"""
Sector Manager Utility

This script provides utilities to manage sector data including:
- Adding stocks to sectors
- Removing stocks from sectors
- Creating new sectors
- Listing all sectors and their stocks
- Validating sector data
"""

import json
import argparse
import logging
from pathlib import Path
from agents.sector import SectorClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SectorManager:
    def __init__(self):
        self.classifier = SectorClassifier()
    
    def list_sectors(self):
        """List all sectors with their stock counts."""
        print("\n=== SECTOR OVERVIEW ===")
        sectors = self.classifier.get_all_sectors()
        
        for sector_info in sectors:
            sector_code = sector_info['code']
            display_name = sector_info['name']
            stocks = self.classifier.get_sector_stocks(sector_code)
            print(f"{sector_code:20} | {display_name:35} | {len(stocks):3} stocks")
        
        print(f"\nTotal sectors: {len(sectors)}")
        print(f"Total stocks: {len(self.classifier.stock_to_sector)}")
    
    def list_sector_stocks(self, sector: str):
        """List all stocks in a specific sector."""
        if sector not in self.classifier.sector_mappings:
            print(f"Error: Sector '{sector}' not found")
            return
        
        stocks = self.classifier.get_sector_stocks(sector)
        display_name = self.classifier.get_sector_display_name(sector)
        
        print(f"\n=== {sector.upper()} SECTOR ({display_name}) ===")
        print(f"Primary Index: {self.classifier.get_primary_sector_index(sector)}")
        print(f"All Indices: {', '.join(self.classifier.get_sector_indices(sector))}")
        print(f"Stock Count: {len(stocks)}")
        print("\nStocks:")
        
        for i, stock in enumerate(sorted(stocks), 1):
            print(f"  {i:2d}. {stock}")
    
    def add_stock(self, stock: str, sector: str):
        """Add a stock to a sector."""
        stock = stock.upper()
        sector = sector.upper()
        
        if sector not in self.classifier.sector_mappings:
            print(f"Error: Sector '{sector}' not found")
            return
        
        if self.classifier.add_stock_to_sector(stock, sector):
            print(f"Successfully added {stock} to {sector} sector")
            # Save the updated sector data
            if self.classifier.save_sector_data(sector):
                print(f"Saved updated {sector} sector data to file")
            else:
                print(f"Warning: Failed to save {sector} sector data")
        else:
            print(f"Failed to add {stock} to {sector} sector")
    
    def remove_stock(self, stock: str, sector: str):
        """Remove a stock from a sector."""
        stock = stock.upper()
        sector = sector.upper()
        
        if sector not in self.classifier.sector_mappings:
            print(f"Error: Sector '{sector}' not found")
            return
        
        if self.classifier.remove_stock_from_sector(stock, sector):
            print(f"Successfully removed {stock} from {sector} sector")
            # Save the updated sector data
            if self.classifier.save_sector_data(sector):
                print(f"Saved updated {sector} sector data to file")
            else:
                print(f"Warning: Failed to save {sector} sector data")
        else:
            print(f"Failed to remove {stock} from {sector} sector")
    
    def find_stock_sector(self, stock: str):
        """Find which sector a stock belongs to."""
        stock = stock.upper()
        sector = self.classifier.get_stock_sector(stock)
        
        if sector:
            display_name = self.classifier.get_sector_display_name(sector)
            print(f"Stock '{stock}' belongs to: {sector} ({display_name})")
        else:
            print(f"Stock '{stock}' not found in any sector")
    
    def create_sector(self, sector_code: str, display_name: str, primary_index: str, description: str = ""):
        """Create a new sector."""
        sector_code = sector_code.upper()
        
        if sector_code in self.classifier.sector_mappings:
            print(f"Error: Sector '{sector_code}' already exists")
            return
        
        # Create the sector data
        sector_data = {
            'sector_code': sector_code,
            'display_name': display_name,
            'indices': [primary_index],
            'primary_index': primary_index,
            'stocks': [],
            'description': description
        }
        
        # Save to JSON file
        json_file = Path("sector_category") / f"{sector_code.lower()}.json"
        try:
            with open(json_file, 'w') as f:
                json.dump(sector_data, f, indent=2)
            
            # Reload the classifier to include the new sector
            self.classifier.reload_sector_data()
            print(f"Successfully created sector '{sector_code}' ({display_name})")
            print(f"Saved to: {json_file}")
        except Exception as e:
            print(f"Error creating sector: {e}")
    
    def validate_data(self):
        """Validate sector data for consistency."""
        print("\n=== VALIDATING SECTOR DATA ===")
        
        issues = []
        
        # Check for duplicate stocks across sectors
        stock_counts = {}
        for stock, sector in self.classifier.stock_to_sector.items():
            if stock in stock_counts:
                stock_counts[stock].append(sector)
            else:
                stock_counts[stock] = [sector]
        
        for stock, sectors in stock_counts.items():
            if len(sectors) > 1:
                issues.append(f"Stock '{stock}' appears in multiple sectors: {', '.join(sectors)}")
        
        # Check for empty sectors
        for sector_code, data in self.classifier.sector_mappings.items():
            if not data['stocks']:
                issues.append(f"Sector '{sector_code}' has no stocks")
        
        # Check for missing required fields
        for sector_code, data in self.classifier.sector_mappings.items():
            required_fields = ['display_name', 'primary_index', 'indices']
            for field in required_fields:
                if not data.get(field):
                    issues.append(f"Sector '{sector_code}' missing required field: {field}")
        
        if issues:
            print("Found the following issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No issues found. Sector data is consistent.")
        
        return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description="Sector Manager Utility")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List sectors command
    subparsers.add_parser('list', help='List all sectors')
    
    # List sector stocks command
    list_stocks_parser = subparsers.add_parser('stocks', help='List stocks in a sector')
    list_stocks_parser.add_argument('sector', help='Sector code (e.g., BANKING, IT)')
    
    # Add stock command
    add_parser = subparsers.add_parser('add', help='Add a stock to a sector')
    add_parser.add_argument('stock', help='Stock symbol')
    add_parser.add_argument('sector', help='Sector code')
    
    # Remove stock command
    remove_parser = subparsers.add_parser('remove', help='Remove a stock from a sector')
    remove_parser.add_argument('stock', help='Stock symbol')
    remove_parser.add_argument('sector', help='Sector code')
    
    # Find stock command
    find_parser = subparsers.add_parser('find', help='Find which sector a stock belongs to')
    find_parser.add_argument('stock', help='Stock symbol')
    
    # Create sector command
    create_parser = subparsers.add_parser('create', help='Create a new sector')
    create_parser.add_argument('sector_code', help='Sector code (e.g., NEW_SECTOR)')
    create_parser.add_argument('display_name', help='Display name (e.g., "New Sector")')
    create_parser.add_argument('primary_index', help='Primary index symbol')
    create_parser.add_argument('--description', default='', help='Sector description')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate sector data for consistency')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = SectorManager()
    
    if args.command == 'list':
        manager.list_sectors()
    elif args.command == 'stocks':
        manager.list_sector_stocks(args.sector.upper())
    elif args.command == 'add':
        manager.add_stock(args.stock, args.sector)
    elif args.command == 'remove':
        manager.remove_stock(args.stock, args.sector)
    elif args.command == 'find':
        manager.find_stock_sector(args.stock)
    elif args.command == 'create':
        manager.create_sector(args.sector_code, args.display_name, args.primary_index, args.description)
    elif args.command == 'validate':
        manager.validate_data()

if __name__ == "__main__":
    main() 