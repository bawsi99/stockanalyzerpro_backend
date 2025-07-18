#!/usr/bin/env python3
"""
Comprehensive test suite for the sector classification system.
Tests all components including basic classifier, enhanced classifier, and data integrity.
"""

import unittest
import tempfile
import json
import pandas as pd
from pathlib import Path
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sector_classifier import SectorClassifier
from enhanced_sector_classifier import EnhancedSectorClassifier
from sector_manager import SectorManager

class TestSectorClassifier(unittest.TestCase):
    """Test cases for the basic SectorClassifier."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.sector_folder = Path(self.temp_dir) / "test_sectors"
        self.sector_folder.mkdir()
        
        # Create test sector data
        self.test_sectors = {
            'BANKING': {
                'sector_code': 'BANKING',
                'display_name': 'Banking & Financial Services',
                'indices': ['NIFTY BANK'],
                'primary_index': 'NIFTY BANK',
                'stocks': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK'],
                'description': 'Banking and financial services companies'
            },
            'IT': {
                'sector_code': 'IT',
                'display_name': 'Information Technology',
                'indices': ['NIFTY IT'],
                'primary_index': 'NIFTY IT',
                'stocks': ['TCS', 'INFY', 'WIPRO', 'HCLTECH'],
                'description': 'Information technology companies'
            }
        }
        
        # Write test sector files
        for sector_code, data in self.test_sectors.items():
            sector_file = self.sector_folder / f"{sector_code.lower()}.json"
            with open(sector_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        self.classifier = SectorClassifier(str(self.sector_folder))
    
    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sector_loading(self):
        """Test that sectors are loaded correctly."""
        self.assertEqual(len(self.classifier.sector_mappings), 2)
        self.assertIn('BANKING', self.classifier.sector_mappings)
        self.assertIn('IT', self.classifier.sector_mappings)
    
    def test_stock_to_sector_mapping(self):
        """Test stock to sector mapping."""
        self.assertEqual(self.classifier.get_stock_sector('HDFCBANK'), 'BANKING')
        self.assertEqual(self.classifier.get_stock_sector('TCS'), 'IT')
        self.assertIsNone(self.classifier.get_stock_sector('UNKNOWN'))
    
    def test_sector_display_names(self):
        """Test sector display names."""
        self.assertEqual(self.classifier.get_sector_display_name('BANKING'), 'Banking & Financial Services')
        self.assertEqual(self.classifier.get_sector_display_name('IT'), 'Information Technology')
    
    def test_sector_indices(self):
        """Test sector indices."""
        self.assertEqual(self.classifier.get_primary_sector_index('BANKING'), 'NIFTY BANK')
        self.assertEqual(self.classifier.get_primary_sector_index('IT'), 'NIFTY IT')
    
    def test_sector_stocks(self):
        """Test getting stocks in a sector."""
        banking_stocks = self.classifier.get_sector_stocks('BANKING')
        self.assertEqual(len(banking_stocks), 4)
        self.assertIn('HDFCBANK', banking_stocks)
        self.assertIn('ICICIBANK', banking_stocks)
    
    def test_all_sectors(self):
        """Test getting all sectors."""
        all_sectors = self.classifier.get_all_sectors()
        self.assertEqual(len(all_sectors), 2)
        sector_codes = [s['code'] for s in all_sectors]
        self.assertIn('BANKING', sector_codes)
        self.assertIn('IT', sector_codes)
    
    def test_data_integrity(self):
        """Test data integrity validation."""
        issues = self.classifier.validate_data_integrity()
        self.assertEqual(len(issues['errors']), 0)
        self.assertEqual(len(issues['warnings']), 0)
    
    def test_sector_statistics(self):
        """Test sector statistics."""
        stats = self.classifier.get_sector_statistics()
        self.assertEqual(stats['total_sectors'], 2)
        self.assertEqual(stats['total_stocks'], 8)
        self.assertIn('BANKING', stats['sector_details'])
        self.assertIn('IT', stats['sector_details'])

class TestEnhancedSectorClassifier(unittest.TestCase):
    """Test cases for the EnhancedSectorClassifier."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.sector_folder = Path(self.temp_dir) / "test_sectors"
        self.sector_folder.mkdir()
        
        # Create test sector data
        test_sectors = {
            'BANKING': {
                'sector_code': 'BANKING',
                'display_name': 'Banking & Financial Services',
                'indices': ['NIFTY BANK'],
                'primary_index': 'NIFTY BANK',
                'stocks': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK'],
                'description': 'Banking and financial services companies'
            }
        }
        
        # Write test sector files
        for sector_code, data in test_sectors.items():
            sector_file = self.sector_folder / f"{sector_code.lower()}.json"
            with open(sector_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Create mock instruments CSV
        self.instruments_csv = Path(self.temp_dir) / "zerodha_instruments.csv"
        instruments_data = {
            'instrument_token': [123456, 123457, 123458, 123459],
            'tradingsymbol': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK'],
            'name': ['HDFC Bank', 'ICICI Bank', 'SBI', 'Axis Bank'],
            'exchange': ['NSE', 'NSE', 'NSE', 'NSE'],
            'instrument_type': ['EQ', 'EQ', 'EQ', 'EQ'],
            'segment': ['NSE', 'NSE', 'NSE', 'NSE'],
            'expiry': ['', '', '', ''],
            'strike': [0, 0, 0, 0],
            'tick_size': [0.05, 0.05, 0.05, 0.05],
            'lot_size': [1, 1, 1, 1]
        }
        instruments_df = pd.DataFrame(instruments_data)
        instruments_df.to_csv(self.instruments_csv, index=False)
        
        # Mock the instrument_filter to use our test CSV
        import instrument_filter
        original_load_method = instrument_filter.instrument_filter.load_instruments_from_csv
        
        def mock_load_instruments():
            return instruments_df
        
        instrument_filter.instrument_filter.load_instruments_from_csv = mock_load_instruments
        
        self.enhanced_classifier = EnhancedSectorClassifier(str(self.sector_folder))
        
        # Restore original method
        instrument_filter.instrument_filter.load_instruments_from_csv = original_load_method
    
    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_enhanced_sector_classification(self):
        """Test enhanced sector classification with major stocks filtering."""
        # Test that major stocks are properly filtered
        major_stocks = self.enhanced_classifier.get_major_stocks()
        self.assertIn('HDFCBANK', major_stocks)
        self.assertIn('ICICIBANK', major_stocks)
    
    def test_sector_stocks_enhanced(self):
        """Test enhanced sector stocks retrieval."""
        banking_stocks = self.enhanced_classifier.get_sector_stocks_enhanced('BANKING')
        self.assertEqual(len(banking_stocks), 4)
        self.assertIn('HDFCBANK', banking_stocks)
    
    def test_system_summary(self):
        """Test system summary."""
        summary = self.enhanced_classifier.get_system_summary()
        self.assertEqual(summary['base_sectors'], 1)
        self.assertEqual(summary['total_stocks'], 4)
        self.assertIn('enhanced_features', summary)
    
    def test_real_time_sector_performance(self):
        """Test real-time sector performance data structure."""
        performance = self.enhanced_classifier.get_real_time_sector_performance('BANKING')
        self.assertIsNotNone(performance)
        self.assertEqual(performance['sector'], 'BANKING')
        self.assertEqual(performance['primary_index'], 'NIFTY BANK')
        self.assertIn('performance_metrics', performance)
        self.assertIn('sector_trend', performance)
    
    def test_sector_correlation_matrix(self):
        """Test sector correlation matrix."""
        correlation_matrix = self.enhanced_classifier.get_sector_correlation_matrix()
        self.assertIsInstance(correlation_matrix, pd.DataFrame)
        self.assertEqual(len(correlation_matrix), 1)  # Only one sector in test data
    
    def test_sector_risk_metrics(self):
        """Test sector risk metrics."""
        risk_metrics = self.enhanced_classifier.get_sector_risk_metrics('BANKING')
        self.assertIsNotNone(risk_metrics)
        self.assertEqual(risk_metrics['sector'], 'BANKING')
        self.assertIn('risk_metrics', risk_metrics)
        self.assertIn('concentration_risk', risk_metrics)
        self.assertIn('liquidity_metrics', risk_metrics)

class TestSectorManager(unittest.TestCase):
    """Test cases for the SectorManager."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.sector_folder = Path(self.temp_dir) / "test_sectors"
        self.sector_folder.mkdir()
        
        # Create test sector data
        test_sectors = {
            'BANKING': {
                'sector_code': 'BANKING',
                'display_name': 'Banking & Financial Services',
                'indices': ['NIFTY BANK'],
                'primary_index': 'NIFTY BANK',
                'stocks': ['HDFCBANK', 'ICICIBANK'],
                'description': 'Banking and financial services companies'
            }
        }
        
        # Write test sector files
        for sector_code, data in test_sectors.items():
            sector_file = self.sector_folder / f"{sector_code.lower()}.json"
            with open(sector_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Mock the sector folder in SectorManager
        import sector_manager
        original_init = sector_manager.SectorManager.__init__
        
        def mock_init(self):
            self.classifier = SectorClassifier(str(self.sector_folder))
        
        sector_manager.SectorManager.__init__ = mock_init
        
        self.manager = SectorManager()
        
        # Restore original method
        sector_manager.SectorManager.__init__ = original_init
    
    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_find_stock_sector(self):
        """Test finding stock sector."""
        # This would test the CLI functionality
        # For now, test the underlying method
        sector = self.manager.classifier.get_stock_sector('HDFCBANK')
        self.assertEqual(sector, 'BANKING')
    
    def test_add_stock_to_sector(self):
        """Test adding stock to sector."""
        success = self.manager.classifier.add_stock_to_sector('NEWBANK', 'BANKING')
        self.assertTrue(success)
        
        # Verify stock was added
        sector = self.manager.classifier.get_stock_sector('NEWBANK')
        self.assertEqual(sector, 'BANKING')
    
    def test_remove_stock_from_sector(self):
        """Test removing stock from sector."""
        success = self.manager.classifier.remove_stock_from_sector('HDFCBANK', 'BANKING')
        self.assertTrue(success)
        
        # Verify stock was removed
        sector = self.manager.classifier.get_stock_sector('HDFCBANK')
        self.assertIsNone(sector)

class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def setUp(self):
        """Set up performance test data."""
        self.classifier = SectorClassifier()
    
    def test_lookup_performance(self):
        """Test lookup performance."""
        import time
        
        # Test multiple lookups
        start_time = time.time()
        for _ in range(1000):
            self.classifier.get_stock_sector('RELIANCE')
        end_time = time.time()
        
        # Should complete 1000 lookups in under 1 second
        self.assertLess(end_time - start_time, 1.0)
    
    def test_caching_effectiveness(self):
        """Test caching effectiveness."""
        # First call should be slower (cache miss)
        import time
        
        start_time = time.time()
        self.classifier.get_stock_sector('RELIANCE')
        first_call_time = time.time() - start_time
        
        # Second call should be faster (cache hit)
        start_time = time.time()
        self.classifier.get_stock_sector('RELIANCE')
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster
        self.assertLess(second_call_time, first_call_time * 0.5)

def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("Running Performance Benchmark...")
    
    classifier = SectorClassifier()
    
    # Test data loading performance
    import time
    start_time = time.time()
    classifier.reload_sector_data()
    load_time = time.time() - start_time
    print(f"Data loading time: {load_time:.3f} seconds")
    
    # Test lookup performance
    test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    start_time = time.time()
    for _ in range(10000):
        for stock in test_stocks:
            classifier.get_stock_sector(stock)
    lookup_time = time.time() - start_time
    print(f"50,000 lookups time: {lookup_time:.3f} seconds")
    print(f"Average lookup time: {lookup_time/50000*1000:.3f} milliseconds")
    
    # Test memory usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {memory_usage:.1f} MB")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmark
    print("\n" + "="*50)
    run_performance_benchmark() 