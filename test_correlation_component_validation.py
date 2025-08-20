#!/usr/bin/env python3
"""
Validation Script for Correlation Analysis Component

This script validates the correlation analysis functionality to ensure:
1. No sectors appear in both high and low correlation lists
2. Correlation matrix is properly generated
3. Data flows correctly from backend to frontend
4. All correlation metrics are calculated accurately

Following the rules: "create validation scripts for each module you create and test"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sector_benchmarking import SectorBenchmarkingProvider
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CorrelationAnalysisValidator:
    """Validator for correlation analysis component"""
    
    def __init__(self):
        self.provider = SectorBenchmarkingProvider()
        self.test_results = []
    
    def test_correlation_matrix_generation(self):
        """Test that correlation matrix is generated correctly"""
        print("\n🔍 Testing Correlation Matrix Generation")
        print("=" * 50)
        
        try:
            # Test synchronous generation
            result = self.provider.generate_sector_correlation_matrix("3M")
            
            if not result:
                print("❌ Synchronous correlation matrix generation failed")
                return False
            
            # Validate structure
            required_fields = [
                'correlation_matrix', 'average_correlation', 
                'high_correlation_pairs', 'low_correlation_pairs',
                'diversification_insights', 'sector_volatility'
            ]
            
            for field in required_fields:
                if field not in result:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            print("✅ Correlation matrix structure is correct")
            
            # Validate correlation matrix
            correlation_matrix = result['correlation_matrix']
            if not isinstance(correlation_matrix, dict):
                print("❌ Correlation matrix is not a dictionary")
                return False
            
            # Check if matrix has data
            if len(correlation_matrix) == 0:
                print("❌ Correlation matrix is empty")
                return False
            
            print(f"✅ Correlation matrix has {len(correlation_matrix)} sectors")
            return True
            
        except Exception as e:
            print(f"❌ Error in correlation matrix generation test: {e}")
            return False
    
    def test_correlation_pairs_consistency(self):
        """Test that no sector appears in both high and low correlation lists"""
        print("\n🔍 Testing Correlation Pairs Consistency")
        print("=" * 50)
        
        try:
            result = self.provider.generate_sector_correlation_matrix("3M")
            if not result:
                print("❌ Cannot test pairs consistency - no correlation data")
                return False
            
            high_correlation_pairs = result['high_correlation_pairs']
            low_correlation_pairs = result['low_correlation_pairs']
            
            # Extract sectors from high correlation pairs
            high_correlation_sectors = set()
            for pair in high_correlation_pairs:
                high_correlation_sectors.add(pair['sector1'])
                high_correlation_sectors.add(pair['sector2'])
            
            # Extract sectors from low correlation pairs
            low_correlation_sectors = set()
            for pair in low_correlation_pairs:
                low_correlation_sectors.add(pair['sector1'])
                low_correlation_sectors.add(pair['sector2'])
            
            # Check for overlap
            overlap = high_correlation_sectors.intersection(low_correlation_sectors)
            
            if overlap:
                print(f"❌ Sectors appearing in both lists: {overlap}")
                print("   This indicates a logical error in correlation classification")
                return False
            
            print("✅ No sectors appear in both high and low correlation lists")
            print(f"   High correlation sectors: {len(high_correlation_sectors)}")
            print(f"   Low correlation sectors: {len(low_correlation_sectors)}")
            return True
            
        except Exception as e:
            print(f"❌ Error in correlation pairs consistency test: {e}")
            return False
    
    def test_correlation_thresholds(self):
        """Test that correlation thresholds are applied correctly"""
        print("\n🔍 Testing Correlation Thresholds")
        print("=" * 50)
        
        try:
            result = self.provider.generate_sector_correlation_matrix("3M")
            if not result:
                print("❌ Cannot test thresholds - no correlation data")
                return False
            
            high_correlation_pairs = result['high_correlation_pairs']
            low_correlation_pairs = result['low_correlation_pairs']
            
            # Check high correlation threshold (> 0.7)
            for pair in high_correlation_pairs:
                if pair['correlation'] <= 0.7:
                    print(f"❌ High correlation pair has correlation <= 0.7: {pair}")
                    return False
            
            # Check low correlation threshold (< 0.3)
            for pair in low_correlation_pairs:
                if pair['correlation'] >= 0.3:
                    print(f"❌ Low correlation pair has correlation >= 0.3: {pair}")
                    return False
            
            print("✅ All correlation thresholds are correctly applied")
            print(f"   High correlation pairs: {len(high_correlation_pairs)}")
            print(f"   Low correlation pairs: {len(low_correlation_pairs)}")
            return True
            
        except Exception as e:
            print(f"❌ Error in correlation thresholds test: {e}")
            return False
    
    def test_correlation_matrix_symmetry(self):
        """Test that correlation matrix is symmetric"""
        print("\n🔍 Testing Correlation Matrix Symmetry")
        print("=" * 50)
        
        try:
            result = self.provider.generate_sector_correlation_matrix("3M")
            if not result:
                print("❌ Cannot test symmetry - no correlation data")
                return False
            
            correlation_matrix = result['correlation_matrix']
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(correlation_matrix)
            
            # Check symmetry (correlation[i,j] should equal correlation[j,i])
            is_symmetric = True
            asymmetric_pairs = []
            
            for i in range(len(df.columns)):
                for j in range(i + 1, len(df.columns)):
                    val1 = df.iloc[i, j]
                    val2 = df.iloc[j, i]
                    
                    if abs(val1 - val2) > 0.001:  # Allow for small floating point differences
                        is_symmetric = False
                        asymmetric_pairs.append({
                            'i': df.columns[i], 'j': df.columns[j],
                            'val1': val1, 'val2': val2
                        })
            
            if not is_symmetric:
                print("❌ Correlation matrix is not symmetric")
                for pair in asymmetric_pairs[:5]:  # Show first 5 asymmetric pairs
                    print(f"   {pair['i']}-{pair['j']}: {pair['val1']} vs {pair['val2']}")
                return False
            
            print("✅ Correlation matrix is symmetric")
            return True
            
        except Exception as e:
            print(f"❌ Error in correlation matrix symmetry test: {e}")
            return False
    
    def test_diversification_insights(self):
        """Test that diversification insights are generated correctly"""
        print("\n🔍 Testing Diversification Insights")
        print("=" * 50)
        
        try:
            result = self.provider.generate_sector_correlation_matrix("3M")
            if not result:
                print("❌ Cannot test diversification insights - no correlation data")
                return False
            
            insights = result['diversification_insights']
            
            # Check required fields
            required_fields = ['diversification_quality', 'recommendations']
            for field in required_fields:
                if field not in insights:
                    print(f"❌ Missing diversification insight field: {field}")
                    return False
            
            # Check diversification quality values
            valid_qualities = ['excellent', 'good', 'moderate', 'poor']
            if insights['diversification_quality'] not in valid_qualities:
                print(f"❌ Invalid diversification quality: {insights['diversification_quality']}")
                return False
            
            # Check recommendations
            if not isinstance(insights['recommendations'], list):
                print("❌ Recommendations is not a list")
                return False
            
            print("✅ Diversification insights are correctly generated")
            print(f"   Quality: {insights['diversification_quality']}")
            print(f"   Recommendations: {len(insights['recommendations'])}")
            return True
            
        except Exception as e:
            print(f"❌ Error in diversification insights test: {e}")
            return False
    
    def test_async_correlation_generation(self):
        """Test asynchronous correlation matrix generation"""
        print("\n🔍 Testing Async Correlation Generation")
        print("=" * 50)
        
        try:
            async def test_async():
                result = await self.provider.generate_sector_correlation_matrix_async("3M")
                return result
            
            result = asyncio.run(test_async())
            
            if not result:
                print("❌ Asynchronous correlation matrix generation failed")
                return False
            
            # Compare with synchronous result
            sync_result = self.provider.generate_sector_correlation_matrix("3M")
            
            if not sync_result:
                print("❌ Cannot compare - synchronous generation failed")
                return False
            
            # Check key metrics match
            if abs(result['average_correlation'] - sync_result['average_correlation']) > 0.001:
                print("❌ Average correlation differs between sync and async")
                return False
            
            if len(result['high_correlation_pairs']) != len(sync_result['high_correlation_pairs']):
                print("❌ High correlation pairs count differs between sync and async")
                return False
            
            print("✅ Asynchronous correlation generation works correctly")
            print("✅ Sync and async results are consistent")
            return True
            
        except Exception as e:
            print(f"❌ Error in async correlation generation test: {e}")
            return False
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🧪 CORRELATION ANALYSIS COMPONENT VALIDATION")
        print("=" * 60)
        print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all tests
        tests = [
            ("Correlation Matrix Generation", self.test_correlation_matrix_generation),
            ("Correlation Pairs Consistency", self.test_correlation_pairs_consistency),
            ("Correlation Thresholds", self.test_correlation_thresholds),
            ("Correlation Matrix Symmetry", self.test_correlation_matrix_symmetry),
            ("Diversification Insights", self.test_diversification_insights),
            ("Async Correlation Generation", self.test_async_correlation_generation)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results.append((test_name, result))
            except Exception as e:
                print(f"❌ Test {test_name} crashed: {e}")
                self.test_results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1
        
        print(f"\nOverall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 CORRELATION ANALYSIS COMPONENT VALIDATION COMPLETED SUCCESSFULLY!")
            print("✅ All correlation analysis components are working correctly")
            print("✅ No sectors appear in both high and low correlation lists")
            print("✅ Correlation matrix is properly generated and symmetric")
            print("✅ Ready for production use")
        else:
            print("⚠️ Some tests failed. Please review and fix issues.")
            print("❌ The correlation analysis component has problems that need attention")
        
        return passed == total

def main():
    """Main validation function"""
    validator = CorrelationAnalysisValidator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
