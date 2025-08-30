#!/usr/bin/env python3
"""
test_production_endpoints.py

Test script to validate the endpoints of the combined production service.
This script sends requests to various endpoints to ensure they're working correctly.

Usage:
    python test_production_endpoints.py [--host localhost] [--port 8000]
"""

import argparse
import requests
import json
import sys
import time
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table

console = Console()

class EndpointTester:
    """Tests the endpoints of the combined production service."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self.data_url = f"{self.base_url}/data"
        self.analysis_url = f"{self.base_url}/analysis"
        self.results = []
    
    def test_endpoint(self, endpoint: str, method: str = "GET", payload: Dict[str, Any] = None, description: str = ""):
        """Test a specific endpoint and record the result."""
        full_url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(full_url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(full_url, json=payload, timeout=10)
            else:
                return {"endpoint": endpoint, "status": "SKIPPED", "reason": f"Method {method} not supported", "time": 0}
            
            elapsed = time.time() - start_time
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "time": elapsed,
                "description": description
            }
            
            if 200 <= response.status_code < 300:
                result["status"] = "SUCCESS"
                try:
                    result["response"] = response.json()
                except:
                    result["response"] = "Non-JSON response"
            else:
                result["status"] = "FAILED"
                result["reason"] = f"Status code: {response.status_code}"
                
            self.results.append(result)
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "endpoint": endpoint,
                "method": method,
                "status": "ERROR",
                "reason": str(e),
                "time": elapsed,
                "description": description
            }
            self.results.append(result)
            return result
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        return self.test_endpoint("/health", "GET", description="Health check endpoint")
    
    def test_data_endpoints(self):
        """Test various endpoints of the data service."""
        endpoints = [
            {"path": "/", "method": "GET", "description": "Data service root endpoint"},
            {"path": "/health", "method": "GET", "description": "Data service health check"},
            {"path": "/market/status", "method": "GET", "description": "Current market status"}
        ]
        
        results = []
        for endpoint in endpoints:
            path = endpoint["path"]
            full_path = f"/data{path}"
            if "params" in endpoint:
                query_params = "&".join([f"{k}={v}" for k, v in endpoint["params"].items()])
                full_path = f"{full_path}?{query_params}"
            
            result = self.test_endpoint(
                full_path, 
                endpoint["method"], 
                description=endpoint["description"]
            )
            results.append(result)
        
        return results
    
    def test_analysis_endpoints(self):
        """Test various endpoints of the analysis service."""
        endpoints = [
            {"path": "/", "method": "GET", "description": "Analysis service root endpoint"},
            {"path": "/health", "method": "GET", "description": "Analysis service health check"},
            {"path": "/analyze", "method": "POST", "payload": {"stock_symbol": "NIFTY50", "exchange": "NSE", "period": 365, "interval": "day"}, 
             "description": "Technical analysis for NIFTY50"}
        ]
        
        results = []
        for endpoint in endpoints:
            path = endpoint["path"]
            full_path = f"/analysis{path}"
            payload = endpoint.get("payload")
            
            result = self.test_endpoint(
                full_path, 
                endpoint["method"], 
                payload=payload,
                description=endpoint["description"]
            )
            results.append(result)
        
        return results
    
    def run_all_tests(self):
        """Run all endpoint tests."""
        console.print("[bold]🧪 Testing Production Service Endpoints[/bold]")
        console.print(f"Base URL: {self.base_url}")
        console.print()
        
        # Test health endpoint
        console.print("[bold]Testing Health Endpoint...[/bold]")
        health_result = self.test_health_endpoint()
        self._print_result(health_result)
        
        # Test data service endpoints
        console.print("\n[bold]Testing Data Service Endpoints...[/bold]")
        data_results = self.test_data_endpoints()
        for result in data_results:
            self._print_result(result)
        
        # Test analysis service endpoints
        console.print("\n[bold]Testing Analysis Service Endpoints...[/bold]")
        analysis_results = self.test_analysis_endpoints()
        for result in analysis_results:
            self._print_result(result)
        
        # Summary
        self._print_summary()
    
    def _print_result(self, result):
        """Print a nicely formatted test result."""
        status = result["status"]
        endpoint = result["endpoint"]
        
        if status == "SUCCESS":
            status_str = "[green]SUCCESS[/green]"
        elif status == "FAILED":
            status_str = "[red]FAILED[/red]"
        else:
            status_str = "[yellow]ERROR[/yellow]"
        
        console.print(f"  {status_str} - {endpoint} - {result['description']}")
        
        if status == "SUCCESS":
            console.print(f"    Response time: {result['time']:.2f}s")
        else:
            console.print(f"    Reason: {result.get('reason', 'Unknown error')}")
    
    def _print_summary(self):
        """Print a summary of all test results."""
        success = sum(1 for r in self.results if r["status"] == "SUCCESS")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        error = sum(1 for r in self.results if r["status"] == "ERROR")
        total = len(self.results)
        
        table = Table(title="Endpoint Test Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        table.add_row("Success", str(success), f"{success/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Failed", str(failed), f"{failed/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Error", str(error), f"{error/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Total", str(total), "100%" if total > 0 else "0%")
        
        console.print("\n[bold]Test Summary[/bold]")
        console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Stock Analyzer Pro Production Endpoints")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port for the service")
    
    args = parser.parse_args()
    
    try:
        tester = EndpointTester(host=args.host, port=args.port)
        tester.run_all_tests()
    except KeyboardInterrupt:
        console.print("\n[bold red]Test interrupted by user[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error running tests: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
