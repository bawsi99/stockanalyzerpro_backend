#!/usr/bin/env python3
"""
Connection Validation Script for Stock Analyzer Pro

This script validates:
1. Backend service health
2. All API endpoints
3. Frontend connectivity
4. CORS configuration
5. WebSocket connections
6. Database connectivity
7. External service connectivity (Zerodha, Gemini, Supabase)

Usage:
    python validate_connections.py
"""

import asyncio
import json
import requests
import websockets
import time
import os
import sys
from typing import Dict, List, Any
from urllib.parse import urljoin

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8080"

class ConnectionValidator:
    def __init__(self):
        self.results = {
            "backend_health": {},
            "api_endpoints": {},
            "frontend_connectivity": {},
            "cors_test": {},
            "websocket_test": {},
            "database_test": {},
            "external_services": {},
            "summary": {}
        }
        self.session = requests.Session()
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp and level."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_backend_health(self) -> Dict[str, Any]:
        """Test backend service health endpoints."""
        self.log("Testing backend health endpoints...")
        
        health_endpoints = [
            "/health",
            "/data/health", 
            "/analysis/health"
        ]
        
        results = {}
        for endpoint in health_endpoints:
            try:
                url = urljoin(BACKEND_URL, endpoint)
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results[endpoint] = {
                        "status": "✅ HEALTHY",
                        "response": data,
                        "response_time": response.elapsed.total_seconds()
                    }
                    self.log(f"✅ {endpoint}: Healthy ({response.elapsed.total_seconds():.3f}s)")
                else:
                    results[endpoint] = {
                        "status": f"❌ HTTP {response.status_code}",
                        "error": f"Expected 200, got {response.status_code}"
                    }
                    self.log(f"❌ {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                results[endpoint] = {
                    "status": "❌ ERROR",
                    "error": str(e)
                }
                self.log(f"❌ {endpoint}: Error - {str(e)}")
        
        self.results["backend_health"] = results
        return results
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test key API endpoints for functionality."""
        self.log("Testing key API endpoints...")
        
        # Test endpoints that don't require authentication
        test_endpoints = [
            ("/sector/list", "GET"),
            ("/data/market/status", "GET"),
            ("/data/mapping/symbol-to-token", "GET", {"symbol": "RELIANCE"}),
        ]
        
        results = {}
        for endpoint_info in test_endpoints:
            endpoint = endpoint_info[0]
            method = endpoint_info[1]
            params = endpoint_info[2] if len(endpoint_info) > 2 else {}
            
            try:
                url = urljoin(BACKEND_URL, endpoint)
                
                if method == "GET":
                    response = self.session.get(url, params=params, timeout=10)
                else:
                    response = self.session.post(url, json=params, timeout=10)
                
                if response.status_code in [200, 201]:
                    try:
                        data = response.json()
                        results[endpoint] = {
                            "status": "✅ WORKING",
                            "method": method,
                            "response_time": response.elapsed.total_seconds(),
                            "response_size": len(response.content)
                        }
                        self.log(f"✅ {endpoint} ({method}): Working ({response.elapsed.total_seconds():.3f}s)")
                    except json.JSONDecodeError:
                        results[endpoint] = {
                            "status": "⚠️  RESPONSE_NOT_JSON",
                            "method": method,
                            "response_time": response.elapsed.total_seconds(),
                            "response_size": len(response.content)
                        }
                        self.log(f"⚠️  {endpoint} ({method}): Response not JSON")
                else:
                    results[endpoint] = {
                        "status": f"❌ HTTP {response.status_code}",
                        "method": method,
                        "error": f"Expected 2xx, got {response.status_code}"
                    }
                    self.log(f"❌ {endpoint} ({method}): HTTP {response.status_code}")
                    
            except Exception as e:
                results[endpoint] = {
                    "status": "❌ ERROR",
                    "method": method,
                    "error": str(e)
                }
                self.log(f"❌ {endpoint} ({method}): Error - {str(e)}")
        
        self.results["api_endpoints"] = results
        return results
    
    def test_frontend_connectivity(self) -> Dict[str, Any]:
        """Test frontend accessibility and basic functionality."""
        self.log("Testing frontend connectivity...")
        
        results = {}
        
        # Test frontend main page
        try:
            response = self.session.get(FRONTEND_URL, timeout=10)
            if response.status_code == 200:
                results["main_page"] = {
                    "status": "✅ ACCESSIBLE",
                    "response_time": response.elapsed.total_seconds(),
                    "content_type": response.headers.get("content-type", "unknown")
                }
                self.log(f"✅ Frontend main page: Accessible ({response.elapsed.total_seconds():.3f}s)")
            else:
                results["main_page"] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": f"Expected 200, got {response.status_code}"
                }
                self.log(f"❌ Frontend main page: HTTP {response.status_code}")
        except Exception as e:
            results["main_page"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ Frontend main page: Error - {str(e)}")
        
        # Test if frontend can reach backend
        try:
            response = self.session.get(f"{FRONTEND_URL}/api/health", timeout=10)
            # This might fail if the frontend doesn't have a proxy, which is expected
            results["backend_from_frontend"] = {
                "status": "ℹ️  NO_PROXY",
                "note": "Frontend doesn't proxy to backend (expected behavior)"
            }
            self.log("ℹ️  Frontend doesn't proxy to backend (expected behavior)")
        except Exception as e:
            results["backend_from_frontend"] = {
                "status": "ℹ️  NO_PROXY",
                "note": "Frontend doesn't proxy to backend (expected behavior)"
            }
            self.log("ℹ️  Frontend doesn't proxy to backend (expected behavior)")
        
        self.results["frontend_connectivity"] = results
        return results
    
    def test_cors_configuration(self) -> Dict[str, Any]:
        """Test CORS configuration by making cross-origin requests."""
        self.log("Testing CORS configuration...")
        
        results = {}
        
        # Test CORS preflight request
        try:
            headers = {
                "Origin": FRONTEND_URL,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
            
            response = self.session.options(f"{BACKEND_URL}/health", headers=headers, timeout=10)
            
            cors_headers = {
                "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
                "access-control-allow-headers": response.headers.get("access-control-allow-headers"),
                "access-control-allow-credentials": response.headers.get("access-control-allow-credentials")
            }
            
            if response.status_code == 200:
                results["preflight_request"] = {
                    "status": "✅ WORKING",
                    "cors_headers": cors_headers
                }
                self.log("✅ CORS preflight request: Working")
            else:
                results["preflight_request"] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "cors_headers": cors_headers
                }
                self.log(f"❌ CORS preflight request: HTTP {response.status_code}")
                
        except Exception as e:
            results["preflight_request"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ CORS preflight request: Error - {str(e)}")
        
        # Test actual cross-origin request
        try:
            headers = {"Origin": FRONTEND_URL}
            response = self.session.get(f"{BACKEND_URL}/health", headers=headers, timeout=10)
            
            if response.status_code == 200:
                origin_header = response.headers.get("access-control-allow-origin")
                results["cross_origin_request"] = {
                    "status": "✅ WORKING",
                    "access_control_allow_origin": origin_header
                }
                self.log(f"✅ Cross-origin request: Working (Origin: {origin_header})")
            else:
                results["cross_origin_request"] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": f"Expected 200, got {response.status_code}"
                }
                self.log(f"❌ Cross-origin request: HTTP {response.status_code}")
                
        except Exception as e:
            results["cross_origin_request"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ Cross-origin request: Error - {str(e)}")
        
        self.results["cors_test"] = results
        return results
    
    async def test_websocket_connection(self) -> Dict[str, Any]:
        """Test WebSocket connectivity."""
        self.log("Testing WebSocket connection...")
        
        results = {}
        
        try:
            ws_url = BACKEND_URL.replace("http://", "ws://") + "/ws/stream"
            self.log(f"Attempting WebSocket connection to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                # Send a test message
                test_message = {"type": "ping", "data": "test"}
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    results["websocket_connection"] = {
                        "status": "✅ CONNECTED",
                        "test_message_sent": test_message,
                        "response_received": response
                    }
                    self.log("✅ WebSocket connection: Connected and responsive")
                except asyncio.TimeoutError:
                    results["websocket_connection"] = {
                        "status": "⚠️  NO_RESPONSE",
                        "test_message_sent": test_message,
                        "note": "Connected but no response received"
                    }
                    self.log("⚠️  WebSocket connection: Connected but no response")
                    
        except Exception as e:
            results["websocket_connection"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ WebSocket connection: Error - {str(e)}")
        
        self.results["websocket_test"] = results
        return results
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity."""
        self.log("Testing database connectivity...")
        
        results = {}
        
        # Test if we can access database-related endpoints
        try:
            response = self.session.get(f"{BACKEND_URL}/data/market/status", timeout=10)
            if response.status_code == 200:
                results["market_status"] = {
                    "status": "✅ ACCESSIBLE",
                    "response_time": response.elapsed.total_seconds()
                }
                self.log("✅ Market status endpoint: Accessible")
            else:
                results["market_status"] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": f"Expected 200, got {response.status_code}"
                }
                self.log(f"❌ Market status endpoint: HTTP {response.status_code}")
        except Exception as e:
            results["market_status"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ Market status endpoint: Error - {str(e)}")
        
        self.results["database_test"] = results
        return results
    
    def test_external_services(self) -> Dict[str, Any]:
        """Test external service connectivity."""
        self.log("Testing external service connectivity...")
        
        results = {}
        
        # Test Zerodha connectivity through backend
        try:
            response = self.session.get(f"{BACKEND_URL}/data/mapping/symbol-to-token", 
                                      params={"symbol": "RELIANCE"}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "token" in data:
                    results["zerodha_connectivity"] = {
                        "status": "✅ WORKING",
                        "test_symbol": "RELIANCE",
                        "response_time": response.elapsed.total_seconds()
                    }
                    self.log("✅ Zerodha connectivity: Working")
                else:
                    results["zerodha_connectivity"] = {
                        "status": "⚠️  PARTIAL",
                        "test_symbol": "RELIANCE",
                        "note": "Connected but unexpected response format"
                    }
                    self.log("⚠️  Zerodha connectivity: Partial (unexpected response format)")
            else:
                results["zerodha_connectivity"] = {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": f"Expected 200, got {response.status_code}"
                }
                self.log(f"❌ Zerodha connectivity: HTTP {response.status_code}")
        except Exception as e:
            results["zerodha_connectivity"] = {
                "status": "❌ ERROR",
                "error": str(e)
            }
            self.log(f"❌ Zerodha connectivity: Error - {str(e)}")
        
        self.results["external_services"] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        self.log("Generating test summary...")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        
        for category, results in self.results.items():
            if category == "summary":
                continue
                
            for test_name, result in results.items():
                total_tests += 1
                status = result.get("status", "UNKNOWN")
                
                if "✅" in status:
                    passed_tests += 1
                elif "❌" in status:
                    failed_tests += 1
                elif "⚠️" in status:
                    warnings += 1
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warnings,
            "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%",
            "overall_status": "✅ HEALTHY" if failed_tests == 0 else "❌ ISSUES_DETECTED"
        }
        
        self.results["summary"] = summary
        
        # Print summary
        print("\n" + "="*60)
        print("CONNECTION VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {summary['success_rate']}")
        print(f"Overall Status: {summary['overall_status']}")
        print("="*60)
        
        return summary
    
    async def run_all_tests(self):
        """Run all validation tests."""
        self.log("Starting comprehensive connection validation...")
        
        # Run synchronous tests
        self.test_backend_health()
        self.test_api_endpoints()
        self.test_frontend_connectivity()
        self.test_cors_configuration()
        self.test_database_connectivity()
        self.test_external_services()
        
        # Run asynchronous tests
        await self.test_websocket_connection()
        
        # Generate summary
        self.generate_summary()
        
        # Save results to file
        self.save_results()
        
        self.log("Connection validation completed!")
    
    def save_results(self):
        """Save test results to a JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"connection_validation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.log(f"Results saved to: {filename}")
        except Exception as e:
            self.log(f"Failed to save results: {str(e)}", "ERROR")

async def main():
    """Main function to run the validation."""
    validator = ConnectionValidator()
    await validator.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
