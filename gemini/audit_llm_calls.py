#!/usr/bin/env python3
"""
Comprehensive Audit Script for LLM Calls and Response Extraction Patterns

This script identifies all LLM API calls across the codebase and checks if they use
the robust response extraction pattern to avoid "No text content found in LLM response" errors.
"""

import os
import re
import sys
from typing import List, Dict, Tuple


def find_llm_calls_in_file(file_path: str) -> List[Dict[str, any]]:
    """Find all LLM calls in a single file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return []
    
    lines = content.split('\n')
    llm_calls = []
    
    # Patterns to search for LLM calls
    patterns = [
        (r'await.*\.call_llm\(', 'call_llm'),
        (r'await.*\.call_llm_with_image\(', 'call_llm_with_image'),  
        (r'await.*\.call_llm_with_images\(', 'call_llm_with_images'),
        (r'await.*\.call_llm_with_code_execution\(', 'call_llm_with_code_execution'),
        (r'\.call_llm\([^)]*\)', 'sync_call_llm'),
        (r'response\s*=.*\.call_llm', 'response_call_llm'),
        (r'text.*=.*\.call_llm', 'text_call_llm'),
    ]
    
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
            
        for pattern, call_type in patterns:
            if re.search(pattern, line):
                # Get method context
                method_name = "unknown"
                for i in range(max(0, line_num - 20), line_num):
                    method_match = re.search(r'def\s+(\w+)\s*\(', lines[i])
                    if method_match:
                        method_name = method_match.group(1)
                        break
                
                llm_calls.append({
                    'file': file_path,
                    'line': line_num,
                    'method': method_name,
                    'call_type': call_type,
                    'code': line.strip(),
                    'is_robust': 'call_llm_with_code_execution' in line or 'fallback' in line.lower()
                })
    
    return llm_calls


def analyze_method_robustness(file_path: str, method_name: str, start_line: int) -> Tuple[bool, str]:
    """Analyze if a method uses the robust response extraction pattern"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return False, "Could not read file"
    
    # Look for the method and analyze 50 lines after it
    method_lines = lines[start_line-1:start_line+50]
    method_code = '\n'.join(method_lines)
    
    has_code_execution = 'call_llm_with_code_execution' in method_code
    has_fallback = 'fallback' in method_code.lower() and 'call_llm(' in method_code
    has_robust_pattern = has_code_execution and has_fallback
    
    if has_robust_pattern:
        return True, "✅ Uses robust pattern (call_llm_with_code_execution + fallback)"
    elif has_code_execution:
        return True, "⚡ Uses call_llm_with_code_execution (primary robust method)"
    elif 'call_llm_with_image' in method_code and 'gemini_core.py' in file_path:
        return True, "✅ Uses improved call_llm_with_image (fixed in gemini_core.py)"
    elif has_fallback:
        return False, "⚠️ Has fallback but no primary robust method"
    else:
        return False, "❌ Uses vulnerable direct call_llm pattern"


def audit_directory(directory: str) -> List[Dict[str, any]]:
    """Audit all Python files in a directory for LLM calls"""
    
    all_calls = []
    
    # Walk through all Python files
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = ['.git', '__pycache__', '.pytest_cache', 'node_modules']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                calls = find_llm_calls_in_file(file_path)
                all_calls.extend(calls)
    
    return all_calls


def main():
    """Main audit function"""
    
    print("🔍 COMPREHENSIVE LLM CALLS AUDIT")
    print("=" * 60)
    
    # Get the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir.endswith('/gemini'):
        backend_dir = os.path.dirname(backend_dir)
    
    print(f"📁 Scanning directory: {backend_dir}")
    print()
    
    # Find all LLM calls
    all_calls = audit_directory(backend_dir)
    
    if not all_calls:
        print("✅ No LLM calls found!")
        return
    
    print(f"🔍 Found {len(all_calls)} potential LLM calls")
    print()
    
    # Group by file and method
    by_method = {}
    for call in all_calls:
        key = f"{call['file']}::{call['method']}"
        if key not in by_method:
            by_method[key] = []
        by_method[key].append(call)
    
    # Analyze each method
    results = {
        'robust': [],
        'vulnerable': [],
        'mixed': []
    }
    
    for method_key, calls in by_method.items():
        file_path, method_name = method_key.split('::', 1)
        
        # Analyze the method
        first_call = calls[0]
        is_robust, reason = analyze_method_robustness(file_path, method_name, first_call['line'])
        
        method_info = {
            'method_key': method_key,
            'file': os.path.relpath(file_path, backend_dir),
            'method': method_name,
            'calls': calls,
            'is_robust': is_robust,
            'reason': reason,
            'call_count': len(calls)
        }
        
        if is_robust:
            results['robust'].append(method_info)
        else:
            results['vulnerable'].append(method_info)
    
    # Print results
    print("📊 AUDIT RESULTS")
    print("=" * 60)
    
    print(f"✅ ROBUST METHODS ({len(results['robust'])})")
    print("-" * 40)
    for method in results['robust']:
        print(f"  {method['file']}::{method['method']}")
        print(f"    {method['reason']}")
        print(f"    Calls: {method['call_count']}")
        print()
    
    print(f"❌ VULNERABLE METHODS ({len(results['vulnerable'])})")
    print("-" * 40)
    for method in results['vulnerable']:
        print(f"  {method['file']}::{method['method']}")
        print(f"    {method['reason']}")
        print(f"    Calls: {method['call_count']}")
        for call in method['calls']:
            print(f"      Line {call['line']}: {call['call_type']} - {call['code'][:100]}...")
        print()
    
    # Summary
    total_methods = len(results['robust']) + len(results['vulnerable'])
    robust_percentage = (len(results['robust']) / total_methods * 100) if total_methods > 0 else 0
    
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"Total methods with LLM calls: {total_methods}")
    print(f"Robust methods: {len(results['robust'])} ({robust_percentage:.1f}%)")
    print(f"Vulnerable methods: {len(results['vulnerable'])}")
    
    if results['vulnerable']:
        print()
        print("🚨 ACTION REQUIRED:")
        print("The following methods need to be updated with robust response extraction:")
        for method in results['vulnerable']:
            print(f"  - {method['file']}::{method['method']}")
        
        return False  # Indicate that fixes are needed
    else:
        print()
        print("🎉 ALL METHODS USE ROBUST RESPONSE EXTRACTION!")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)