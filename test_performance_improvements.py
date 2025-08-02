#!/usr/bin/env python3
"""
Test script to validate performance improvements implementation.
This script tests the core functionality to ensure optimizations don't break existing features.
"""

import requests
import json
import time
import sys

# Test configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 30

def test_health_check():
    """Test basic health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_search():
    """Test single search functionality."""
    print("\nTesting single search...")
    payload = {
        "query_text": "machine learning artificial intelligence",
        "column": "judul",
        "top_k": 5
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search", json=payload, timeout=TIMEOUT)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            execution_time = (end_time - start_time) * 1000
            print(f"✅ Single search passed in {execution_time:.2f}ms")
            print(f"   Found {len(data.get('results', []))} results")
            return True
        else:
            print(f"❌ Single search failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single search error: {e}")
        return False

def test_bulk_search():
    """Test bulk search functionality."""
    print("\nTesting bulk search...")
    payload = {
        "texts": [
            {
                "proposal_id": "test1",
                "judul": "machine learning",
                "skema": "PDP"
            },
            {
                "proposal_id": "test2",
                "ringkasan": "artificial intelligence research",
                "skema": "PDP"
            }
        ],
        "top_k": 3
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/search_bulk", json=payload, timeout=TIMEOUT)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            execution_time = (end_time - start_time) * 1000
            print(f"✅ Bulk search passed in {execution_time:.2f}ms")
            print(f"   Processed {data.get('total_queries', 0)} queries")
            return True
        else:
            print(f"❌ Bulk search failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Bulk search error: {e}")
        return False

def test_webhook_validation():
    """Test webhook URL validation (SSRF protection)."""
    print("\nTesting webhook SSRF protection...")
    
    # Test with dangerous localhost URL
    payload = {
        "query_text": "test query",
        "column": "judul",
        "webhook_url": "http://localhost:22/admin"  # Dangerous URL
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search", json=payload, timeout=TIMEOUT)
        if response.status_code == 400:
            print("✅ SSRF protection working - dangerous URL blocked")
            return True
        else:
            print(f"❌ SSRF protection failed - dangerous URL allowed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ SSRF test error: {e}")
        return False

def test_invalid_bulk_format():
    """Test strict API contract enforcement."""
    print("\nTesting strict API contract...")
    
    # Send invalid format (string instead of dict)
    payload = {
        "texts": ["invalid string instead of dict"],
        "top_k": 1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search_bulk", json=payload, timeout=TIMEOUT)
        if response.status_code == 400:
            print("✅ Strict API contract working - invalid format rejected")
            return True
        else:
            print(f"❌ API contract not enforced - invalid format accepted: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API contract test error: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("🚀 Starting performance improvements validation tests...")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Search", test_single_search),
        ("Bulk Search", test_bulk_search),
        ("SSRF Protection", test_webhook_validation),
        ("API Contract", test_invalid_bulk_format)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance improvements validated successfully!")
        return 0
    else:
        print("⚠️  Some tests failed - please check the implementation")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)