#!/usr/bin/env python3
"""
Test script for Qdrant-based plagiarism detection
"""

import requests
import json
import time

def test_single_search():
    """Test single search endpoint."""
    print("Testing single search...")
    
    url = "http://localhost:5001/search"
    payload = {
        "query_text": "pengembangan aplikasi mobile untuk pendidikan",
        "column": "judul",
        "top_k": 5,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Single search successful: {len(results['results'])} results")
            for i, result in enumerate(results['results'][:3]):
                print(f"  {i+1}. Score: {result['similarity_score']:.3f} - {result['text'][:100]}...")
        else:
            print(f"❌ Single search failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Single search error: {e}")

def test_bulk_search():
    """Test bulk search endpoint."""
    print("\nTesting bulk search...")
    
    url = "http://localhost:5001/search_bulk"
    payload = {
        "texts": [
            {
                "judul": "pengembangan aplikasi mobile untuk pendidikan",
                "ringkasan": "aplikasi ini bertujuan untuk meningkatkan kualitas pendidikan"
            },
            {
                "judul": "sistem deteksi plagiarisme menggunakan AI",
                "masalah": "plagiarisme menjadi masalah serius dalam dunia akademik"
            }
        ],
        "top_k": 3,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Bulk search successful: {len(results['bulk_results'])} queries processed")
            for i, query_result in enumerate(results['bulk_results']):
                print(f"  Query {i+1}: {len(query_result['results'])} results")
        else:
            print(f"❌ Bulk search failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Bulk search error: {e}")

def test_performance():
    """Test search performance."""
    print("\nTesting performance...")
    
    url = "http://localhost:5001/search"
    payload = {
        "query_text": "teknologi informasi untuk pendidikan",
        "top_k": 10,
        "threshold": 0.3
    }
    
    times = []
    for i in range(5):
        start = time.time()
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Query {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  Query {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Average query time: {avg_time:.3f}s")

def check_health():
    """Check if service is ready."""
    print("Checking service health...")
    
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check: {health}")
            return health.get('initialization_complete', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Qdrant Plagiarism Detection Test Suite")
    print("=" * 50)
    
    # Wait for service to be ready
    max_wait = 60
    waited = 0
    while waited < max_wait:
        if check_health():
            break
        print(f"Waiting for service... ({waited}s)")
        time.sleep(5)
        waited += 5
    
    if waited >= max_wait:
        print("❌ Service not ready after 60s")
        return
    
    # Run tests
    test_single_search()
    test_bulk_search()
    test_performance()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()