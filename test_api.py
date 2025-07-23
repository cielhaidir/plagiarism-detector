import requests
import json

# Test the API endpoints
API_BASE = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("=== Testing Health Endpoint ===")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_info():
    """Test the info endpoint"""
    print("=== Testing Info Endpoint ===")
    response = requests.get(f"{API_BASE}/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_single_search():
    """Test the single search endpoint"""
    print("=== Testing Single Search Endpoint ===")
    
    search_data = {
        "query_text": "penelitian tentang teknologi",
        "column": "judul",
        "top_k": 5
    }
    
    response = requests.post(f"{API_BASE}/search", json=search_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_bulk_search():
    """Test the bulk search endpoint with correct format"""
    print("=== Testing Bulk Search Endpoint ===")
    
    bulk_data = {
        "texts": [
            {
                "text": "penelitian teknologi informasi",
                "column": "judul",
                "skema": "PKM"
            },
            {
                "text": "metodologi penelitian kualitatif",
                "column": "metode"
            }
        ],
        "top_k": 3
    }
    
    response = requests.post(f"{API_BASE}/search_bulk", json=bulk_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_bulk_search_wrong_format():
    """Test bulk search with wrong format to see the error"""
    print("=== Testing Bulk Search with Wrong Format ===")
    
    # This will cause the error you're seeing
    wrong_data = {
        "texts": [
            ["text1", "column1"],  # This is a list, not a dict!
            ["text2", "column2"]
        ]
    }
    
    response = requests.post(f"{API_BASE}/search_bulk", json=wrong_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    try:
        test_health()
        test_info()
        test_single_search()
        test_bulk_search()
        test_bulk_search_wrong_format()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the Flask app is running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")