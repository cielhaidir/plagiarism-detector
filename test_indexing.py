#!/usr/bin/env python3
"""
Test script for the new yearly data indexing functionality.
Tests both Flask and FastAPI endpoints with sample 2025+ data.
"""

import requests
import json
import time
import threading
from datetime import datetime

# Configuration
FLASK_BASE_URL = "http://localhost:5000"
FASTAPI_BASE_URL = "http://localhost:5001"

# Test data - proposals from 2025 onwards
test_proposals_2025 = [
    {
        "id": 99991,
        "judul": "Penerapan Machine Learning untuk Deteksi Plagiarisme pada Dokumen Akademik",
        "skema": "Penelitian Dasar",
        "tahun": 2025,
        "ringkasan": "Penelitian ini mengembangkan sistem deteksi plagiarisme menggunakan pendekatan machine learning untuk meningkatkan akurasi deteksi pada dokumen akademik.",
        "pendahuluan": "Plagiarisme merupakan masalah serius dalam dunia akademik yang memerlukan solusi teknologi yang lebih canggih.",
        "masalah": "Bagaimana meningkatkan akurasi deteksi plagiarisme pada dokumen akademik menggunakan machine learning?",
        "metode": "Menggunakan algoritma ensemble learning dengan kombinasi TF-IDF dan semantic embeddings",
        "solusi": "Sistem deteksi plagiarisme berbasis ML dengan akurasi 95% untuk dokumen akademik"
    },
    {
        "id": 99992,
        "judul": "Analisis Sentimen pada Review Produk E-commerce Menggunakan Deep Learning",
        "skema": "Penelitian Terapan",
        "tahun": 2025,
        "ringkasan": "Menganalisis sentimen pelanggan terhadap produk e-commerce menggunakan pendekatan deep learning untuk meningkatkan pengalaman berbelanja.",
        "pendahuluan": "Review produk menjadi sumber informasi penting bagi pembeli dan penjual dalam platform e-commerce.",
        "masalah": "Bagaimana meningkatkan akurasi analisis sentimen pada review produk e-commerce?",
        "metode": "Implementasi LSTM dan transformer models untuk analisis sentimen multi-kelas",
        "solusi": "Model analisis sentimen dengan akurasi 92% untuk review produk e-commerce"
    }
]

# Test data - proposals from 2024 (should be rejected)
test_proposals_2024 = [
    {
        "id": 88881,
        "judul": "Studi Klasik tentang Algoritma Pencocokan String",
        "skema": "Penelitian Dasar",
        "tahun": 2024,
        "ringkasan": "Studi mendalam tentang algoritma pencocokan string klasik untuk deteksi plagiarisme.",
        "pendahuluan": "Algoritma pencocokan string menjadi dasar penting dalam berbagai aplikasi.",
        "masalah": "Bagaimana meningkatkan efisiensi algoritma pencocokan string?",
        "metode": "Analisis kompleksitas dan optimasi algoritma",
        "solusi": "Optimasi algoritma dengan kompleksitas O(n log n)"
    }
]

# Webhook server for testing async processing
class WebhookServer:
    def __init__(self, port=8080):
        self.port = port
        self.received_data = []
        self.server = None
    
    def start(self):
        from flask import Flask, request
        app = Flask(__name__)
        
        @app.route('/webhook', methods=['POST'])
        def webhook():
            data = request.get_json()
            self.received_data.append({
                'timestamp': datetime.now().isoformat(),
                'data': data
            })
            print(f"Webhook received: {data.get('status', 'unknown')}")
            return {'status': 'received'}, 200
        
        def run_server():
            app.run(host='0.0.0.0', port=self.port, debug=False)
        
        self.server = threading.Thread(target=run_server)
        self.server.daemon = True
        self.server.start()
        time.sleep(1)  # Give server time to start

def test_flask_sync():
    """Test Flask synchronous indexing."""
    print("\n=== Testing Flask Synchronous Indexing ===")
    
    # Test 2025 proposals
    print("Testing 2025 proposals...")
    response = requests.post(
        f"{FLASK_BASE_URL}/index_proposals",
        json={"proposals": test_proposals_2025}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
    
    # Test 2024 proposals (should be rejected)
    print("\nTesting 2024 proposals (should be rejected)...")
    response = requests.post(
        f"{FLASK_BASE_URL}/index_proposals",
        json={"proposals": test_proposals_2024}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Result: {result}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_flask_async():
    """Test Flask asynchronous indexing with webhook."""
    print("\n=== Testing Flask Asynchronous Indexing ===")
    
    webhook = WebhookServer(port=8080)
    webhook.start()
    
    # Test async with 2025 proposals
    print("Testing async 2025 proposals...")
    response = requests.post(
        f"{FLASK_BASE_URL}/index_proposals",
        json={
            "proposals": test_proposals_2025,
            "webhook_url": "http://localhost:8080/webhook"
        }
    )
    
    if response.status_code == 202:
        job_data = response.json()
        print(f"✅ Job started: {job_data['job_id']}")
        
        # Wait for webhook
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            if webhook.received_data:
                last_data = webhook.received_data[-1]
                if last_data['data'].get('job_id') == job_data['job_id']:
                    print(f"✅ Webhook received: {last_data['data']['status']}")
                    break
            time.sleep(1)
        else:
            print("⚠️  Webhook timeout")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_fastapi_sync():
    """Test FastAPI synchronous indexing."""
    print("\n=== Testing FastAPI Synchronous Indexing ===")
    
    # Test 2025 proposals
    print("Testing 2025 proposals...")
    response = requests.post(
        f"{FASTAPI_BASE_URL}/index_proposals",
        json={"proposals": test_proposals_2025}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_fastapi_async():
    """Test FastAPI asynchronous indexing with webhook."""
    print("\n=== Testing FastAPI Asynchronous Indexing ===")
    
    webhook = WebhookServer(port=8081)
    webhook.start()
    
    # Test async with 2025 proposals
    print("Testing async 2025 proposals...")
    response = requests.post(
        f"{FASTAPI_BASE_URL}/index_proposals",
        json={
            "proposals": test_proposals_2025,
            "webhook_url": "http://localhost:8081/webhook"
        }
    )
    
    if response.status_code == 202:
        job_data = response.json()
        print(f"✅ Job started: {job_data['job_id']}")
        
        # Wait for webhook
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            if webhook.received_data:
                last_data = webhook.received_data[-1]
                if last_data['data'].get('job_id') == job_data['job_id']:
                    print(f"✅ Webhook received: {last_data['data']['status']}")
                    break
            time.sleep(1)
        else:
            print("⚠️  Webhook timeout")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_validation():
    """Test validation errors."""
    print("\n=== Testing Validation Errors ===")
    
    # Test missing required fields
    invalid_proposal = {"id": 12345, "judul": "Test"}  # Missing skema and tahun
    response = requests.post(
        f"{FLASK_BASE_URL}/index_proposals",
        json={"proposals": [invalid_proposal]}
    )
    
    if response.status_code == 400:
        error_data = response.json()
        print(f"✅ Validation error caught: {error_data}")
    else:
        print(f"❌ Expected validation error, got: {response.status_code}")

def test_year_filtering():
    """Test year filtering with different thresholds."""
    print("\n=== Testing Year Filtering ===")
    
    # Test with custom year threshold
    response = requests.post(
        f"{FLASK_BASE_URL}/index_proposals",
        json={
            "proposals": test_proposals_2024,
            "year_threshold": 2023  # Should accept 2024
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Custom threshold result: {result}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def run_all_tests():
    """Run all tests."""
    print("🧪 Starting comprehensive indexing tests...")
    
    # Check if services are running
    try:
        flask_health = requests.get(f"{FLASK_BASE_URL}/health")
        fastapi_health = requests.get(f"{FASTAPI_BASE_URL}/health")
        
        if flask_health.status_code != 200:
            print("❌ Flask service not running")
            return
        
        if fastapi_health.status_code != 200:
            print("❌ FastAPI service not running")
            return
        
        print("✅ Both services are running")
        
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("Please ensure both Flask and FastAPI services are running:")
        print("  - Flask: python app.py (port 5000)")
        print("  - FastAPI: python fast_api.py (port 5001)")
        return
    
    # Run tests
    test_flask_sync()
    test_flask_async()
    test_fastapi_sync()
    test_fastapi_async()
    test_validation()
    test_year_filtering()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    run_all_tests()