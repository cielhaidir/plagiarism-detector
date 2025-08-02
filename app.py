# Set environment variables before importing torch-based libraries
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Enable MPS fallback for better compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Disable progress bars from transformers and other libraries
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['DISABLE_TQDM'] = 'true'

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import threading
import requests
import uuid
from datetime import datetime
import urllib.parse
import socket
import ipaddress
import time
import math
from similarity_utils import (
    jaccard_similarity,
    levenshtein_similarity,
    tfidf_cosine_similarity,
    sentence_embedding_similarity,
    calculate_final_score,
    initialize_sentence_model
)
from performance_monitor import performance_monitor, PerformanceTracker, log_search_metrics, log_bulk_metrics

app = Flask(__name__)

# Global variables to store loaded indices
indices = {}
metadata = None
stemmer = None
stopword_remover = None
# Load original proposal texts for matched_text retrieval
original_texts = None

def is_safe_webhook_url(url):
    """
    Validate webhook URL to prevent SSRF attacks.
    Returns True if URL is safe to use, False otherwise.
    """
    try:
        # Parse the URL
        parsed = urllib.parse.urlparse(url)
        
        # Check if scheme is allowed
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Check if hostname is provided
        if not parsed.hostname:
            return False
        
        # Resolve hostname to IP
        ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip)
        
        # Block private and reserved IP ranges
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_multicast or ip_obj.is_reserved:
            return False
        
        # Block localhost variations
        if parsed.hostname.lower() in ['localhost', '127.0.0.1', '::1']:
            return False
        
        return True
        
    except (socket.gaierror, ValueError, ipaddress.AddressValueError):
        return False

def preprocess_text(text, stemmer, stopword_remover):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

def load_indices():
    global indices, metadata, stemmer, stopword_remover, original_texts
    print("Loading Indonesian language tools...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    print("Loading metadata...")
    metadata = pd.read_csv('indices/metadata.csv')
    metadata.set_index('id', inplace=True)
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    print("Loading indices for each column...")
    for column in text_columns:
        print(f"  Loading {column}...")
        vectorizer_path = f'indices/vectorizer_{column}.pkl'
        matrix_path = f'indices/tfidf_matrix_{column}.pkl'
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = joblib.load(matrix_path)
        indices[column] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix
        }
    print("Loading original proposal texts for matched_text retrieval...")
    original_texts = pd.read_csv('skripsi_with_skema.csv')
    original_texts.set_index('id', inplace=True)
    print("Pre-loading sentence transformer model for faster queries...")
    initialize_sentence_model()
    print("All indices and models loaded successfully!")

@performance_monitor("Single Search Processing")
def _process_single_search(query_text, column, skema_filter=None, top_k=10):
    """
    Helper function to process a single search request.
    Used by both synchronous and asynchronous endpoints.
    """
    start_time = time.time()
    results = search_column(query_text, column, skema_filter, top_k)
    total_time_ms = (time.time() - start_time) * 1000
    log_search_metrics(query_text, column, len(results), total_time_ms)
    return results

def process_query_thread(query_data):
    """
    Process a single query in a thread.
    This function runs in the same process but different thread for better GPU sharing.
    """
    query_idx, text_item, top_k = query_data
    
    # Validate input format
    if not isinstance(text_item, dict):
        return {
            "query_index": query_idx,
            "error": f"Invalid input format at index {query_idx}. Expected dictionary but got {type(text_item).__name__}"
        }
    
    skema_filter = text_item.get('skema')
    item_results = []
    
    # Loop through all possible columns
    for column in indices.keys():
        query_text = text_item.get(column)
        if query_text:
            results = search_column(query_text, column, skema_filter, top_k)
            for result in results:
                result['searched_column'] = column
            item_results.extend(results)
    
    # Sort and limit results
    item_results.sort(key=lambda x: x['final_score'], reverse=True)
    item_results = item_results[:top_k]
    
    return {
        "query_index": query_idx,
        "proposal_id": text_item.get('proposal_id'),
        "results": item_results,
        "query_info": {
            "skema_filter": skema_filter,
            "columns_searched": [col for col in indices.keys() if text_item.get(col)],
            "total_results": len(item_results)
        }
    }

@performance_monitor("Threaded Bulk Search Processing")
def _process_bulk_search_parallel(texts, top_k=1, max_workers=None):
    """
    Helper function to process bulk search requests using threading for better GPU sharing.
    Uses thread-based parallelism to share GPU resources efficiently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    start_time = time.time()
    
    # Determine number of workers (default to number of CPU cores)
    if max_workers is None:
        import os
        max_workers = min(os.cpu_count(), len(texts))
    
    print(f"[BULK] Processing {len(texts)} queries using {max_workers} parallel threads")
    
    # Prepare query data for threading
    query_data_list = [
        (i, text_item, top_k)
        for i, text_item in enumerate(texts)
    ]
    
    # Process queries in parallel using threads
    all_results = []
    total_results = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(process_query_thread, query_data): query_data[0]
                for query_data in query_data_list
            }
            
            # Collect results as they complete with progress bar
            with tqdm(total=len(texts), desc="[BULK] Processing queries", unit="query") as pbar:
                for future in as_completed(future_to_query):
                    query_idx = future_to_query[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        total_results += len(result.get('results', []))
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n[BULK] Error processing query {query_idx}: {e}")
                        # Add error result for this query
                        all_results.append({
                            "query_index": query_idx,
                            "error": f"Processing error: {str(e)}"
                        })
                        pbar.update(1)
    
    except Exception as e:
        print(f"[BULK] Critical error in threaded processing: {e}")
        return None, {"error": f"Threaded processing failed: {str(e)}"}
    
    # Sort results by query_index to maintain order
    all_results.sort(key=lambda x: x['query_index'])
    
    total_time_ms = (time.time() - start_time) * 1000
    log_bulk_metrics(len(texts), total_results, total_time_ms)
    
    print(f"[BULK] Threaded processing completed in {total_time_ms:.2f}ms")
    return all_results, None

@performance_monitor("Sequential Bulk Search Processing")
def _process_bulk_search(texts, top_k=1):
    """
    Helper function to process bulk search requests sequentially.
    Used by both synchronous and asynchronous endpoints.
    """
    start_time = time.time()
    bulk_results = []
    total_results = 0
    
    for i, text_item in enumerate(texts):
        print(f"[BULK] Processing query {i+1}/{len(texts)}")
        
        # Validate input format - enforce strict API contract
        if not isinstance(text_item, dict):
            return None, {
                "error": f"Invalid input format at index {i}. Expected dictionary but got {type(text_item).__name__}",
                "query_index": i
            }
        
        skema_filter = text_item.get('skema')
        item_results = []
        
        # Loop through all possible columns
        for column in indices.keys():
            query_text = text_item.get(column)
            if query_text:
                print(f"[BULK]   - Searching column '{column}' for query {i+1}")
                with PerformanceTracker(f"Column {column} search for query {i+1}"):
                    results = search_column(query_text, column, skema_filter, top_k)
                for result in results:
                    result['searched_column'] = column
                item_results.extend(results)
        
        # Sort and limit results
        item_results.sort(key=lambda x: x['final_score'], reverse=True)
        item_results = item_results[:top_k]
        total_results += len(item_results)
        print(f"[BULK]   - Finished query {i+1}, found {len(item_results)} results")
        
        bulk_results.append({
            "query_index": i,
            "proposal_id": text_item.get('proposal_id'),
            "results": item_results,
            "query_info": {
                "skema_filter": skema_filter,
                "columns_searched": [col for col in indices.keys() if text_item.get(col)],
                "total_results": len(item_results)
            }
        })
    
    total_time_ms = (time.time() - start_time) * 1000
    log_bulk_metrics(len(texts), total_results, total_time_ms)
    return bulk_results, None

@performance_monitor("TF-IDF Search")
def search_column(query_text, column, skema_filter=None, top_k=10):
    global indices, metadata, stemmer, stopword_remover, original_texts
    if column not in indices:
        return []
    processed_query = preprocess_text(query_text, stemmer, stopword_remover)
    vectorizer = indices[column]['vectorizer']
    query_vector = vectorizer.transform([processed_query])
    similarity_matrix = cosine_similarity(query_vector, indices[column]['matrix'])
    similarities = similarity_matrix[0]
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for idx in sorted_indices:
        similarity_score = similarities[idx]
        if similarity_score < 0.1:
            break
        proposal_id = metadata.index[idx]
        proposal_skema = metadata.iloc[idx]['skema']
        if skema_filter and proposal_skema != skema_filter:
            continue

        # Retrieve the matched text for this proposal and column
        matched_text = ""
        if proposal_id in original_texts.index and column in original_texts.columns:
            matched_text = str(original_texts.loc[proposal_id, column]) if pd.notna(original_texts.loc[proposal_id, column]) else ""

        # Calculate advanced similarity metrics
        exact_score = jaccard_similarity(query_text, matched_text)
        fuzzy_score = levenshtein_similarity(query_text, matched_text)
        semantic_score = sentence_embedding_similarity(query_text, matched_text)
        final_score = calculate_final_score(exact_score, fuzzy_score, semantic_score, query_text, matched_text)

        results.append({
            'id': int(proposal_id),
            'skema': proposal_skema,
            'similarity_score': float(similarity_score),  # legacy TF-IDF score
            'exact_score': float(exact_score),
            'fuzzy_score': float(fuzzy_score),
            'semantic_score': float(semantic_score),
            'final_score': float(final_score),
            'column': column,
            'matched_text': matched_text[:500] + "..." if len(matched_text) > 500 else matched_text  # Truncate for API response
        })
        if len(results) >= top_k:
            break
    return results

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query_text = data.get('query_text')
        column = data.get('column')
        skema_filter = data.get('skema')
        top_k = data.get('top_k', 10)
        webhook_url = data.get('webhook_url')
        
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        if not column:
            return jsonify({"error": "column is required"}), 400
        if column not in indices:
            valid_columns = list(indices.keys())
            return jsonify({"error": f"Invalid column. Valid columns are: {valid_columns}"}), 400
        
        # If webhook URL is provided, process asynchronously
        if webhook_url:
            # Validate webhook URL to prevent SSRF attacks
            # if not is_safe_webhook_url(webhook_url):
            #     return jsonify({"error": "Invalid or unsafe webhook URL"}), 400
            
            job_id = str(uuid.uuid4())
            
            def async_search():
                try:
                    results = _process_single_search(query_text, column, skema_filter, top_k)
                    
                    webhook_payload = {
                        "job_id": job_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "results": results,
                        "query_info": {
                            "column": column,
                            "skema_filter": skema_filter,
                            "total_results": len(results)
                        }
                    }
                    
                    # Send results to webhook
                    requests.post(webhook_url, json=webhook_payload, timeout=30)
                    print(f"Search results sent to webhook for job {job_id}")
                    
                except Exception as e:
                    error_payload = {
                        "job_id": job_id,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    try:
                        requests.post(webhook_url, json=error_payload, timeout=30)
                    except:
                        pass
                    print(f"Search failed for job {job_id}: {e}")
            
            # Start async processing
            search_thread = threading.Thread(target=async_search)
            search_thread.daemon = True
            search_thread.start()
            
            return jsonify({
                "job_id": job_id,
                "status": "processing",
                "message": "Search started. Results will be sent to webhook when complete.",
                "webhook_url": webhook_url
            }), 202
        
        # Synchronous processing (original behavior)
        else:
            results = _process_single_search(query_text, column, skema_filter, top_k)
            return jsonify({
                "results": results,
                "query_info": {
                    "column": column,
                    "skema_filter": skema_filter,
                    "total_results": len(results)
                }
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_bulk', methods=['POST'])
def search_bulk():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        texts = data.get('texts', [])
        top_k = data.get('top_k', 1)
        webhook_url = data.get('webhook_url')
        use_parallel = data.get('use_parallel', True)  # Enable parallel processing by default
        max_workers = data.get('max_workers', None)  # Allow custom worker count
        
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        
        # Choose processing method
        process_func = _process_bulk_search_parallel if use_parallel else _process_bulk_search
        
        # If webhook URL is provided, process asynchronously
        if webhook_url:
            # Validate webhook URL to prevent SSRF attacks
            # if not is_safe_webhook_url(webhook_url):
            #     return jsonify({"error": "Invalid or unsafe webhook URL"}), 400
            
            job_id = str(uuid.uuid4())
            
            def async_bulk_search():
                try:
                    if use_parallel:
                        bulk_results, error = _process_bulk_search_parallel(texts, top_k, max_workers)
                    else:
                        bulk_results, error = _process_bulk_search(texts, top_k)
                        
                    if error:
                        error_payload = {
                            "job_id": job_id,
                            "status": "failed",
                            "timestamp": datetime.now().isoformat(),
                            "error": error["error"],
                            "query_index": error.get("query_index")
                        }
                        requests.post(webhook_url, json=error_payload, timeout=30)
                        print(f"Bulk search failed for job {job_id}: {error['error']}")
                        return
                    
                    webhook_payload = {
                        "job_id": job_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "bulk_results": bulk_results,
                        "total_queries": len(texts),
                        "processing_method": "parallel" if use_parallel else "sequential"
                    }
                    
                    # Send results to webhook
                    requests.post(webhook_url, json=webhook_payload, timeout=30)
                    print(f"Bulk search results sent to webhook for job {job_id}")
                    
                except Exception as e:
                    error_payload = {
                        "job_id": job_id,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    try:
                        requests.post(webhook_url, json=error_payload, timeout=30)
                    except:
                        pass
                    print(f"Bulk search failed for job {job_id}: {e}")
            
            # Start async processing
            search_thread = threading.Thread(target=async_bulk_search)
            search_thread.daemon = True
            search_thread.start()
            
            return jsonify({
                "job_id": job_id,
                "status": "processing",
                "message": f"Bulk search started using {'parallel' if use_parallel else 'sequential'} processing. Results will be sent to webhook when complete.",
                "webhook_url": webhook_url,
                "processing_method": "parallel" if use_parallel else "sequential",
                "max_workers": max_workers if use_parallel else None
            }), 202
        
        # Synchronous processing
        else:
            if use_parallel:
                bulk_results, error = _process_bulk_search_parallel(texts, top_k, max_workers)
            else:
                bulk_results, error = _process_bulk_search(texts, top_k)
                
            if error:
                return jsonify(error), 400
            
            return jsonify({
                "bulk_results": bulk_results,
                "total_queries": len(texts),
                "processing_method": "parallel" if use_parallel else "sequential",
                "max_workers": max_workers if use_parallel else None
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "indices_loaded": len(indices) > 0})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "available_columns": list(indices.keys()),
        "total_proposals": len(metadata) if metadata is not None else 0,
        "unique_skemas": list(metadata['skema'].unique()) if metadata is not None else []
    })

def init_app():
    """Initialize the application - load indices if not already loaded."""
    global indices, metadata, original_texts
    if not indices or metadata is None or original_texts is None:
        print("Initializing plagiarism detection API...")
        load_indices()
        print("API initialization complete!")

# Ensure initialization happens before first request
def ensure_initialized():
    """Ensure the app is initialized before processing requests."""
    if not indices:
        init_app()

# Add initialization check to all endpoints
@app.before_request
def before_request():
    ensure_initialized()

if __name__ == '__main__':
    print("Starting plagiarism detection API...")
    init_app()
    print("API server ready!")
    # For development only - use Gunicorn for production
    app.run(debug=False, host='0.0.0.0', port=5000)