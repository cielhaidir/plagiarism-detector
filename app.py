from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
from similarity_utils import (
    jaccard_similarity,
    levenshtein_similarity,
    tfidf_cosine_similarity,
    sentence_embedding_similarity,
    calculate_final_score,
    initialize_sentence_model,
    cleanup_caches,
    save_embedding_cache
)
import atexit

app = Flask(__name__)

# Global variables to store loaded indices
indices = {}
metadata = None
stemmer = None
stopword_remover = None
# Load original proposal texts for matched_text retrieval
original_texts = None
# Threading lock for thread-safe operations
_lock = threading.Lock()
_sentence_model_loaded = False

def preprocess_text(text, stemmer_param=None, stopword_remover_param=None):
    """Optimized text preprocessing with caching"""
    global stemmer, stopword_remover
    
    # Use global variables if parameters not provided
    if stemmer_param is None:
        stemmer_param = stemmer
    if stopword_remover_param is None:
        stopword_remover_param = stopword_remover
    
    if not isinstance(text, str):
        return ""
    
    # Create a cache key based on text content
    cache_key = hash(text)
    
    # Simple in-memory cache (you could use more sophisticated caching)
    if not hasattr(preprocess_text, '_cache'):
        preprocess_text._cache = {}
    
    if cache_key in preprocess_text._cache:
        return preprocess_text._cache[cache_key]
    
    # Process the text
    processed = text.lower()
    processed = re.sub(r'http\S+|www\S+|https\S+', '', processed, flags=re.MULTILINE)
    processed = re.sub(r'\d+', '', processed)
    processed = re.sub(r'[^\w\s]', '', processed)
    processed = processed.strip()
    processed = stopword_remover_param.remove(processed)
    processed = stemmer_param.stem(processed)
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(preprocess_text._cache) > 1000:
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(preprocess_text._cache.keys())[:100]
        for key in oldest_keys:
            del preprocess_text._cache[key]
    
    preprocess_text._cache[cache_key] = processed
    return processed

def load_language_tools():
    """Load Indonesian language processing tools"""
    print("Loading Indonesian language tools...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    return stemmer, stopword_remover

def load_metadata():
    """Load metadata CSV"""
    print("Loading metadata...")
    return pd.read_csv('indices/metadata.csv')

def load_original_texts():
    """Load original proposal texts"""
    print("Loading original proposal texts...")
    return pd.read_csv('skripsi_with_skema.csv')

def load_column_index(column):
    """Load vectorizer and TF-IDF matrix for a specific column"""
    print(f"  Loading {column}...")
    vectorizer_path = f'indices/vectorizer_{column}.pkl'
    matrix_path = f'indices/tfidf_matrix_{column}.pkl'
    
    # Load both files concurrently if possible
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(matrix_path)
    
    return column, {
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix
    }

def lazy_load_sentence_model():
    """Lazy load sentence transformer model only when needed"""
    global _sentence_model_loaded
    if not _sentence_model_loaded:
        with _lock:
            if not _sentence_model_loaded:
                print("Loading sentence transformer model...")
                initialize_sentence_model()
                _sentence_model_loaded = True

def load_indices():
    global indices, metadata, stemmer, stopword_remover, original_texts
    
    start_time = time.time()
    print("Starting optimized loading process...")
    
    # Define what to load
    text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
    
    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all loading tasks
        futures = {}
        
        # Language tools (blocking, needed for other operations)
        lang_future = executor.submit(load_language_tools)
        
        # Metadata (needed early)
        metadata_future = executor.submit(load_metadata)
        
        # Original texts (can be loaded in parallel)
        texts_future = executor.submit(load_original_texts)
        
        # Column indices (can be loaded in parallel)
        column_futures = []
        for column in text_columns:
            future = executor.submit(load_column_index, column)
            column_futures.append(future)
        
        # Wait for language tools first (needed for preprocessing)
        stemmer, stopword_remover = lang_future.result()
        print("✓ Language tools loaded")
        
        # Wait for metadata
        metadata = metadata_future.result()
        print("✓ Metadata loaded")
        
        # Wait for original texts
        original_texts = texts_future.result()
        print("✓ Original texts loaded")
        
        # Wait for all column indices
        print("Loading indices for columns...")
        for future in as_completed(column_futures):
            column, index_data = future.result()
            indices[column] = index_data
            print(f"✓ {column} index loaded")
    
    elapsed = time.time() - start_time
    print(f"All indices loaded successfully in {elapsed:.2f} seconds!")
    print("Note: Sentence transformer model will be loaded on first use for faster startup.")
    
    # Register cleanup function for graceful shutdown
    atexit.register(cleanup_caches)

def pre_warm_embeddings():
    """Pre-warm embeddings for common query patterns to speed up first searches."""
    print("Pre-warming embeddings for faster first searches...")
    
    # Sample some common texts from the dataset for pre-warming
    if original_texts is not None and not original_texts.empty:
        # Get a sample of texts to pre-compute embeddings
        sample_size = min(50, len(original_texts))  # Pre-warm with up to 50 samples
        sample_texts = []
        
        text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
        for column in text_columns:
            if column in original_texts.columns:
                # Get non-null values and sample them
                non_null_texts = original_texts[column].dropna()
                if not non_null_texts.empty:
                    sample_texts.extend(non_null_texts.head(10).tolist())
        
        # Common query patterns to pre-warm
        common_queries = [
            "machine learning",
            "data analysis",
            "sistem informasi",
            "aplikasi web",
            "database",
            "algoritma",
            "implementasi",
            "analisis data",
            "web application",
            "mobile app"
        ]
        sample_texts.extend(common_queries)
        
        # Pre-compute embeddings in background
        from similarity_utils import get_cached_embedding
        warmed_count = 0
        for text in sample_texts:
            if text and isinstance(text, str) and len(text.strip()) > 0:
                try:
                    get_cached_embedding(text.strip())
                    warmed_count += 1
                except Exception as e:
                    continue  # Skip problematic texts
        
        print(f"Pre-warmed {warmed_count} embeddings for faster searches")
        # Save the pre-warmed cache
        save_embedding_cache()

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
        proposal_id = metadata.iloc[idx]['id']
        proposal_skema = metadata.iloc[idx]['skema']
        if skema_filter and proposal_skema != skema_filter:
            continue

        # Retrieve the matched text for this proposal and column
        matched_row = original_texts[original_texts['id'] == proposal_id]
        matched_text = ""
        if not matched_row.empty and column in matched_row.columns:
            matched_text = str(matched_row.iloc[0][column]) if pd.notna(matched_row.iloc[0][column]) else ""

        # Calculate advanced similarity metrics
        exact_score = jaccard_similarity(query_text, matched_text)
        fuzzy_score = levenshtein_similarity(query_text, matched_text)
        
        # Lazy load sentence model only when needed
        lazy_load_sentence_model()
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
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        if not column:
            return jsonify({"error": "column is required"}), 400
        if column not in indices:
            valid_columns = list(indices.keys())
            return jsonify({"error": f"Invalid column. Valid columns are: {valid_columns}"}), 400
        results = search_column(query_text, column, skema_filter, top_k)
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
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        bulk_results = []
        for i, text_item in enumerate(texts):
            print(f"[BULK] Processing query {i+1}/{len(texts)}")
            # Handle case where text_item is a list containing a dict
            if isinstance(text_item, list):
                if len(text_item) > 0 and isinstance(text_item[0], dict):
                    text_item = text_item[0]  # Extract the dict from the list
                else:
                    bulk_results.append({
                        "query_index": i,
                        "error": f"Invalid list format: {text_item}"
                    })
                    continue
            
            # Check if it's a dictionary
            if not isinstance(text_item, dict):
                bulk_results.append({
                    "query_index": i,
                    "error": f"Expected dict but got {type(text_item).__name__}: {text_item}"
                })
                continue
                
            skema_filter = text_item.get('skema')
            item_results = []
            # Loop through all possible columns
            for column in indices.keys():
                query_text = text_item.get(column)
                if query_text:
                    print(f"[BULK]   - Searching column '{column}' for query {i+1}")
                    results = search_column(query_text, column, skema_filter, top_k)
                    for result in results:
                        result['searched_column'] = column
                    item_results.extend(results)
            # Sort and limit results
            item_results.sort(key=lambda x: x['final_score'], reverse=True)
            item_results = item_results[:top_k]
            print(f"[BULK]   - Finished query {i+1}, found {len(item_results)} results")
            bulk_results.append({
                "query_index": i,
                "results": item_results,
                "query_info": {
                    "skema_filter": skema_filter,
                    "columns_searched": [col for col in indices.keys() if text_item.get(col)],
                    "total_results": len(item_results)
                }
            })
        return jsonify({
            "bulk_results": bulk_results,
            "total_queries": len(texts)
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

if __name__ == '__main__':
    import sys
    print("Starting plagiarism detection API...")
    load_indices()
    
    # Optional pre-warming (can be disabled with --no-prewarm)
    if '--no-prewarm' not in sys.argv:
        try:
            # Only pre-warm if sentence model gets loaded
            lazy_load_sentence_model()
            pre_warm_embeddings()
        except Exception as e:
            print(f"Pre-warming failed (continuing anyway): {e}")
    
    print("API server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)