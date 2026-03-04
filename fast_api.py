from flask import Flask, request, jsonify
from qdrant_search import get_qdrant_search
import threading
import time
import requests
import uuid
from datetime import datetime
import pandas as pd
import sys
sys.path.append('.')
# Remove import of append_indices - we'll use qdrant_search.append_proposals instead

app = Flask(__name__)

# Global Qdrant search instance
qdrant_search = None
initialization_complete = True

def initialize_qdrant(force_reinit=False):
    """Initialize Qdrant search in background."""
    global qdrant_search, initialization_complete
    try:
        import os
        host = os.getenv('QDRANT_HOST')
        port = int(os.getenv('QDRANT_PORT', 6333))
        
        print(f"Initializing Qdrant-based plagiarism search...")
        if host:
            print(f"Connecting to Qdrant server: {host}:{port}")
        else:
            print("Using in-memory Qdrant (localhost)")
            
        qdrant_search = get_qdrant_search(host=host, port=port, force_reinit=force_reinit)
        if force_reinit or not hasattr(qdrant_search, '_collection_initialized'):
            qdrant_search.initialize_collection()
            qdrant_search._collection_initialized = True
        initialization_complete = True
        print("Qdrant search initialized successfully!")
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        initialization_complete = True

@app.route('/search', methods=['POST'])
def search():
    """Fast vector similarity search with webhook support."""
    if not initialization_complete:
        return jsonify({"error": "System is still initializing, please wait..."}), 503
    
    if qdrant_search is None:
        return jsonify({"error": "Qdrant search not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query_text = data.get('query_text')
        column = data.get('column')
        skema_filter = data.get('skema')
        limit = data.get('top_k', 10)
        threshold = data.get('threshold', 0.7)
        webhook_url = data.get('webhook_url')
        proposal_id = data.get('proposal_id', None)
        
        
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        
        # If webhook URL is provided, process asynchronously
        if webhook_url:
            job_id = str(uuid.uuid4())
            
            def async_search():
                try:
                    results = qdrant_search.search(
                        query_text=query_text,
                        column=column,
                        skema_filter=skema_filter,
                        limit=limit,
                        threshold=threshold
                    )
                    
                    webhook_payload = {
                        "job_id": job_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "results": results,
                        "proposal_id": proposal_id,
                        "query_info": {
                            "column": column,
                            "skema_filter": skema_filter,
                            "total_results": len(results),
                            "threshold": threshold
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
            results = qdrant_search.search(
                query_text=query_text,
                column=column,
                skema_filter=skema_filter,
                limit=limit,
                threshold=threshold
            )
            
            return jsonify({
                "results": results,
                "query_info": {
                    "column": column,
                    "skema_filter": skema_filter,
                    "total_results": len(results),
                    "threshold": threshold
                }
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_bulk', methods=['POST'])
def search_bulk():
    """Fast bulk search with webhook support."""
    if not initialization_complete:
        return jsonify({"error": "System is still initializing, please wait..."}), 503
    
    if qdrant_search is None:
        return jsonify({"error": "Qdrant search not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        texts = data.get('texts', [])
        limit = data.get('top_k', 5)
        threshold = data.get('threshold', 0.7)
        webhook_url = data.get('webhook_url')
        
        print(f"Webhook URL: {webhook_url}")
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        
        # If webhook URL is provided, process asynchronously
        if webhook_url:
            job_id = str(uuid.uuid4())
            
            def async_bulk_search():
                try:
                    results = qdrant_search.search_bulk(
                        texts=texts,
                        limit=limit,
                        threshold=threshold
                    )
                    
                    # # Transform results to match frontend expectations
                    # transformed_results = []
                    # for bulk_item in results:
                    #     query_index = bulk_item["query_index"]
                    #     item_results = bulk_item["results"]
                        
                    #     # For each query, create entries for each matching proposal
                    #     for result in item_results:
                    #         transformed_results.append({
                    #             "query_index": query_index,
                    #             "proposal_id": result["id"],
                    #             "results": {
                    #                 "similarity_score": result["similarity_score"],
                    #                 "column": result["column"],
                    #                 "skema": result["skema"],
                    #                 "matched_text": result.get("matched_text", result.get("text", "")),
                    #                 "judul": result.get("judul", ""),
                    #                 "original_highlighted": result.get("original_highlighted", result.get("matched_text", result.get("text", "")))
                    #             }
                    #         })
                    
                    # Add proposal_id to each bulk result item from input parameters
                    enhanced_results = []
                    for i, result_item in enumerate(results):
                        enhanced_item = result_item.copy()
                        # Get proposal_id from original input text item
                        if i < len(texts):
                            enhanced_item["proposal_id"] = texts[i].get('proposal_id')
                        else:
                            enhanced_item["proposal_id"] = None
                        enhanced_results.append(enhanced_item)
                    
                    webhook_payload = {
                        "job_id": job_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "bulk_results": enhanced_results,
                        "total_queries": len(texts)
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
                "message": "Bulk search started. Results will be sent to webhook when complete.",
                "webhook_url": webhook_url
            }), 202
        
        # Synchronous processing (original behavior)
        else:
            results = qdrant_search.search_bulk(
                texts=texts,
                limit=limit,
                threshold=threshold
            )
            
            return jsonify({
                "bulk_results": results,
                "total_queries": len(texts)
            })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if initialization_complete else "initializing",
        "qdrant_ready": qdrant_search is not None,
        "initialization_complete": initialization_complete
    })

@app.route('/info', methods=['GET'])
def info():
    """Get system information."""
    if not initialization_complete or qdrant_search is None:
        return jsonify({"error": "System not ready"}), 503
    
    try:
        stats = qdrant_search.get_stats()
        return jsonify({
            "search_engine": "Qdrant",
            "available_columns": qdrant_search.text_columns,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get detailed statistics."""
    if not initialization_complete or qdrant_search is None:
        return jsonify({"error": "System not ready"}), 503
    
    try:
        stats = qdrant_search.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/refresh', methods=['GET'])
def refresh():
    """Force refresh/reinitialize the Qdrant collection."""
    global qdrant_search, initialization_complete
    
    try:
        initialization_complete = False
        print("Manual refresh triggered - reinitializing Qdrant collection...")
        
        # Start reinitialization in background thread
        def refresh_worker():
            initialize_qdrant(force_reinit=True)
        
        refresh_thread = threading.Thread(target=refresh_worker)
        refresh_thread.daemon = True
        refresh_thread.start()
        
        return jsonify({
            "message": "Qdrant collection refresh initiated",
            "status": "refreshing"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/index_proposals', methods=['POST'])
def index_proposals():
    """
    Endpoint to index new proposals with year filtering (2025+).
    Supports both synchronous and asynchronous processing with webhook support.
    """
    if not initialization_complete:
        return jsonify({"error": "System is still initializing, please wait..."}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        proposals = data.get('proposals', [])
        webhook_url = data.get('webhook_url')
        year_threshold_raw = data.get('year_threshold', 2025)
        
        # Handle null/invalid year_threshold values
        try:
            if year_threshold_raw is None or year_threshold_raw == 'null' or year_threshold_raw == '':
                year_threshold = 2025
            else:
                year_threshold = int(year_threshold_raw)
        except (ValueError, TypeError):
            print(f"Invalid year_threshold value: {year_threshold_raw}, defaulting to 2025")
            year_threshold = 2025
        
        if not proposals:
            return jsonify({"error": "proposals array is required"}), 400
        
        # Validate proposal structure
        required_fields = ['id', 'judul', 'skema', 'tahun', 'pengusul']
        optional_fields = ['year']  # 'year' is optional since we can fall back to 'tahun'
        text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
        
        validation_errors = []
        for i, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                validation_errors.append(f"Proposal at index {i} must be an object")
                continue
                
            missing_fields = [field for field in required_fields if field not in proposal]
            if missing_fields:
                validation_errors.append(f"Proposal at index {i} missing required fields: {missing_fields}")
        
        if validation_errors:
            return jsonify({
                "error": "Validation failed",
                "details": validation_errors
            }), 400
        
        # If webhook URL is provided, process asynchronously
        if webhook_url:
            job_id = str(uuid.uuid4())
            
            def async_index_proposals():
                try:
                    # Convert proposals to DataFrame
                    proposals_df = pd.DataFrame(proposals)
                    
                    # Process indexing using Qdrant search instance
                    result = qdrant_search.append_proposals(proposals_df, year_threshold)
                    
                    webhook_payload = {
                        "job_id": job_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "result": result,
                        "year_threshold": year_threshold
                    }
                    
                    # Send results to webhook
                    requests.post(webhook_url, json=webhook_payload, timeout=30)
                    print(f"Indexing results sent to webhook for job {job_id}")
                    
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
                    print(f"Indexing failed for job {job_id}: {e}")
            
            # Start async processing
            index_thread = threading.Thread(target=async_index_proposals)
            index_thread.daemon = True
            index_thread.start()
            
            return jsonify({
                "job_id": job_id,
                "status": "processing",
                "message": "Indexing started. Results will be sent to webhook when complete.",
                "webhook_url": webhook_url,
                "year_threshold": year_threshold
            }), 202
        
        # Synchronous processing
        else:
            proposals_df = pd.DataFrame(proposals)
            result = qdrant_search.append_proposals(proposals_df, year_threshold)
            
            return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting fast plagiarism detection API with Qdrant...")
    
    # Only initialize once at startup
    print("Connecting to Qdrant (without initializing collection)...")
    import os
    host = os.getenv('QDRANT_HOST')
    port = int(os.getenv('QDRANT_PORT', 6333))
    qdrant_search = get_qdrant_search(host=host, port=port)
    initialization_complete = True
    print("Qdrant connection established. Use /refresh to initialize collection data.")
    
    print("API server starting...")
    app.run(debug=True, host='0.0.0.0', port=5001)