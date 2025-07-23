from flask import Flask, request, jsonify
from qdrant_search import get_qdrant_search
import threading
import time

app = Flask(__name__)

# Global Qdrant search instance
qdrant_search = None
initialization_complete = False

def initialize_qdrant():
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
            
        qdrant_search = get_qdrant_search(host=host, port=port)
        qdrant_search.initialize_collection()
        initialization_complete = True
        print("Qdrant search initialized successfully!")
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        initialization_complete = True

@app.route('/search', methods=['POST'])
def search():
    """Fast vector similarity search."""
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
        
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        
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
    """Fast bulk search."""
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
        
        if not texts:
            return jsonify({"error": "texts array is required"}), 400
        
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

if __name__ == '__main__':
    print("Starting fast plagiarism detection API with Qdrant...")
    
    # Start Qdrant initialization in background
    init_thread = threading.Thread(target=initialize_qdrant)
    init_thread.daemon = True
    init_thread.start()
    
    print("API server starting (Qdrant initialization in background)...")
    app.run(debug=True, host='0.0.0.0', port=5001)