import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import joblib
from typing import List, Dict, Any
import os

class QdrantPlagiarismSearch:
    def __init__(self, collection_name="indonesian_proposals", host=None, port=None):
        self.collection_name = collection_name
        
        # Configure Qdrant client
        if host and port:
            # Remote Qdrant server
            self.client = qdrant_client.QdrantClient(host=host, port=port)
            print(f"Connected to Qdrant server at {host}:{port}")
        elif host:
            # Remote Qdrant server with default port
            self.client = qdrant_client.QdrantClient(host=host)
            print(f"Connected to Qdrant server at {host}:6333")
        else:
            # Local in-memory Qdrant
            self.client = qdrant_client.QdrantClient(":memory:")
            print("Using in-memory Qdrant (localhost)")
            
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        # Initialize Indonesian text processing
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = stopword_factory.create_stop_word_remover()
        
        self.text_columns = ['judul', 'ringkasan', 'pendahuluan', 'masalah', 'metode', 'solusi']
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess Indonesian text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.strip()
        text = self.stopword_remover.remove(text)
        text = self.stemmer.stem(text)
        return text
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create sentence embeddings for texts."""
        return self.model.encode(texts).tolist()
    
    def initialize_collection(self, csv_path: str = "proposal.csv"):
        """Initialize Qdrant collection with proposal data."""
        print("Loading proposal data...")
        df = pd.read_csv(csv_path)
        
        # Create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=512,  # distiluse-base-multilingual-cased-v2 dimension
                distance=Distance.COSINE
            )
        )
        
        print("Creating embeddings and indexing...")
        points = []
        
        for idx, row in df.iterrows():
            for column in self.text_columns:
                text = str(row.get(column, ""))
                if text and len(text.strip()) > 10:  # Skip empty/short texts
                    # Create embedding
                    embedding = self.create_embeddings([text])[0]
                    
                    # Store point
                    points.append(PointStruct(
                        id=len(points),
                        vector=embedding,
                        payload={
                            "proposal_id": int(row['id']),
                            "skema": str(row['skema']),
                            "column": column,
                            "text": text,
                            "original_text": text  # Keep original for display
                        }
                    ))
        
        # Batch upload
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Indexed {len(points)} text segments from {len(df)} proposals")
        return len(points)
    
    def search(self, query_text: str, column: str = None, skema_filter: str = None, 
               limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Fast vector similarity search."""
        if not query_text or len(query_text.strip()) < 3:
            return []
        
        # Create query embedding
        query_embedding = self.create_embeddings([query_text])[0]
        
        # Build filter
        search_filter = None
        if column or skema_filter:
            conditions = []
            if column:
                conditions.append(models.FieldCondition(
                    key="column",
                    match=models.MatchValue(value=column)
                ))
            if skema_filter:
                conditions.append(models.FieldCondition(
                    key="skema",
                    match=models.MatchValue(value=skema_filter)
                ))
            search_filter = models.Filter(must=conditions)
        
        # Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=threshold
        )
        
        # Format results
        results = []
        for hit in search_result:
            payload = hit.payload
            results.append({
                "id": payload["proposal_id"],
                "skema": payload["skema"],
                "column": payload["column"],
                "text": payload["text"][:500] + "..." if len(payload["text"]) > 500 else payload["text"],
                "similarity_score": float(hit.score),
                "matched_text": payload["original_text"][:500] + "..." if len(payload["original_text"]) > 500 else payload["original_text"]
            })
        
        return results
    
    def search_bulk(self, texts: List[Dict[str, str]], limit: int = 5, 
                    threshold: float = 0.7) -> List[List[Dict[str, Any]]]:
        """Bulk search for multiple texts."""
        results = []
        
        for i, text_item in enumerate(texts):
            item_results = []
            
            for column in self.text_columns:
                query_text = text_item.get(column)
                if query_text:
                    column_results = self.search(
                        query_text, 
                        column=column, 
                        skema_filter=text_item.get('skema'),
                        limit=limit,
                        threshold=threshold
                    )
                    item_results.extend(column_results)
            
            # Sort by similarity and limit
            item_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            item_results = item_results[:limit]
            
            results.append({
                "query_index": i,
                "results": item_results,
                "total_results": len(item_results)
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "collection_name": self.collection_name,
            "vector_size": info.config.params.vectors.size,
            "distance_metric": str(info.config.params.vectors.distance)
        }

# Global instance
_qdrant_search = None

def get_qdrant_search(host=None, port=None):
    """Get or create global Qdrant search instance.
    
    Args:
        host: Qdrant server host (default: None for in-memory)
        port: Qdrant server port (default: 6333)
    """
    global _qdrant_search
    if _qdrant_search is None:
        # Check environment variables
        import os
        env_host = os.getenv('QDRANT_HOST', host)
        env_port = int(os.getenv('QDRANT_PORT', port or 6333))
        
        _qdrant_search = QdrantPlagiarismSearch(
            host=env_host,
            port=env_port if env_host else None
        )
    return _qdrant_search