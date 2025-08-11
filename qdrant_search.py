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
import difflib

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
        # text = self.stopword_remover.remove(text)
        # text = self.stemmer.stem(text)
        return text
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create sentence embeddings for texts."""
        return self.model.encode(texts).tolist()
    
    def _highlight_similarities(self, query_text: str, matched_text: str) -> str:
        """Highlight similar words/phrases between query and matched text with brackets."""
        if not query_text or not matched_text:
            return matched_text
        
        # Preprocess both texts to normalize for comparison
        query_processed = self.preprocess_text(query_text)
        matched_processed = self.preprocess_text(matched_text)
        
        # Split into words
        query_words = query_processed.split()
        matched_words = matched_processed.split()
        original_words = matched_text.split()
        
        if not query_words or not matched_words:
            return matched_text
        
        # Use SequenceMatcher to find matching blocks
        matcher = difflib.SequenceMatcher(None, query_words, matched_words)
        matching_blocks = matcher.get_matching_blocks()
        
        # Create a set of indices that should be highlighted in the matched text
        highlight_indices = set()
        for match in matching_blocks:
            # match.b is the start index in matched_words, match.size is the length
            if match.size > 0:  # Only consider non-empty matches
                for i in range(match.b, match.b + match.size):
                    highlight_indices.add(i)
        
        # Build the highlighted text
        result_words = []
        i = 0
        while i < len(original_words):
            if i in highlight_indices:
                # Start of a highlighted section
                start_idx = i
                # Find the end of consecutive highlighted words
                while i < len(original_words) and i in highlight_indices:
                    i += 1
                # Add the highlighted section
                highlighted_section = " ".join(original_words[start_idx:i])
                result_words.append(f"[{highlighted_section}]")
            else:
                # Regular word, not highlighted
                result_words.append(original_words[i])
                i += 1
        
        return " ".join(result_words)
    
    
    def initialize_collection(self, csv_path: str = "skripsi_with_skema.csv"):
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
                            "original_text": text,  # Keep original for display
                            "judul": str(row['judul'])  # Add judul field
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
            original_text = payload["original_text"]
            
            if float(hit.score) >= 0.7:
                highlighted_text = self._highlight_similarities(query_text, original_text)
            else:
                highlighted_text = original_text  # atau kosong, tergantung kebutuhan
                
            results.append({
                "id": payload["proposal_id"],
                "skema": payload["skema"],
                "column": payload["column"],
                "text": payload["text"] + "..." if len(payload["text"]) > 500 else payload["text"],
                "similarity_score": float(hit.score),
                "matched_text": highlighted_text,  # Use highlighted text
                "original_highlighted": highlighted_text,  # Add original_highlighted field for frontend
                "judul": payload.get("judul", "")  # Add judul field
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
                "proposal_id": text_item.get('proposal_id'),
                "results": item_results,
                "total_results": len(item_results)
            })
        
        return results
    
    def collection_exists(self) -> bool:
        """Check if the collection exists and has data."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count > 0
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_points": info.points_count,
            "collection_name": self.collection_name,
            "vector_size": info.config.params.vectors.size,
            "distance_metric": str(info.config.params.vectors.distance)
        }
    
    def append_proposals(self, proposals_df: pd.DataFrame, year_threshold: int = 2025) -> Dict[str, Any]:
        """Append new proposals to the existing Qdrant collection.
        
        Args:
            proposals_df: DataFrame containing new proposals
            year_threshold: Minimum year for proposals to be indexed
            
        Returns:
            Dict with indexing results and statistics
        """
        if not self.collection_exists():
            return {
                "error": "Collection does not exist. Please initialize collection first.",
                "indexed_count": 0,
                "filtered_count": 0,
                "total_input": len(proposals_df)
            }
        
        # Filter proposals by year
        if 'tahun' in proposals_df.columns:
            # Ensure year_threshold is numeric, handle null/invalid values
            try:
                if year_threshold is None or year_threshold == 'null' or year_threshold == '':
                    year_threshold = 2025
                else:
                    year_threshold = int(year_threshold)
            except (ValueError, TypeError):
                print(f"Invalid year_threshold value: {year_threshold}, defaulting to 2025")
                year_threshold = 2025
            
            # Convert tahun to numeric, handling any non-numeric values
            proposals_df = proposals_df.copy()  # Avoid modifying original DataFrame
            proposals_df['tahun'] = pd.to_numeric(proposals_df['tahun'], errors='coerce')
            
            # Filter out rows with invalid/NaN years and apply year threshold
            valid_years_mask = proposals_df['tahun'].notna()
            year_threshold_mask = proposals_df['tahun'] >= year_threshold
            
            filtered_df = proposals_df[valid_years_mask & year_threshold_mask].copy()
            filtered_count = len(proposals_df) - len(filtered_df)
            
            print(f"Year filtering: {len(proposals_df)} total -> {len(filtered_df)} after filtering (>= {year_threshold})")
        else:
            filtered_df = proposals_df.copy()
            filtered_count = 0
        
        if len(filtered_df) == 0:
            return {
                "message": f"No proposals found with year >= {year_threshold}",
                "indexed_count": 0,
                "filtered_count": filtered_count,
                "total_input": len(proposals_df),
                "year_threshold": year_threshold
            }
        
        # Get current max point ID to avoid conflicts
        try:
            stats = self.get_stats()
            current_max_id = stats["total_points"]
        except:
            current_max_id = 0
        
        print(f"Appending {len(filtered_df)} proposals to Qdrant collection...")
        points = []
        
        for idx, row in filtered_df.iterrows():
            for column in self.text_columns:
                text = str(row.get(column, ""))
                if text and len(text.strip()) > 10:  # Skip empty/short texts
                    # Create embedding
                    embedding = self.create_embeddings([text])[0]
                    
                    # Store point with unique ID
                    points.append(PointStruct(
                        id=current_max_id + len(points),
                        vector=embedding,
                        payload={
                            "proposal_id": int(row['id']),
                            "skema": str(row['skema']),
                            "column": column,
                            "text": text,
                            "original_text": text,  # Keep original for display
                            "judul": str(row['judul'])  # Add judul field
                        }
                    ))
        
        if points:
            # Batch upload new points
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"Successfully appended {len(points)} text segments from {len(filtered_df)} proposals")
        
        return {
            "message": f"Successfully indexed {len(filtered_df)} proposals",
            "indexed_count": len(filtered_df),
            "text_segments_added": len(points),
            "filtered_count": filtered_count,
            "total_input": len(proposals_df),
            "year_threshold": year_threshold
        }

# Global instance</search>

# Global instance
_qdrant_search = None

def get_qdrant_search(host=None, port=None, force_reinit=False):
    """Get or create global Qdrant search instance.
    
    Args:
        host: Qdrant server host (default: None for in-memory)
        port: Qdrant server port (default: 6333)
        force_reinit: Force recreation of the instance (default: False)
    """
    global _qdrant_search
    if _qdrant_search is None or force_reinit:
        # Check environment variables
        import os
        env_host = os.getenv('QDRANT_HOST', host)
        env_port = int(os.getenv('QDRANT_PORT', port or 6333))
        
        _qdrant_search = QdrantPlagiarismSearch(
            host=env_host,
            port=env_port if env_host else None
        )
    return _qdrant_search