# Plagiarism Detection API

Indonesian language plagiarism detection system with separate indexing for each text column.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create indices (run once):
```bash
python indexer.py
```

3. Start the API server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

### Get System Info
```
GET /info
```

### Single Search
```
POST /search
Content-Type: application/json

{
    "query_text": "text to check for plagiarism",
    "column": "judul",
    "skema": "PKM",           // optional - filter by schema
    "top_k": 10              // optional - number of results
}
```

### Bulk Search
```
POST /search_bulk
Content-Type: application/json

{
    "texts": [
        {
            "text": "text to check",
            "column": "ringkasan",
            "skema": "PKM"       // optional
        },
        {
            "text": "another text",
            "column": "metode"
        }
    ],
    "top_k": 5               // optional
}
```

## Available Columns

- `judul` - Title
- `ringkasan` - Summary  
- `pendahuluan` - Introduction
- `masalah` - Problem statement
- `metode` - Method
- `solusi` - Solution

## Response Format

Search results include:
- `id`: Proposal ID
- `skema`: Schema type
- `similarity_score`: Similarity score (0-1)
- `column`: Column that was searched

## Files Structure

- `indexer.py` - Creates TF-IDF indices for each column
- `app.py` - Flask API server
- `indices/` - Directory containing all index files
- `test_api.py` - API testing script