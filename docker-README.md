# Docker Setup for Plagiarism Detection System

This Docker setup provides a complete containerized plagiarism detection system with Qdrant vector search.

## Quick Start

### 1. Start the Fast Qdrant-based System
```bash
docker-compose up -d
```

### 2. Start with Original TF-IDF API
```bash
docker-compose --profile original up -d
```

### 3. Start with Load Balancer
```bash
docker-compose --profile production up -d
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **qdrant** | 6333 | Qdrant vector database |
| **plagiarism_api** | 5001 | Fast Qdrant-based API |
| **plagiarism_api_original** | 5000 | Original TF-IDF API |
| **nginx** | 80 | Load balancer (production) |

## API Endpoints

### Fast Qdrant API (Port 5001)
- `POST /search` - Single search
- `POST /search_bulk` - Bulk search
- `GET /health` - Health check
- `GET /stats` - System statistics

### Original API (Port 5000)
- `POST /search` - Single search
- `POST /search_bulk` - Bulk search
- `GET /health` - Health check

## Usage Examples

### 1. Single Search
```bash
curl -X POST http://localhost:5001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "pengembangan aplikasi mobile untuk pendidikan",
    "column": "judul",
    "top_k": 5,
    "threshold": 0.7
  }'
```

### 2. Bulk Search
```bash
curl -X POST http://localhost:5001/search_bulk \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {
        "judul": "pengembangan aplikasi mobile",
        "ringkasan": "aplikasi untuk pendidikan"
      }
    ],
    "top_k": 3,
    "threshold": 0.5
  }'
```

### 3. Check Health
```bash
curl http://localhost:5001/health
```

## Environment Configuration

Create `.env` file:
```bash
# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# API Configuration
API_HOST=0.0.0.0
API_PORT=5001

# Search Configuration
DEFAULT_THRESHOLD=0.7
DEFAULT_LIMIT=10
```

## Docker Commands

### Build and Start
```bash
docker-compose up --build
```

### View Logs
```bash
docker-compose logs -f plagiarism_api
docker-compose logs -f qdrant
```

### Stop Services
```bash
docker-compose down
```

### Clean Up
```bash
docker-compose down -v
docker system prune -f
```

## Performance Testing

### Test Fast API
```bash
python test_qdrant.py
```

### Test Original API
```bash
python test_api.py
```

## Production Deployment

### With Load Balancer
```bash
docker-compose --profile production up -d
```

Access via:
- Fast API: http://localhost/api/fast/
- Original API: http://localhost/api/original/

## Troubleshooting

### Check Service Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs
```

### Restart Services
```bash
docker-compose restart
```

### Access Container Shell
```bash
docker-compose exec plagiarism_api bash
```

## Data Persistence

- **Qdrant Data**: Stored in `qdrant_storage` volume
- **Application Data**: Mounted from local `./proposal.csv`
- **Logs**: Available via `docker-compose logs`

## Resource Requirements

- **Memory**: ~2GB RAM
- **Storage**: ~500MB for Qdrant + ~100MB for images
- **CPU**: 2+ cores recommended