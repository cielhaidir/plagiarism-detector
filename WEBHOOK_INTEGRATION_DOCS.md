# Plagiarism Check API - Webhook Integration Documentation

## Overview

Both plagiarism check APIs (`app.py` on port 5000 and `fast_api.py` on port 5001) now support webhook-based asynchronous processing to prevent timeout issues during long-running search operations.

## How Webhook Integration Works

### 1. Request with Webhook URL
Add a `webhook_url` parameter to your search requests to enable asynchronous processing:

```json
{
  "query_text": "sample text to search",
  "column": "judul",
  "skema": "PKM-GT",
  "top_k": 10,
  "threshold": 0.7,
  "webhook_url": "https://your-frontend.com/webhook/plagiarism"
}
```

### 2. Immediate Response
When webhook URL is provided, the API immediately returns a job tracking response:

```json
{
  "job_id": "7e086c73-704d-4f24-bfc5-35f380bce377",
  "status": "processing", 
  "message": "Search started. Results will be sent to webhook when complete.",
  "webhook_url": "https://your-frontend.com/webhook/plagiarism"
}
```

**HTTP Status:** `202 Accepted`

### 3. Webhook Callback
When processing completes, the API sends a POST request to your webhook URL with the results.

## Webhook Payload Structures

### Single Search (`/search`)

#### Success Payload
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "timestamp": "2025-01-08T19:11:31.123456",
  "results": [
    {
      "id": 123,
      "skema": "PKM-GT",
      "column": "judul",
      "similarity_score": 0.85,
      "exact_score": 0.75,
      "fuzzy_score": 0.80,
      "semantic_score": 0.90,
      "final_score": 0.82,
      "matched_text": "This is the [matching text] from proposal...",
      "original_highlighted": "This is the [matching text] from proposal...",
      "judul": "Title of the matching proposal"
    }
  ],
  "query_info": {
    "column": "judul",
    "skema_filter": "PKM-GT",
    "total_results": 5,
    "threshold": 0.7
  }
}
```

#### Error Payload
```json
{
  "job_id": "uuid-string",
  "status": "failed",
  "timestamp": "2025-01-08T19:11:31.123456",
  "error": "Error message describing what went wrong"
}
```

### Bulk Search (`/search_bulk`)

#### Success Payload
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "timestamp": "2025-01-08T19:11:31.123456",
  "bulk_results": [
    {
      "query_index": 0,
      "results": [
        {
          "id": 123,
          "skema": "PKM-GT", 
          "similarity_score": 0.85,
          "exact_score": 0.75,
          "fuzzy_score": 0.80,
          "semantic_score": 0.90,
          "final_score": 0.82,
          "column": "judul",
          "searched_column": "judul",
          "matched_text": "This is the [matching text]...",
          "original_highlighted": "This is the [matching text]...",
          "judul": "Title of matching proposal"
        }
      ],
      "query_info": {
        "skema_filter": "PKM-GT",
        "columns_searched": ["judul", "ringkasan"],
        "total_results": 2
      }
    },
    {
      "query_index": 1,
      "results": [...],
      "query_info": {...}
    }
  ],
  "total_queries": 50
}
```

#### Error Payload
```json
{
  "job_id": "uuid-string", 
  "status": "failed",
  "timestamp": "2025-01-08T19:11:31.123456",
  "error": "Error message describing what went wrong"
}
```

## API Endpoints

### App.py (Port 5000) - Traditional API

#### Single Search
```
POST http://localhost:5000/search
```

**Request Body:**
```json
{
  "query_text": "text to search for plagiarism",
  "column": "judul|ringkasan|pendahuluan|masalah|metode|solusi",
  "skema": "PKM-GT|PKM-RE|etc", // optional
  "top_k": 10, // optional, default 10
  "webhook_url": "https://your-webhook-endpoint.com" // optional
}
```

#### Bulk Search  
```
POST http://localhost:5000/search_bulk
```

**Request Body:**
```json
{
  "texts": [
    {
      "judul": "Title text to check",
      "ringkasan": "Summary text to check",
      "skema": "PKM-GT",
      "proposal_id": 123
    },
    {
      "metode": "Method text to check",
      "skema": "PKM-RE",
      "proposal_id": 456
    }
  ],
  "top_k": 5, // optional, default 1
  "webhook_url": "https://your-webhook-endpoint.com" // optional
}
```

### Fast_api.py (Port 5001) - Qdrant-based API

#### Single Search
```
POST http://localhost:5001/search
```

**Request Body:**
```json
{
  "query_text": "text to search for plagiarism",
  "column": "judul|ringkasan|pendahuluan|masalah|metode|solusi", // optional
  "skema": "PKM-GT|PKM-RE|etc", // optional  
  "top_k": 10, // optional, default 10
  "threshold": 0.7, // optional, default 0.7
  "webhook_url": "https://your-webhook-endpoint.com" // optional
}
```

#### Bulk Search
```
POST http://localhost:5001/search_bulk
```

**Request Body:**
```json
{
  "texts": [
    {
      "judul": "Title text to check",
      "ringkasan": "Summary text to check",
      "skema": "PKM-GT",
      "proposal_id": 123
    }
  ],
  "top_k": 5, // optional, default 5
  "threshold": 0.7, // optional, default 0.7
  "webhook_url": "https://your-webhook-endpoint.com" // optional
}
```

## Key Differences Between APIs

### App.py (Traditional)
- Uses TF-IDF + cosine similarity
- Requires `column` parameter for single search
- Provides multiple similarity scores (exact, fuzzy, semantic, final)
- More detailed result analysis

### Fast_api.py (Qdrant)
- Uses sentence transformer embeddings + vector search
- `column` parameter optional (searches all if not specified)
- Single similarity score from vector similarity
- Faster performance for large datasets
- Provides text highlighting with brackets `[matched text]`

## Frontend Integration Examples

### JavaScript/Fetch Example

```javascript
// Start asynchronous plagiarism check
async function startPlagiarismCheck(queryData) {
  const response = await fetch('http://localhost:5000/search_bulk', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
    },
    body: JSON.stringify({
      ...queryData,
      webhook_url: 'https://your-app.com/api/webhook/plagiarism'
    })
  });
  
  if (response.status === 202) {
    const data = await response.json();
    console.log('Job started:', data.job_id);
    // Show loading indicator to user
    showLoadingIndicator(data.job_id);
  }
}

// Webhook endpoint handler (backend)
app.post('/api/webhook/plagiarism', (req, res) => {
  const { job_id, status, bulk_results, error } = req.body;
  
  if (status === 'completed') {
    // Process bulk_results
    bulk_results.forEach(item => {
      const { query_index, results } = item;
      results.forEach(result => {
        // Update database with plagiarism results
        updatePlagiarismResult(query_index, result);
      });
    });
  } else if (status === 'failed') {
    console.error('Plagiarism check failed:', error);
  }
  
  res.status(200).json({ received: true });
});
```

### PHP/Laravel Example

```php
// Start plagiarism check
public function startPlagiarismCheck(Request $request) 
{
    $response = Http::post('http://localhost:5000/search_bulk', [
        'texts' => $request->texts,
        'top_k' => $request->top_k ?? 5,
        'webhook_url' => route('webhook.plagiarism')
    ]);
    
    if ($response->status() === 202) {
        $data = $response->json();
        return response()->json([
            'success' => true,
            'job_id' => $data['job_id'],
            'message' => 'Plagiarism check started'
        ]);
    }
}

// Webhook handler
public function handlePlagiarismWebhook(Request $request)
{
    $jobId = $request->job_id;
    $status = $request->status;
    
    if ($status === 'completed') {
        $bulkResults = $request->bulk_results;
        
        foreach ($bulkResults as $item) {
            $queryIndex = $item['query_index'];
            $results = $item['results'];
            
            foreach ($results as $result) {
                // Update proposal with plagiarism result
                Proposal::where('id', $result['id'])
                    ->update([
                        'plagiarism_score' => $result['final_score'] ?? $result['similarity_score'],
                        'plagiarism_details' => json_encode($result)
                    ]);
            }
        }
    }
    
    return response()->json(['received' => true]);
}
```

## Backward Compatibility

Both APIs still support synchronous operation when no `webhook_url` is provided:

```javascript
// Synchronous operation (original behavior)
const response = await fetch('http://localhost:5000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query_text: "sample text",
    column: "judul"
    // No webhook_url = synchronous processing
  })
});

const data = await response.json();
console.log(data.results); // Results returned immediately
```

## Error Handling

### Webhook Delivery Failures
- The API will attempt to send to webhook with 30-second timeout
- Failed webhook deliveries are logged but don't affect the search processing
- No retry mechanism is implemented (implement on your webhook endpoint if needed)

### Webhook Endpoint Requirements
Your webhook endpoint should:
1. Accept POST requests
2. Handle JSON payloads
3. Return 2xx status code for successful receipt
4. Process webhook data asynchronously to avoid timeout

### Common Issues and Solutions

1. **"Undefined array key 'bulk_results'"** - Check that your webhook handler expects the correct payload structure shown above

2. **Webhook not received** - Verify your webhook URL is publicly accessible and accepts POST requests

3. **Request timeout** - Use webhook mode for long-running operations instead of synchronous requests

## Security Considerations

- Implement API key validation on your webhook endpoints
- Validate webhook payload structure before processing
- Consider implementing webhook signature verification for production
- Log all webhook activities for monitoring and debugging

## Testing Webhook Integration

```bash
# Test webhook with curl
curl -X POST http://localhost:5000/search_bulk \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [{"judul": "Test plagiarism check", "skema": "PKM-GT"}],
    "top_k": 3,
    "webhook_url": "https://httpbin.org/post"
  }'
```

This will send results to httpbin.org which echoes back the received payload for testing.

## Important Field Clarification

### `proposal_id` vs `id` Fields

- **`proposal_id`**: The ID of the original proposal being checked for plagiarism (from your request)
- **`id`**: The ID of the matched/similar proposal found in the database

**Example:**
```json
{
  "bulk_results": [
    {
      "query_index": 0,
      "proposal_id": 123,  // Original proposal you're checking
      "results": [
        {
          "id": 275,  // Matched proposal found in database
          "column": "metode",
          "similarity_score": 0.85,
          "matched_text": "Similar content found..."
        }
      ]
    }
  ]
}
```

This means: "Proposal 123 (your proposal) has similar content to Proposal 275 (found in database)."

Make sure to include `proposal_id` in each text item when making bulk requests:
```json
{
  "texts": [
    {
      "judul": "Proposal title to check",
      "ringkasan": "Summary to check",
      "proposal_id": 123,  // Required: ID of proposal being checked
      "skema": "PKM-GT"
    }
  ]
}
```