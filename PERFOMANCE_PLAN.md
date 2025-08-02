Performance Improvement Plan: Plagiarism Detection API
Objective: To enhance the performance, scalability, and security of the Flask-based plagiarism detection API. This plan outlines a phased approach to address current bottlenecks and prepare the application for production-level workloads.

Current Date: Saturday, 2 August 2025

Phase 1: Immediate Optimizations (Next 1-3 Days)
This phase focuses on critical, low-effort changes that will yield significant improvements in performance and security.

1. Optimize DataFrame Lookups (High Priority)

Problem: The current code uses original_texts[original_texts['id'] == proposal_id] inside a loop, which is inefficient for large datasets as it performs a full scan of the DataFrame on each iteration.

Action:

In the load_indices function, set the id column as the index for both the metadata and original_texts DataFrames immediately after loading them from CSV.

# In load_indices()
metadata = pd.read_csv('indices/metadata.csv')
metadata.set_index('id', inplace=True)

original_texts = pd.read_csv('skripsi_with_skema.csv')
original_texts.set_index('id', inplace=True)

In the search_column function, modify the lookup to use the much faster .loc method.

# In search_column()
proposal_id = metadata.index[idx]
matched_row = original_texts.loc[[proposal_id]]

Expected Outcome: Drastically reduced search latency, especially for queries that return many potential matches from the initial TF-IDF stage.

2. Implement a Production-Grade WSGI Server (Critical for Deployment)

Problem: The application is currently run using app.run(debug=True), which is the Flask development server. It is not secure, stable, or performant enough for a live environment.

Action:

Install a production WSGI server like Gunicorn: pip install gunicorn.

Run the application using Gunicorn, specifying multiple workers to handle concurrent requests.

gunicorn --workers 4 --bind 0.0.0.0:5000 your_app_module:app

Expected Outcome: Increased stability, security, and throughput. The application will be able to handle multiple simultaneous requests efficiently.

3. Patch Webhook SSRF Vulnerability (Critical Security Fix)

Problem: The webhook_url parameter is not validated, creating a Server-Side Request Forgery (SSRF) vulnerability. An attacker could use this to scan your internal network or attack internal services.

Action:

Implement a function to validate the webhook URL. This function should ensure the URL uses http or httpshttps and does not resolve to a private or reserved IP address.

Call this validation function in both the /search and /search_bulk endpoints before initiating the background thread. Return a 400 Bad Request if validation fails.

Expected Outcome: Mitigates a critical security risk, preventing the API from being used for malicious purposes.

Phase 2: Code Refactoring & Scalability (Next 1-2 Weeks)
This phase focuses on improving code quality and maintainability to support future development.

1. Refactor and Deduplicate API Logic

Problem: The synchronous and asynchronous (webhook) paths within the /search and /search_bulk endpoints contain significant code duplication.

Action:

Create private helper functions (e.g., _process_single_search, _process_bulk_search) that contain the core search logic.

Have the API endpoint functions handle only request parsing and response formatting. They will call the helper functions either directly (for synchronous requests) or within a thread (for asynchronous requests).

Expected Outcome: Cleaner, more maintainable code (Don't Repeat Yourself - DRY principle). Future changes to the search logic will only need to be made in one place.

2. Enforce a Stricter API Contract

Problem: The /search_bulk endpoint contains complex logic to handle malformed inputs (e.g., a list containing a dict). This makes the code brittle and harder to read.

Action:

Simplify the code by removing the special handling for malformed items.

Instead, perform a validation check at the beginning of the loop. If an item is not in the expected format (e.g., a dictionary), immediately return a 400 Bad Request error with a descriptive message.

Expected Outcome: A more robust and predictable API. Clients will receive clear feedback on invalid requests, and the server-side code will be simpler.

Phase 3: Architectural Enhancements (Long-Term)
This phase involves more significant architectural changes to ensure the application can scale to handle high-volume traffic.

1. Replace threading with a Dedicated Task Queue

Problem: Using the threading module for background tasks is not scalable in a multi-process environment (like Gunicorn with multiple workers) and lacks features like automatic retries and persistent jobs.

Action:

Integrate a distributed task queue system like Celery.

Use a message broker like Redis or RabbitMQ to manage the queue of tasks.

Convert the async_search and async_bulk_search functions into Celery tasks. The API endpoints will now push jobs to the queue instead of spawning threads.

Expected Outcome: A highly scalable and resilient system for background processing. The API can handle a massive influx of requests without being overwhelmed, and tasks are guaranteed to be processed even if the application restarts.

2. Implement a Caching Layer

Problem: Identical or very similar queries will re-compute all similarity scores every time, consuming unnecessary CPU resources.

Action:

Use a caching solution like Redis to store the results of recently processed queries.

Before executing a search, generate a unique key based on the query parameters (query_text, column, skema_filter) and check if the result exists in the cache. If so, return the cached result.

Expected Outcome: Significantly lower response times for frequent queries and reduced overall server load.

Monitoring & Measurement
To validate the effectiveness of these changes, it's crucial to measure performance before and after implementation.

Benchmarking: Use a tool like ab (Apache Benchmark) or wrk to load-test your API endpoints. Measure requests per second and average latency for both simple and bulk searches.

Profiling: Use a Python profiler to identify any remaining CPU or memory bottlenecks in the code.

Logging: Enhance logging to include the processing time for each stage of the search (TF-IDF, advanced metrics) to pinpoint areas for further optimization.