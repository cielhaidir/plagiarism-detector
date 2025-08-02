# macOS-compatible Gunicorn configuration
# Addresses fork() issues with PyTorch and sentence-transformers

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Single worker to avoid fork() issues on macOS
workers = 16
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Disable preloading to avoid fork() issues
preload_app = False

# Memory management
max_requests = 20  # Very frequent restarts to handle memory issues
max_requests_jitter = 2

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "plagiarism_api_macos"

# Graceful handling
graceful_timeout = 120

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# macOS specific settings
worker_tmp_dir = None  # Don't use shared memory on macOS