# Gunicorn configuration file for production deployment

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - reduced to prevent memory issues
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for ML processing
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 100  # Reduced to prevent memory accumulation
max_requests_jitter = 10

# Memory management
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance
worker_rlimit_as = 2147483648  # 2GB memory limit per worker

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "plagiarism_api"

# Disable preload to avoid memory sharing issues with ML models
preload_app = False

# Signal handling for graceful shutdowns
graceful_timeout = 30

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Threading settings for ML model compatibility
thread_workers = 1