import time
import functools
import logging
from datetime import datetime

# Configure logging for performance monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_monitor')

def performance_monitor(stage_name):
    """
    Decorator to monitor performance of specific functions.
    Logs execution time for performance analysis.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                logger.info(f"{stage_name} completed in {execution_time:.2f}ms")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                logger.error(f"{stage_name} failed after {execution_time:.2f}ms: {str(e)}")
                raise
        return wrapper
    return decorator

class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks.
    """
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = (end_time - self.start_time) * 1000
        if exc_type is None:
            logger.info(f"{self.operation_name} completed in {execution_time:.2f}ms")
        else:
            logger.error(f"{self.operation_name} failed after {execution_time:.2f}ms: {str(exc_val)}")

def log_search_metrics(query_text, column, results_count, total_time_ms):
    """
    Log detailed search metrics for analysis.
    """
    logger.info(f"SEARCH_METRICS - Column: {column}, Query_Length: {len(query_text)}, "
                f"Results: {results_count}, Time: {total_time_ms:.2f}ms, "
                f"Rate: {results_count/total_time_ms*1000:.2f} results/sec")

def log_bulk_metrics(total_queries, total_results, total_time_ms):
    """
    Log bulk search metrics for analysis.
    """
    avg_time_per_query = total_time_ms / total_queries if total_queries > 0 else 0
    logger.info(f"BULK_METRICS - Queries: {total_queries}, Total_Results: {total_results}, "
                f"Total_Time: {total_time_ms:.2f}ms, Avg_Per_Query: {avg_time_per_query:.2f}ms")