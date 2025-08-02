# Troubleshooting Guide for Performance Improvements

## Common Issues and Solutions

### 1. NumPy Version Compatibility Error

**Error Message:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.2 as it may crash
```

**Solution:**
The `requirements.txt` has been updated to use `numpy<2.0.0` and `scikit-learn==1.6.1` for compatibility.

```bash
# Reinstall dependencies with fixed versions
pip install --upgrade -r requirements.txt
```

### 2. Gunicorn Worker Crashes (SIGSEGV)

**Error Message:**
```
Worker (pid:XXXX) was sent SIGSEGV!
```

**Root Cause:**
ML models (sentence transformers, scikit-learn) don't work well with Gunicorn's multi-worker setup due to memory sharing issues.

**Solutions:**

#### Option A: Use Single-Worker Configuration (Recommended)
```bash
./start_production_simple.sh
```

#### Option B: Use Threading Instead of Multi-Processing
```bash
gunicorn --workers 1 --threads 4 app:app
```

#### Option C: Development Mode
```bash
python app.py
```

### 3. Memory Issues

**Symptoms:**
- High memory usage
- Worker restarts
- Slow performance

**Solutions:**
- Use the simplified configuration with frequent worker restarts
- Monitor memory usage with `htop` or similar tools
- Consider reducing `max_requests` in Gunicorn config

### 4. Model Loading Issues

**Symptoms:**
- Long startup times
- Import errors
- Models not found

**Solutions:**
- Ensure all required files exist:
  - `indices/` directory with all `.pkl` files
  - `skripsi_with_skema.csv`
- Check file permissions
- Verify virtual environment activation

### 5. Slow Performance After Optimization

**Debugging Steps:**
1. Check if indices are properly loaded:
   ```bash
   curl http://localhost:5000/health
   ```

2. Monitor performance logs for timing information

3. Verify DataFrame indexing is working:
   - Look for "TF-IDF Search" timing logs
   - Check if search times are reasonable (<100ms for small datasets)

### 6. Port Already in Use

**Error Message:**
```
[Errno 48] Address already in use
```

**Solution:**
```bash
# Find and kill existing process
lsof -ti:5000 | xargs kill -9

# Or use a different port
gunicorn --bind 0.0.0.0:5001 --config gunicorn_simple.conf.py app:app
```

## Deployment Recommendations

### For Development:
```bash
python app.py
```

### For Testing/Staging:
```bash
./start_production_simple.sh
```

### For Production (Small-Medium Load):
```bash
./start_production_simple.sh
```

### For Production (High Load):
Consider using multiple single-worker instances behind a load balancer instead of multi-worker Gunicorn.

## Performance Monitoring

### Check Application Health:
```bash
curl http://localhost:5000/health
```

### Monitor Logs:
```bash
# Watch application logs
tail -f /path/to/gunicorn.log

# Monitor system resources
htop
```

### Load Testing:
```bash
# Simple load test
ab -n 100 -c 5 -T application/json -p test_payload.json http://localhost:5000/search

# Run validation tests
./test_performance_improvements.py
```

## Version Compatibility Matrix

| Component | Working Version | Issues with |
|-----------|----------------|-------------|
| NumPy | < 2.0.0 | >= 2.0.0 (binary incompatibility) |
| scikit-learn | 1.6.1 | 1.7.x (model format changes) |
| Flask | >= 2.0 | < 2.0 (missing features) |
| Gunicorn | >= 20.0 | < 20.0 (config format) |

## Getting Help

If you continue to experience issues:

1. Check the error logs carefully
2. Verify all dependencies are correctly installed
3. Try the single-worker configuration first
4. Ensure all required data files are present
5. Test with a simple request using curl or the test script

For development, consider using the Flask development server (`python app.py`) which is more forgiving of these issues.