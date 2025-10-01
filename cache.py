import time
import hashlib
import pickle
import os
from typing import Any, Optional, Dict, Tuple
import config  # Direct module import for better performance

class Cache:
    """Optimized unified cache system for path integral simulations"""
    
    # Cache statistics
    _stats = {
        'hits': 0,
        'misses': 0,
        'sets': 0,
        'memory_usage': 0
    }
    
    @staticmethod
    def _make_param_str(params: dict) -> str:
        """Create deterministic parameter string for cache keys"""
        try:
            # Sort and normalize parameters
            items = []
            for k in sorted(params.keys()):
                v = params[k]
                # Handle different types consistently
                if isinstance(v, float):
                    v = f"{v:.10f}"  # Consistent float precision
                elif isinstance(v, (list, tuple)):
                    v = str(sorted(v) if all(isinstance(x, (int, float, str)) for x in v) else v)
                items.append((k, str(v)))
            return '|'.join(f"{k}={v}" for k, v in items)
        except Exception:
            # Fallback: best-effort string
            return str(hash(str(params)))
    
    @staticmethod
    def _generate_cache_key(simulation_type: str, operation: str, **params) -> str:
        """Generate optimized cache key with hash for long keys"""
        param_str = Cache._make_param_str(params)
        base_key = f"{simulation_type}::{operation}::{param_str}"
        
        # Use hash for very long keys to avoid filesystem issues
        if len(base_key) > 200:
            key_hash = hashlib.md5(base_key.encode()).hexdigest()
            return f"{simulation_type}::{operation}::{key_hash}"
        return base_key

    @staticmethod
    def generate_typed_cache_key(simulation_type: str, operation: str, **params) -> str:
        """Generate typed cache key (legacy compatibility)"""
        return f"TYPED::{Cache._generate_cache_key(simulation_type, operation, **params)}"

    @staticmethod
    def get_typed_cached_result(simulation_type: str, operation: str, expected_type=None, **params):
        """Get cached result with type checking and statistics"""
        key = Cache.generate_typed_cache_key(simulation_type, operation, **params)
        entry = config._ANALYSIS_CACHE.get(key)
        
        if entry is None:
            Cache._stats['misses'] += 1
            return None
            
        Cache._stats['hits'] += 1
        
        # Handle new-format entry
        if isinstance(entry, dict) and 'data' in entry:
            data = entry.get('data')
            # Check expiration (24 hours default)
            if time.time() - entry.get('timestamp', 0) > 86400:
                del config._ANALYSIS_CACHE[key]
                Cache._stats['misses'] += 1
                return None
            if expected_type is not None and not isinstance(data, expected_type):
                return None
            return data
            
        # Legacy/raw entry
        if expected_type is not None and not isinstance(entry, expected_type):
            return None
        return entry

    @staticmethod
    def set_typed_cached_result(simulation_type: str, operation: str, result, **params):
        """Set cached result with metadata and size tracking"""
        key = Cache.generate_typed_cache_key(simulation_type, operation, **params)
        
        # Estimate memory usage
        try:
            size_estimate = len(pickle.dumps(result))
            Cache._stats['memory_usage'] += size_estimate
        except:
            size_estimate = 0
        
        config._ANALYSIS_CACHE[key] = {
            'data': result,
            'simulation_type': simulation_type,
            'operation': operation,
            'params': params,
            'data_type': type(result).__name__,
            'timestamp': time.time(),
            'size_bytes': size_estimate
        }
        Cache._stats['sets'] += 1
        
        # Auto-cleanup if cache gets too large
        if len(config._ANALYSIS_CACHE) > 1000:
            Cache._cleanup_old_entries()
    
    @staticmethod
    def _cleanup_old_entries(max_entries: int = 500):
        """Remove oldest cache entries to free memory"""
        if len(config._ANALYSIS_CACHE) <= max_entries:
            return
            
        # Sort by timestamp and remove oldest
        entries = [(k, v.get('timestamp', 0) if isinstance(v, dict) else 0) 
                  for k, v in config._ANALYSIS_CACHE.items()]
        entries.sort(key=lambda x: x[1])
        
        to_remove = len(entries) - max_entries
        for i in range(to_remove):
            key = entries[i][0]
            if key in config._ANALYSIS_CACHE:
                del config._ANALYSIS_CACHE[key]

    @staticmethod
    def clear_sample_cache():
        """Clear sample paths cache and update statistics"""
        cleared_count = len(config._SAMPLE_PATHS_CACHE)
        config._SAMPLE_PATHS_CACHE.clear()
        print(f"✓ Cleared {cleared_count} sample cache entries")

    @staticmethod
    def clear_analysis_cache():
        """Clear analysis cache and reset statistics"""
        cleared_count = len(config._ANALYSIS_CACHE)
        config._ANALYSIS_CACHE.clear()
        Cache._stats = {'hits': 0, 'misses': 0, 'sets': 0, 'memory_usage': 0}
        print(f"✓ Cleared {cleared_count} analysis cache entries")

    @staticmethod
    def get_cached_analysis(cache_key: str):
        """Get cached analysis with hit/miss tracking"""
        result = config._ANALYSIS_CACHE.get(cache_key)
        if result is None:
            Cache._stats['misses'] += 1
        else:
            Cache._stats['hits'] += 1
        return result

    @staticmethod
    def set_cached_analysis(cache_key: str, result):
        """Set cached analysis with statistics tracking"""
        config._ANALYSIS_CACHE[cache_key] = result
        Cache._stats['sets'] += 1
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = Cache._stats['hits'] + Cache._stats['misses']
        hit_rate = Cache._stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': f"{hit_rate:.2%}",
            'total_hits': Cache._stats['hits'],
            'total_misses': Cache._stats['misses'],
            'total_sets': Cache._stats['sets'],
            'cache_size': len(config._ANALYSIS_CACHE),
            'sample_cache_size': len(config._SAMPLE_PATHS_CACHE),
            'estimated_memory_mb': Cache._stats['memory_usage'] / (1024 * 1024)
        }
    
    @staticmethod
    def print_cache_stats():
        """Print formatted cache statistics"""
        stats = Cache.get_cache_stats()
        print("\n📊 Cache Performance Statistics:")
        print(f"   Hit Rate: {stats['hit_rate']}")
        print(f"   Cache Entries: {stats['cache_size']} analysis, {stats['sample_cache_size']} samples")
        print(f"   Memory Usage: ~{stats['estimated_memory_mb']:.1f} MB")
        print(f"   Operations: {stats['total_hits']} hits, {stats['total_misses']} misses, {stats['total_sets']} sets")
