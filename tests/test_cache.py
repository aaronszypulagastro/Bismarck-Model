"""
Unit Tests für Cache Manager
"""

import pytest
import time
import os
from bismarck_factor_factory.cache import CacheManager


class TestCacheManager:
    """Tests für CacheManager"""
    
    def test_compute_key(self):
        """Test Key-Generierung"""
        cache = CacheManager("test_cache_1.db")
        key1 = cache.compute_key("test text", "source1", 12345.0)
        key2 = cache.compute_key("test text", "source1", 12345.0)
        key3 = cache.compute_key("different text", "source1", 12345.0)
        
        assert key1 == key2  # Gleiche Inputs = gleicher Key
        assert key1 != key3  # Unterschiedliche Texte = unterschiedliche Keys
    
    def test_put_get(self):
        """Test Put/Get"""
        cache = CacheManager("test_cache_2.db")
        key = cache.compute_key("test article", "newsapi", time.time())
        
        # Put
        cache.put(key, {"score": 0.8}, ttl=60)
        
        # Get
        result = cache.get(key)
        assert result is not None
        assert result["score"] == 0.8
        
        # Cleanup
        os.remove("test_cache_2.db")
    
    def test_ttl_expiration(self):
        """Test TTL Expiration"""
        cache = CacheManager("test_cache_3.db")
        key = cache.compute_key("test", "source", time.time())
        
        # Put with short TTL
        cache.put(key, {"value": "test"}, ttl=1)
        
        # Immediate get should work
        result = cache.get(key)
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be None now
        result = cache.get(key)
        assert result is None
        
        # Cleanup
        os.remove("test_cache_3.db")
    
    def test_stats(self):
        """Test Cache Statistics"""
        cache = CacheManager("test_cache_4.db")
        
        # Add some entries
        for i in range(3):
            key = cache.compute_key(f"text {i}", "source", time.time())
            cache.put(key, {"id": i}, ttl=60)
        
        stats = cache.stats()
        assert stats["total_entries"] == 3
        assert stats["valid_entries"] == 3
        
        # Cleanup
        os.remove("test_cache_4.db")
    
    def test_cleanup_expired(self):
        """Test Cleanup Function"""
        cache = CacheManager("test_cache_5.db")
        
        # Add expired entry
        key = cache.compute_key("expired", "source", time.time() - 100)
        cache.put(key, {"value": "expired"}, ttl=1)
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Cleanup
        deleted = cache.cleanup_expired()
        assert deleted >= 1
        
        # Cleanup
        os.remove("test_cache_5.db")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
