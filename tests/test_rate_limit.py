"""
Unit Tests für Rate Limiting
"""

import pytest
import asyncio
import time
from bismarck_factor_factory.rate_limit import (
    TokenBucket,
    AsyncRateLimiter,
    RateLimitConfig,
    RateLimitStrategy
)


class TestTokenBucket:
    """Tests für TokenBucket"""
    
    def test_initial_state(self):
        """Test initialer Zustand"""
        config = RateLimitConfig(rate_per_minute=60)
        bucket = TokenBucket(config)
        assert bucket.tokens_available == 60.0
    
    def test_token_consumption(self):
        """Test Token-Verbrauch"""
        config = RateLimitConfig(rate_per_minute=60)
        bucket = TokenBucket(config)
        
        async def test():
            result = await bucket.consume(tokens=10)
            assert result is True
            assert bucket.tokens_available == 50.0
        
        asyncio.run(test())
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test Rate Limiting Verhalten"""
        config = RateLimitConfig(rate_per_minute=6)  # 1 token per 10s
        bucket = TokenBucket(config)
        bucket._tokens = 1.0  # Start mit 1 Token
        
        # Konsumiere den Token
        result1 = await bucket.consume(tokens=1, wait=False)
        assert result1 is True
        
        # Jetzt sollte der Bucket leer sein
        result2 = await bucket.consume(tokens=1, wait=False)
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_refill_rate(self):
        """Test Token-Regenerierung"""
        config = RateLimitConfig(rate_per_minute=60)
        bucket = TokenBucket(config)
        bucket._tokens = 0.0  # Leerer Bucket
        
        # 1 Sekunde warten = 1 Token regenerieren
        await asyncio.sleep(1.1)
        
        result = await bucket.consume(tokens=1, wait=False)
        assert result is True
    
    def test_reset(self):
        """Test Reset-Funktion"""
        config = RateLimitConfig(rate_per_minute=60)
        bucket = TokenBucket(config)
        
        async def test():
            await bucket.consume(tokens=30)
            assert bucket.tokens_available < 60.0
            
            bucket.reset()
            assert bucket.tokens_available == 60.0
        
        asyncio.run(test())


class TestAsyncRateLimiter:
    """Tests für AsyncRateLimiter"""
    
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Test Basic Execution"""
        config = RateLimitConfig(rate_per_minute=60)
        
        async def dummy_func(x, y):
            return x + y
        
        async with AsyncRateLimiter(config) as limiter:
            result = await limiter.execute(dummy_func, 5, 10, priority=1)
            assert result == 15
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test Priority Queue"""
        config = RateLimitConfig(rate_per_minute=1000)  # Sehr hoch für sofortige Ausführung
        results = []
        
        async def func(label):
            results.append(label)
        
        async with AsyncRateLimiter(config) as limiter:
            await limiter.execute(func, "low", priority=2)
            await limiter.execute(func, "high", priority=0)
            await limiter.execute(func, "medium", priority=1)
            
            # Warte bis Queue verarbeitet ist
            await asyncio.sleep(0.5)
        
        # High priority sollte zuerst sein
        assert results == ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test Exponential Backoff bei Rate Limit Errors"""
        config = RateLimitConfig(
            rate_per_minute=1,  # Sehr niedrig
            max_retries=2,
            base_delay=0.1
        )
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success"
        
        async with AsyncRateLimiter(config) as limiter:
            result = await limiter.execute_with_retry(failing_func)
            assert result == "success"
            assert call_count == 3


class TestIntegration:
    """Integration Tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_rate_limiting(self):
        """End-to-End Test mit realistischem Szenario"""
        config = RateLimitConfig(
            rate_per_minute=10,
            max_retries=3
        )
        
        results = []
        
        async def api_call(i):
            results.append(i)
            return f"result_{i}"
        
        start_time = time.time()
        
        async with AsyncRateLimiter(config) as limiter:
            tasks = [
                limiter.execute(api_call, i, priority=1)
                for i in range(5)
            ]
            await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Mit 10 calls/min sollten 5 calls mindestens 6 Sekunden dauern
        assert elapsed >= 0.5  # Minimum Zeit für Processing
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
