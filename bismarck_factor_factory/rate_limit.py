"""
Rate Limiting & Token Bucket
============================

Implementiert TokenBucket mit verschiedenen Strategien:
- Fixed Window
- Token Bucket (leaky bucket)
- Exponential Backoff mit Jitter
"""

import time
import asyncio
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Strategien für Rate Limiting"""
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """Konfiguration für Rate Limiting"""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    rate_per_minute: int = 60
    capacity: Optional[int] = None  # None = unbegrenzt
    burst_size: int = 10  # Erlaubte Burst-Größe
    
    # Exponential Backoff
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


class TokenBucket:
    """
    TokenBucket für Rate Limiting
    
    Implementiert einen leaky bucket Algorithmus:
    - Tokens werden kontinuierlich mit rate/min regeneriert
    - Jede Anfrage konsumiert tokens
    - Bei leerem Bucket wird blockiert
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._rate = config.rate_per_minute / 60.0  # tokens per second
        self._capacity = config.capacity or config.rate_per_minute
        self._tokens = float(self._capacity)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
        
        logger.info(f"TokenBucket initialized: rate={self._rate:.2f} tokens/s, capacity={self._capacity}")
    
    @property
    def tokens_available(self) -> float:
        """Aktuell verfügbare Tokens"""
        self._refill()
        return self._tokens
    
    def _refill(self) -> None:
        """Fülle Tokens nach verstrichener Zeit"""
        now = time.time()
        elapsed = now - self._last_update
        
        # Regeneriere Tokens basierend auf Rate
        self._tokens = min(
            self._capacity,
            self._tokens + (elapsed * self._rate)
        )
        self._last_update = now
    
    async def consume(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Konsumiere Tokens
        
        Args:
            tokens: Anzahl zu konsumierender Tokens
            wait: Wenn True, warte bis Tokens verfügbar sind
            
        Returns:
            True wenn konsumiert, False wenn abgelehnt
        """
        async with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                logger.debug(f"Consumed {tokens} tokens, {self._tokens:.2f} remaining")
                return True
            
            if not wait:
                logger.warning(f"Rate limit hit: need {tokens}, have {self._tokens:.2f}")
                return False
            
            # Warte bis genug Tokens verfügbar sind
            wait_time = (tokens - self._tokens) / self._rate
            logger.info(f"Waiting {wait_time:.2f}s for {tokens} tokens")
            await asyncio.sleep(wait_time)
            
            # Retry nach Wartezeit
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            return False
    
    def reset(self) -> None:
        """Reset bucket auf capacity"""
        self._tokens = self._capacity
        self._last_update = time.time()


class AsyncRateLimiter:
    """
    Async Rate Limiter mit Priority Queue
    
    Führt async Funktionen mit Rate Limiting aus:
    - Priority Queue (high/medium/low)
    - Exponential Backoff mit Jitter
    - Automatic retry bei 429 errors
    """
    
    def __init__(self, config: RateLimitConfig):
        self.bucket = TokenBucket(config)
        self.config = config
        self.queue = asyncio.PriorityQueue()
        self._queue_processor = None
        self._running = False
        
    async def __aenter__(self):
        """Context Manager Entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        await self.stop()
    
    async def start(self) -> None:
        """Starte Queue Processor"""
        if self._running:
            return
        
        self._running = True
        self._queue_processor = asyncio.create_task(self._process_queue())
        logger.info("AsyncRateLimiter started")
    
    async def stop(self) -> None:
        """Stoppe Queue Processor"""
        self._running = False
        if self._queue_processor:
            await self._queue_processor
        logger.info("AsyncRateLimiter stopped")
    
    async def _process_queue(self) -> None:
        """Prozessiere Queue Items"""
        while self._running:
            try:
                # Warte auf nächstes Item (timeout 1s)
                try:
                    priority, item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Konsumiere Token
                await self.bucket.consume()
                
                # Execute function
                func, args, kwargs, future = item
                try:
                    result = await func(*args, **kwargs)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing queue: {e}", exc_info=True)
    
    async def execute(
        self,
        func: Callable,
        *args,
        priority: int = 0,  # 0=high, 1=medium, 2=low
        **kwargs
    ) -> Any:
        """
        Führe Funktion mit Rate Limiting aus
        
        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments
            priority: Priority (0=high, 1=medium, 2=low)
            
        Returns:
            Function result
        """
        future = asyncio.Future()
        await self.queue.put((priority, (func, args, kwargs, future)))
        return await future
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Führe Funktion mit Retry-Logic aus
        
        Implementiert Exponential Backoff mit Jitter
        """
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                return await self.execute(func, *args, **kwargs)
                
            except Exception as e:
                # Prüfe ob es ein Rate Limit Error ist
                is_rate_limit = (
                    "429" in str(e) or 
                    "rate limit" in str(e).lower() or
                    "too many requests" in str(e).lower()
                )
                
                if not is_rate_limit or attempt == max_retries - 1:
                    raise
                
                # Exponential Backoff mit Jitter
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay
                )
                
                if self.config.jitter:
                    import random
                    delay *= random.uniform(0.5, 1.5)
                
                logger.warning(f"Rate limit error (attempt {attempt+1}/{max_retries}), retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise RuntimeError(f"Failed after {max_retries} retries")


# Utility Functions
async def rate_limited_call(
    func: Callable,
    *args,
    rate_per_minute: int = 60,
    **kwargs
) -> Any:
    """
    Convenience Wrapper für Rate-Limited Calls
    
    Usage:
        result = await rate_limited_call(my_async_function, arg1, arg2)
    """
    config = RateLimitConfig(rate_per_minute=rate_per_minute)
    async with AsyncRateLimiter(config) as limiter:
        return await limiter.execute(func, *args, priority=1, **kwargs)
