"""
Cache Manager für Bismarck Factor Factory
"""

import sqlite3
import hashlib
import json
import time
import logging
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """
    SQLite-basierter Cache mit TTL
    
    Usage:
        cache = CacheManager(db_path="cache.db")
        key = cache.compute_key("text", "source", timestamp)
        
        # Get
        result = cache.get(key)
        
        # Put
        cache.put(key, {"score": 0.8}, ttl=86400)
    """
    
    def __init__(self, db_path: str = "bismarck_cache.db"):
        self.db_path = db_path
        self._init_db()
        logger.info(f"CacheManager initialized: {db_path}")
    
    def _init_db(self) -> None:
        """Initialisiere SQLite Datenbank"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
        """)
        
        conn.commit()
        conn.close()
    
    def compute_key(self, text: str, source: str, timestamp: float) -> str:
        """
        Berechne SHA256-basierte Cache-Key
        
        Args:
            text: Artikel-Text
            source: Quellen-Name
            timestamp: Zeitstempel
            
        Returns:
            SHA256 Hash als String
        """
        # Erste 5k Zeichen für Performance
        text_snippet = text[:5000]
        
        payload = f"{source}|{timestamp}|{text_snippet}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Hole Wert aus Cache
        
        Args:
            key: Cache-Key
            
        Returns:
            Cached value oder None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prüfe TTL
        now = time.time()
        cursor.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, now)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode cached value for key: {key}")
                return None
        
        return None
    
    def put(self, key: str, value: Any, ttl: int = 86400) -> None:
        """
        Speichere Wert im Cache
        
        Args:
            key: Cache-Key
            value: Zu speichernder Wert (muss JSON-serializable sein)
            ttl: Time-To-Live in Sekunden (default: 24h)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = time.time()
        expires_at = now + ttl
        
        try:
            value_json = json.dumps(value)
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (key, value_json, now, expires_at))
            
            conn.commit()
            logger.debug(f"Cached value for key: {key}, expires in {ttl}s")
            
        except json.JSONEncodeError as e:
            logger.error(f"Failed to encode value for caching: {e}")
        finally:
            conn.close()
    
    def cleanup_expired(self) -> int:
        """
        Bereinige abgelaufene Cache-Einträge
        
        Returns:
            Anzahl gelöschter Einträge
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = time.time()
        cursor.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired cache entries")
        
        return deleted
    
    def clear(self) -> None:
        """Lösche alle Cache-Einträge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Cache-Statistiken
        
        Returns:
            Dict mit Cache-Stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = time.time()
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]
        
        # Expired entries
        cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at <= ?", (now,))
        expired = cursor.fetchone()[0]
        
        # Valid entries
        valid = total - expired
        
        # Oldest entry
        cursor.execute("SELECT MIN(created_at) FROM cache")
        oldest = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "oldest_entry_ts": oldest
        }


if __name__ == "__main__":
    # Quick test
    cache = CacheManager("test_cache.db")
    
    key = cache.compute_key("Test article text", "test_source", 12345.0)
    
    print(f"Key: {key}")
    
    # Put
    cache.put(key, {"score": 0.8, "meta": "test"})
    
    # Get
    result = cache.get(key)
    print(f"Result: {result}")
    
    # Stats
    stats = cache.stats()
    print(f"Stats: {stats}")
