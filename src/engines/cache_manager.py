"""
Cache Manager - Quản lý cache để tăng tốc độ tính toán
"""
import pickle
import hashlib
import os
from pathlib import Path
from typing import Any, Optional
from functools import wraps
import time


class CacheManager:
    """Quản lý cache cho các tính toán nặng"""
    
    def __init__(self, cache_dir: str = "../../data/cache"):
        """
        Args:
            cache_dir: Thư mục lưu cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._memory_cache = {}
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Tạo cache key từ arguments"""
        # Convert arguments thành string
        key_str = str(args) + str(sorted(kwargs.items()))
        
        # Hash để tạo key ngắn gọn
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Lấy giá trị từ cache
        
        Thử memory cache trước, sau đó disk cache
        """
        # Try memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Try disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                    # Load vào memory cache
                    self._memory_cache[key] = value
                    return value
            except Exception as e:
                print(f"Warning: Failed to load cache {key}: {e}")
                return None
        
        return None
    
    def set(self, key: str, value: Any, memory_only: bool = False):
        """
        Lưu giá trị vào cache
        
        Args:
            key: Cache key
            value: Giá trị cần cache
            memory_only: Chỉ cache trong memory (không lưu disk)
        """
        # Save to memory
        self._memory_cache[key] = value
        
        # Save to disk
        if not memory_only:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                print(f"Warning: Failed to save cache {key}: {e}")
    
    def clear_memory(self):
        """Xóa memory cache"""
        self._memory_cache.clear()
    
    def clear_disk(self):
        """Xóa disk cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def clear_all(self):
        """Xóa tất cả cache"""
        self.clear_memory()
        self.clear_disk()
    
    def cached(self, ttl: Optional[int] = None, memory_only: bool = True):
        """
        Decorator để cache kết quả function
        
        Args:
            ttl: Time to live (seconds) - chưa implement
            memory_only: Chỉ cache trong memory
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Tạo cache key
                cache_key = self.get_cache_key(func.__name__, args, kwargs)
                
                # Try get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Compute
                result = func(*args, **kwargs)
                
                # Save to cache
                self.set(cache_key, result, memory_only=memory_only)
                
                return result
            
            return wrapper
        return decorator


# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# ==================== TESTS ====================

def test_cache_manager():
    """Test CacheManager"""
    print("Testing CacheManager...")
    
    cache = CacheManager(cache_dir="../../data/cache/test")
    
    # Test set/get
    cache.set("test_key", {"value": 123})
    result = cache.get("test_key")
    assert result == {"value": 123}
    
    # Test cache miss
    result = cache.get("nonexistent")
    assert result is None
    
    # Test decorator
    call_count = 0
    
    @cache.cached(memory_only=True)
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)  # Simulate expensive computation
        return x + y
    
    # First call - should compute
    result1 = expensive_function(2, 3)
    assert result1 == 5
    assert call_count == 1
    
    # Second call - should use cache
    result2 = expensive_function(2, 3)
    assert result2 == 5
    assert call_count == 1  # Không tăng vì dùng cache
    
    # Different args - should compute again
    result3 = expensive_function(3, 4)
    assert result3 == 7
    assert call_count == 2
    
    # Clean up
    cache.clear_all()
    
    print("✅ CacheManager tests passed!")


if __name__ == "__main__":
    test_cache_manager()
    print("\n✅ All cache_manager.py tests passed!")