"""
Card Detector AI - Final Version
Gemini API + Cache + Multi-key rotation (unlimited keys)
"""

import os
import re
import io
import time
import logging
import yaml
import hashlib
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from PIL import Image

# Tắt warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Kết quả nhận diện"""
    cards: List[str]
    success: bool
    provider: str
    time: float
    model_used: str = ""
    error: Optional[str] = None
    from_cache: bool = False
    key_used: int = 0  # Key nào đã dùng
    
    @property
    def card_string(self) -> str:
        return ' '.join(self.cards)
    
    @property
    def is_valid(self) -> bool:
        return len(self.cards) == 13 and len(set(self.cards)) == 13


class CardDetector:
    """
    Card Detector - Gemini with unlimited keys support
    """
    
    PROMPT = """Identify exactly 13 playing cards in this image.
Output ONLY the cards in format: RANK+SUIT separated by spaces.
Ranks: A K Q J 10 9 8 7 6 5 4 3 2
Suits: S(spades) H(hearts) D(diamonds) C(clubs)
Example output: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S
Your output (13 cards only):"""

    def __init__(self, config_path: str = "config/ai_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Multi-key support (unlimited)
        self.gemini_keys = self._load_all_keys()
        self.current_key_index = 0
        self.gemini_key = self.gemini_keys[0] if self.gemini_keys else None
        
        # Cache setup
        self.cache_enabled = self.config.get('cache', {}).get('enabled', True)
        self.cache_dir = Path(self.config.get('cache', {}).get('dir', 'data/cache/detections'))
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Log status
        if self.gemini_keys:
            logger.info(f"✅ Loaded {len(self.gemini_keys)} Gemini API key(s)")
        else:
            logger.warning("⚠️  No Gemini API keys found!")
        
        if self.cache_enabled:
            logger.info(f"✅ Cache enabled: {self.cache_dir}")
    
    def _load_config(self, path: str) -> dict:
        """Load config từ YAML"""
        try:
            config_file = Path(path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Config load error: {e}")
        return {}
    
    def _load_all_keys(self) -> List[str]:
        """Load tất cả API keys từ config và environment"""
        keys = []
        
        # ✅ Từ list api_keys trong config (cách mới)
        api_keys_list = self.config.get('gemini', {}).get('api_keys', [])
        if isinstance(api_keys_list, list):
            for key in api_keys_list:
                if self._is_valid_key(key) and key not in keys:
                    keys.append(key)
        
        # ✅ Từ api_key đơn lẻ (backward compatible)
        main_key = self.config.get('gemini', {}).get('api_key', '')
        if self._is_valid_key(main_key) and main_key not in keys:
            keys.append(main_key)
        
        backup_key = self.config.get('gemini', {}).get('api_key_backup', '')
        if self._is_valid_key(backup_key) and backup_key not in keys:
            keys.append(backup_key)
        
        # ✅ Từ environment variables (hỗ trợ đến 10 keys)
        env_vars = ['GEMINI_API_KEY', 'GOOGLE_API_KEY'] + \
                   [f'GEMINI_API_KEY_{i}' for i in range(2, 11)]
        
        for env_var in env_vars:
            env_key = os.environ.get(env_var, '')
            if self._is_valid_key(env_key) and env_key not in keys:
                keys.append(env_key)
        
        return keys
    
    def _is_valid_key(self, key: str) -> bool:
        """Check nếu key hợp lệ"""
        if not key or len(key) < 20:
            return False
        
        invalid_prefixes = ['YOUR', 'PASTE', 'AIza...', 'REPLACE', 'xxx', 'XXX']
        return not any(key.startswith(p) for p in invalid_prefixes)
    
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Nhận diện 13 lá bài
        Flow: Check cache → Call Gemini (with rotation) → Save cache
        """
        start = time.time()
        
        # Check cache
        if self.cache_enabled:
            cached = self._load_from_cache(image)
            if cached:
                logger.info("💾 Cache HIT - không gọi API")
                cached.time = time.time() - start
                cached.from_cache = True
                return cached
        
        # Call Gemini
        logger.info("🌐 Cache MISS - calling Gemini API...")
        result = self._detect_gemini(image)
        result.time = time.time() - start
        
        # Save to cache
        if self.cache_enabled and result.success:
            self._save_to_cache(image, result)
        
        return result
    
    def _detect_gemini(self, image: Image.Image) -> DetectionResult:
        """Nhận diện với auto-rotation qua tất cả keys"""
        
        if not self.gemini_keys:
            return DetectionResult(
                cards=[], success=False, provider="gemini", time=0,
                error="Chưa có API key! Thêm vào config/ai_config.yaml"
            )
        
        last_error = None
        
        # Thử TẤT CẢ keys
        for attempt in range(len(self.gemini_keys)):
            try:
                api_key = self.gemini_keys[self.current_key_index]
                key_num = self.current_key_index + 1
                
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                img = self._resize_image(image)
                model_name = self._find_best_model(genai)
                
                logger.info(f"🔄 Key {key_num}/{len(self.gemini_keys)} → {model_name}")
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([self.PROMPT, img])
                
                cards = self._parse_response(response.text)
                
                logger.info(f"✅ Detected {len(cards)} cards")
                
                return DetectionResult(
                    cards=cards,
                    success=True,
                    provider="gemini",
                    time=0,
                    model_used=model_name,
                    key_used=key_num,
                    error=None if len(cards) == 13 else f"Phát hiện {len(cards)}/13 lá"
                )
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                is_rate_limit = any(kw in error_msg.lower() for kw in 
                                   ['429', 'quota', 'rate limit', 'resource exhausted'])
                
                if is_rate_limit and attempt < len(self.gemini_keys) - 1:
                    logger.warning(f"⚠️  Key {key_num} rate limited → switching...")
                    self.current_key_index = (self.current_key_index + 1) % len(self.gemini_keys)
                    time.sleep(0.5)
                    continue
                else:
                    if is_rate_limit:
                        logger.error(f"❌ All {len(self.gemini_keys)} keys rate limited!")
                        return DetectionResult(
                            cards=[], success=False, provider="gemini", time=0,
                            error=f"Tất cả {len(self.gemini_keys)} API key đều bị rate limit!"
                        )
                    else:
                        logger.error(f"❌ Gemini error: {error_msg}")
                        return DetectionResult(
                            cards=[], success=False, provider="gemini", time=0,
                            error=error_msg
                        )
        
        return DetectionResult(
            cards=[], success=False, provider="gemini", time=0,
            error=f"All keys failed: {last_error}"
        )
    
    def _find_best_model(self, genai) -> str:
        """Tìm model tốt nhất"""
        preferred = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        
        try:
            available = [m.name.replace("models/", "") 
                        for m in genai.list_models() 
                        if 'generateContent' in m.supported_generation_methods]
            
            for p in preferred:
                if p in available:
                    return p
            
            return available[0] if available else "gemini-1.5-flash"
        except:
            return "gemini-1.5-flash"
    
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize ảnh"""
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _parse_response(self, text: str) -> List[str]:
        """Parse AI response"""
        text = text.strip().upper()
        pattern = r'\b(10|[2-9]|[JQKA])([SHDC])\b'
        matches = re.findall(pattern, text)
        
        cards = []
        for rank, suit in matches:
            card = f"{rank}{suit}"
            if card not in cards:
                cards.append(card)
        
        return cards
    
    # ============ CACHE ============
    
    def _get_image_hash(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    
    def _load_from_cache(self, image: Image.Image) -> Optional[DetectionResult]:
        try:
            img_hash = self._get_image_hash(image)
            cache_file = self.cache_dir / f"{img_hash}.json"
            
            if not cache_file.exists():
                return None
            
            max_age = self.config.get('cache', {}).get('max_age_hours', 168)
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            if datetime.now() - file_time > timedelta(hours=max_age):
                cache_file.unlink()
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return DetectionResult(
                cards=data['cards'],
                success=data['success'],
                provider=data['provider'],
                time=0,
                model_used=data.get('model_used', ''),
                error=data.get('error'),
                from_cache=True
            )
        except:
            return None
    
    def _save_to_cache(self, image: Image.Image, result: DetectionResult):
        try:
            img_hash = self._get_image_hash(image)
            cache_file = self.cache_dir / f"{img_hash}.json"
            
            data = {
                'cards': result.cards,
                'success': result.success,
                'provider': result.provider,
                'model_used': result.model_used,
                'error': result.error,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"💾 Cached: {img_hash[:8]}...")
        except Exception as e:
            logger.debug(f"Cache save error: {e}")
    
    def clear_cache(self) -> int:
        """Xóa cache"""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"🗑️ Cleared {count} cache files")
        return count
    
    # ============ UTILITY ============
    
    def detect_from_file(self, path: str) -> DetectionResult:
        return self.detect(Image.open(path))
    
    def get_status(self) -> dict:
        cache_files = list(self.cache_dir.glob("*.json")) if self.cache_enabled else []
        return {
            'total_keys': len(self.gemini_keys),
            'current_key': self.current_key_index + 1,
            'cache_enabled': self.cache_enabled,
            'cache_files': len(cache_files)
        }
    
    def add_key(self, api_key: str) -> bool:
        """Thêm key mới runtime"""
        if self._is_valid_key(api_key) and api_key not in self.gemini_keys:
            self.gemini_keys.append(api_key)
            logger.info(f"✅ Added new key. Total: {len(self.gemini_keys)}")
            return True
        return False


# ============ TEST ============

if __name__ == "__main__":
    import sys
    
    print("🃏 Card Detector - Multi-Key Version")
    print("=" * 50)
    
    d = CardDetector()
    
    status = d.get_status()
    print(f"\n📊 Status:")
    print(f"  Total keys: {status['total_keys']}")
    print(f"  Cache: {'ON' if status['cache_enabled'] else 'OFF'} ({status['cache_files']} files)")
    
    if len(sys.argv) > 1:
        print(f"\n📸 Testing: {sys.argv[1]}")
        
        r = d.detect_from_file(sys.argv[1])
        
        print(f"\n{'='*30}")
        print(f"Provider: {r.provider} ({r.model_used})")
        print(f"Key used: {r.key_used}/{status['total_keys']}")
        print(f"From cache: {r.from_cache}")
        print(f"Time: {r.time:.2f}s")
        print(f"Cards ({len(r.cards)}): {r.card_string}")
        print(f"Valid: {'✅' if r.is_valid else '❌'}")
        
        if r.error:
            print(f"Error: {r.error}")
    else:
        print(f"\n💡 Usage: python card_detector_ai.py <image.jpg>")