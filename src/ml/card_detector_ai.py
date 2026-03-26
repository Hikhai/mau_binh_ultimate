"""
Card Detector AI - Gemini 2.5 Flash
Version: 4.0 - Theo hướng dẫn chính thức Google
Đơn giản + Hiệu quả + Đúng chuẩn SDK mới
"""

import os
import time
import logging
import yaml
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance, ImageFilter

# Import SDK google-genai (chuẩn mới nhất)
GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠️ Cài: pip install -U google-genai")

import warnings
warnings.filterwarnings("ignore")

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
    key_used: int = 0
    raw_response: str = ""
    
    @property
    def card_string(self) -> str:
        return ' '.join(self.cards)
    
    @property
    def is_valid(self) -> bool:
        return len(self.cards) == 13 and len(set(self.cards)) == 13


class CardDetector:
    """
    Card Detector - Gemini 2.5 Flash
    Theo hướng dẫn chính thức Google
    """
    
    # Prompt tối ưu cho nhận diện bài
    PROMPT = """You are an expert playing card recognition AI with exceptional computer vision capabilities.

🎯 TASK: Identify ALL 13 playing cards in this image with 100% accuracy.

📋 SYSTEMATIC ANALYSIS PROCESS:
1. **Initial Scan**: Survey the entire image to locate all card positions
2. **Corner Detection**: Focus on card corners where rank + suit are most visible
3. **Handle Challenges**:
   - Overlapping cards: Use visible portions and position context
   - Tilted/rotated cards: Recognize symbols at any angle
   - Partially obscured cards: Infer from visible suit color and partial symbols
   - Blurry/low quality: Use color (red/black) and shape patterns
4. **Validation**: Cross-check to ensure exactly 13 unique cards
5. **Final Review**: Verify no duplicates, all cards match standard deck

🃏 CARD FORMAT SPECIFICATION:
**Ranks** (in order): A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2
**Suits**:
  - S = ♠ Spades (BLACK)
  - H = ♥ Hearts (RED)
  - D = ♦ Diamonds (RED)
  - C = ♣ Clubs (BLACK)

🔍 RECOGNITION RULES:
✓ MUST identify exactly 13 cards
✓ Each card must be unique (no duplicates)
✓ Use color as primary suit indicator:
  - RED cards → H (Hearts) or D (Diamonds)
  - BLACK cards → S (Spades) or C (Clubs)
✓ For ambiguous cards:
  - Check surrounding cards for context
  - Use partial symbols visible
  - Make educated guess based on probability
✓ Format: Each card as RANK+SUIT (e.g., AS, KH, 10D)
✓ Separate cards with single space

📤 OUTPUT FORMAT:
Return ONLY the 13 cards in a single line, space-separated.
Do NOT include explanations, descriptions, or extra text.

EXAMPLE OUTPUT:
AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S

🚀 YOUR OUTPUT (13 cards only):"""

    def __init__(self, config_path: str = None):
        """Khởi tạo CardDetector"""
        self.api_keys: List[str] = []
        self.current_key_index = 0
        self.cache_dir = Path("data/cache/cards")
        self.cache_enabled = True
        self.stats = {'success': 0, 'failed': 0, 'cache_hits': 0}
        
        # Load config
        self._load_config(config_path)
        
        # Create cache dir
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str = None):
        """Load API keys từ config hoặc environment"""
        
        # Thử load từ file config trước
        paths_to_try = [
            config_path,
            "config/ai_config.yaml",
            "../config/ai_config.yaml",
            "../../config/ai_config.yaml",
            Path(__file__).parent.parent.parent / "config" / "ai_config.yaml",
        ]
        
        for path in paths_to_try:
            if path and Path(path).exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    if config and 'gemini' in config:
                        keys = config['gemini'].get('api_keys', [])
                        if isinstance(keys, list):
                            self.api_keys = [k for k in keys if k and k.strip()]
                        elif isinstance(keys, str):
                            self.api_keys = [keys]
                        
                        if self.api_keys:
                            logger.info(f"✅ Loaded {len(self.api_keys)} keys from {path}")
                            return
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        # Nếu không có config → thử environment variable
        env_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if env_key:
            self.api_keys = [env_key]
            logger.info("✅ Loaded key from environment")
            return
        
        logger.warning("⚠️ No API keys found!")
    
    def get_status(self) -> Dict:
        """Lấy status của detector"""
        return {
            'total_keys': len(self.api_keys),
            'current_key': self.current_key_index + 1 if self.api_keys else 0,
            'cache_enabled': self.cache_enabled,
            'cache_files': len(list(self.cache_dir.glob('*.json'))) if self.cache_dir.exists() else 0,
            'config_found': len(self.api_keys) > 0,
            'genai_available': GENAI_AVAILABLE,
            'stats': self.stats,
        }
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Hash ảnh để cache"""
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return hashlib.md5(img_bytes.getvalue()).hexdigest()[:16]
    
    def _get_cache(self, image_hash: str) -> Optional[DetectionResult]:
        """Lấy kết quả từ cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{image_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check expiry (24h)
                cached_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                if datetime.now() - cached_time < timedelta(hours=24):
                    self.stats['cache_hits'] += 1
                    return DetectionResult(
                        cards=data['cards'],
                        success=True,
                        provider='cache',
                        time=0,
                        from_cache=True,
                    )
            except:
                pass
        return None
    
    def _save_cache(self, image_hash: str, result: DetectionResult):
        """Lưu kết quả vào cache"""
        if not self.cache_enabled or not result.success:
            return
        
        cache_file = self.cache_dir / f"{image_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'cards': result.cards,
                    'provider': result.provider,
                    'model': result.model_used,
                    'timestamp': datetime.now().isoformat(),
                }, f)
        except:
            pass
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Tiền xử lý ảnh để nhận diện tốt hơn"""
        
        # Convert RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize smart
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Upscale nếu quá nhỏ
        min_size = 800
        if max(image.size) < min_size:
            ratio = min_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
        return image
    
    def _parse_cards(self, text: str) -> List[str]:
        """Parse cards từ text response"""
        import re
        
        cards = []
        text = text.upper().strip()
        
        # Pattern: AS, KH, 10D, etc.
        pattern = r'\b(10|[2-9AKQJ])([SHDC])\b'
        matches = re.findall(pattern, text)
        
        for rank, suit in matches:
            card = f"{rank}{suit}"
            if card not in cards:
                cards.append(card)
            
            if len(cards) >= 13:
                break
        
        return cards
    
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Nhận diện cards từ ảnh PIL Image
        
        Args:
            image: PIL Image object
        
        Returns:
            DetectionResult
        """
        start_time = time.time()
        
        # Check cache
        image_hash = self._get_image_hash(image)
        cached = self._get_cache(image_hash)
        if cached:
            logger.info(f"📦 Cache hit: {cached.card_string}")
            return cached
        
        # Run detection
        result = self._detect_gemini(image)
        result.time = time.time() - start_time
        
        # Update stats
        if result.success:
            self.stats['success'] += 1
            self._save_cache(image_hash, result)
        else:
            self.stats['failed'] += 1
        
        return result
    
    def _detect_gemini(self, image: Image.Image) -> DetectionResult:
        """
        Nhận diện với Gemini 2.5 Flash
        THEO ĐÚNG HƯỚNG DẪN CHÍNH THỨC
        """
        
        if not GENAI_AVAILABLE:
            return DetectionResult(
                cards=[], success=False, provider="gemini", time=0,
                error="❌ Chưa cài google-genai! Chạy: pip install -U google-genai"
            )
        
        if not self.api_keys:
            return DetectionResult(
                cards=[], success=False, provider="gemini", time=0,
                error="❌ Không có API key! Thêm vào config/ai_config.yaml hoặc environment"
            )
        
        last_error = None
        
        # Thử từng key
        for attempt in range(len(self.api_keys)):
            try:
                api_key = self.api_keys[self.current_key_index]
                key_num = self.current_key_index + 1
                
                # ============================================================
                # THEO HƯỚNG DẪN: Đặt API key vào environment
                # ============================================================
                os.environ["GEMINI_API_KEY"] = api_key
                
                # ============================================================
                # THEO HƯỚNG DẪN: Tạo client (tự động lấy key từ env)
                # ============================================================
                client = genai.Client()
                
                # Preprocess image
                img = self._preprocess_image(image)
                
                model_name = "gemini-2.5-flash"
                
                logger.info(f"🔄 Key {key_num}/{len(self.api_keys)} → {model_name}")
                
                # ============================================================
                # THEO HƯỚNG DẪN: Gọi generate_content với [text, image]
                # ============================================================
                response = client.models.generate_content(
                    model=model_name,
                    contents=[self.PROMPT, img]  # Đúng như hướng dẫn!
                )
                
                if not response or not response.text:
                    raise Exception("Empty response")
                
                raw_text = response.text.strip()
                logger.info(f"📝 Raw: {raw_text[:200]}")
                
                # Parse cards
                cards = self._parse_cards(raw_text)
                
                logger.info(f"🎯 Parsed: {len(cards)} cards")
                
                return DetectionResult(
                    cards=cards,
                    success=len(cards) > 0,
                    provider="gemini",
                    time=0,
                    model_used=model_name,
                    key_used=key_num,
                    error=None if len(cards) == 13 else f"Detected {len(cards)}/13 cards",
                    raw_response=raw_text[:1000],
                )
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                # Check rate limit / quota
                is_rate_limit = any(kw in error_msg.lower() for kw in 
                                   ['429', 'quota', 'rate limit', 'resource'])
                is_forbidden = '403' in error_msg or 'PERMISSION_DENIED' in error_msg
                
                if is_forbidden or is_rate_limit:
                    logger.warning(f"⚠️ Key {key_num}: {error_msg[:100]}")
                    if attempt < len(self.api_keys) - 1:
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                        time.sleep(1)
                        continue
                
                logger.warning(f"⚠️ Key {key_num} error: {error_msg[:150]}")
                if attempt < len(self.api_keys) - 1:
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    time.sleep(0.5)
                    continue
        
        return DetectionResult(
            cards=[], success=False, provider="gemini", time=0,
            error=f"All keys failed: {last_error}"
        )
    
    def detect_from_file(self, file_path: str) -> DetectionResult:
        """Nhận diện từ file ảnh"""
        try:
            image = Image.open(file_path)
            return self.detect(image)
        except Exception as e:
            return DetectionResult(
                cards=[], success=False, provider="file", time=0,
                error=f"Lỗi đọc file: {e}"
            )
    
    def print_stats(self):
        """In thống kê"""
        print(f"\n📊 Detection Stats:")
        print(f"  Success:    {self.stats['success']}")
        print(f"  Failed:     {self.stats['failed']}")
        print(f"  Cache hits: {self.stats['cache_hits']}")


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import sys
    
    print("🃏 Card Detector - Gemini 2.5 Flash (Chính thức)")
    print("="*60)
    
    d = CardDetector()
    status = d.get_status()
    
    print(f"\n📊 Status:")
    print(f"  Keys:       {status['total_keys']}")
    print(f"  Current:    {status['current_key']}")
    print(f"  GenAI OK:   {status['genai_available']}")
    
    if len(sys.argv) > 1:
        print(f"\n📸 Testing: {sys.argv[1]}")
        
        r = d.detect_from_file(sys.argv[1])
        
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"{'='*60}")
        print(f"  Success:  {'✅' if r.success else '❌'}")
        print(f"  Cards:    {len(r.cards)}/13")
        print(f"  Result:   {r.card_string}")
        print(f"  Time:     {r.time:.2f}s")
        
        if r.error:
            print(f"  Error:    {r.error}")
        
        if r.raw_response:
            print(f"\n{'='*60}")
            print("RAW RESPONSE:")
            print(f"{'='*60}")
            print(r.raw_response[:500])
    else:
        print("\n💡 Usage: python card_detector_ai.py test.jpg")