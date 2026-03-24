"""
AI-Powered Card Detection
Sử dụng AI Vision APIs để nhận diện bài chính xác
Supports: Google Gemini (FREE), OpenAI GPT-4 Vision, Google Cloud Vision
"""

import base64
import json
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIProvider(Enum):
    GEMINI = "gemini"           # Google Gemini - FREE!
    OPENAI = "openai"           # OpenAI GPT-4 Vision
    CLAUDE = "claude"           # Anthropic Claude
    GOOGLE_VISION = "google_vision"  # Google Cloud Vision


@dataclass
class CardDetectionResult:
    """Result from AI card detection"""
    cards: List[str]           # List of card strings like ["AS", "KH", "QD"]
    raw_response: str          # Raw AI response
    confidence: float          # Overall confidence
    provider: AIProvider       # Which AI was used
    processing_time: float     # Time taken
    success: bool
    error: Optional[str] = None
    
    @property
    def card_string(self) -> str:
        """Return space-separated card string"""
        return ' '.join(self.cards)
    
    @property
    def is_valid_hand(self) -> bool:
        """Check if we have exactly 13 unique cards"""
        return len(self.cards) == 13 and len(set(self.cards)) == 13


class AICardDetector:
    """
    Main AI Card Detector class
    Supports multiple AI providers
    """
    
    # Prompt template for card detection
    DETECTION_PROMPT = """Analyze this image of playing cards and identify ALL cards visible.

IMPORTANT RULES:
1. List EXACTLY the cards you can see clearly
2. Format each card as: RankSuit (e.g., AS = Ace of Spades, 10H = Ten of Hearts)
3. Ranks: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
4. Suits: S = Spades (♠), H = Hearts (♥), D = Diamonds (♦), C = Clubs (♣)
5. Separate cards with spaces
6. Only output the card list, nothing else

Example output format:
AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S

Now analyze the image and list all visible cards:"""

    def __init__(self, provider: AIProvider = AIProvider.GEMINI, api_key: str = None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key_from_env()
        
        # Validate API key
        if not self.api_key and provider != AIProvider.GOOGLE_VISION:
            logger.warning(f"No API key provided for {provider.value}. Set environment variable or pass api_key.")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        env_vars = {
            AIProvider.GEMINI: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            AIProvider.OPENAI: ["OPENAI_API_KEY"],
            AIProvider.CLAUDE: ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            AIProvider.GOOGLE_VISION: ["GOOGLE_APPLICATION_CREDENTIALS"],
        }
        
        for var in env_vars.get(self.provider, []):
            key = os.environ.get(var)
            if key:
                return key
        
        return None
    
    def detect_from_image(self, image: Image.Image) -> CardDetectionResult:
        """Detect cards from PIL Image"""
        import time
        start_time = time.time()
        
        try:
            # Convert image to base64
            base64_image = self._image_to_base64(image)
            
            # Call appropriate API
            if self.provider == AIProvider.GEMINI:
                result = self._detect_with_gemini(base64_image)
            elif self.provider == AIProvider.OPENAI:
                result = self._detect_with_openai(base64_image)
            elif self.provider == AIProvider.CLAUDE:
                result = self._detect_with_claude(base64_image)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            processing_time = time.time() - start_time
            
            # Parse the response
            cards = self._parse_card_response(result)
            
            return CardDetectionResult(
                cards=cards,
                raw_response=result,
                confidence=0.95 if len(cards) == 13 else 0.7,
                provider=self.provider,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return CardDetectionResult(
                cards=[],
                raw_response="",
                confidence=0.0,
                provider=self.provider,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def detect_from_file(self, file_path: str) -> CardDetectionResult:
        """Detect cards from file path"""
        image = Image.open(file_path)
        return self.detect_from_image(image)
    
    def detect_from_bytes(self, image_bytes: bytes) -> CardDetectionResult:
        """Detect cards from bytes"""
        image = Image.open(io.BytesIO(image_bytes))
        return self.detect_from_image(image)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        # Resize if too large (API limits)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Encode to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _detect_with_gemini(self, base64_image: str) -> str:
        """Detect cards using Google Gemini (FREE!)"""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini Pro Vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Create image part
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64_image
        }
        
        # Generate response
        response = model.generate_content([
            self.DETECTION_PROMPT,
            image_part
        ])
        
        return response.text
    
    def _detect_with_openai(self, base64_image: str) -> str:
        """Detect cards using OpenAI GPT-4 Vision"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.DETECTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _detect_with_claude(self, base64_image: str) -> str:
        """Detect cards using Anthropic Claude"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": self.DETECTION_PROMPT
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text
    
    def _parse_card_response(self, response: str) -> List[str]:
        """Parse AI response to extract card list"""
        # Clean up response
        response = response.strip().upper()
        
        # Remove common prefixes/suffixes
        response = re.sub(r'^(THE CARDS ARE|CARDS:|HERE ARE THE CARDS|I CAN SEE)[:\s]*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\.$', '', response)
        
        # Extract cards using regex
        # Match patterns like: AS, KH, 10D, 2C, etc.
        card_pattern = r'\b(10|[2-9]|[JQKA])([SHDC])\b'
        matches = re.findall(card_pattern, response)
        
        cards = []
        for rank, suit in matches:
            card = f"{rank}{suit}"
            if card not in cards:  # Avoid duplicates
                cards.append(card)
        
        # If regex didn't work well, try splitting
        if len(cards) < 5:
            # Try space-separated
            tokens = response.split()
            for token in tokens:
                token = token.strip('.,;:()[]')
                if self._is_valid_card(token) and token not in cards:
                    cards.append(token)
        
        return cards
    
    def _is_valid_card(self, card: str) -> bool:
        """Check if string is a valid card"""
        if len(card) < 2 or len(card) > 3:
            return False
        
        valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}
        valid_suits = {'S', 'H', 'D', 'C'}
        
        card = card.upper()
        
        if len(card) == 2:
            rank, suit = card[0], card[1]
        else:  # len == 3, must be 10X
            rank, suit = card[:2], card[2]
        
        return rank in valid_ranks and suit in valid_suits


class MultiProviderDetector:
    """
    Try multiple AI providers for best results
    """
    
    def __init__(self):
        self.providers = []
        
        # Try to initialize available providers
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            self.providers.append(AICardDetector(AIProvider.GEMINI))
        
        if os.environ.get("OPENAI_API_KEY"):
            self.providers.append(AICardDetector(AIProvider.OPENAI))
        
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.providers.append(AICardDetector(AIProvider.CLAUDE))
    
    def detect(self, image: Image.Image) -> CardDetectionResult:
        """Try detection with available providers"""
        for detector in self.providers:
            try:
                result = detector.detect_from_image(image)
                if result.success and result.is_valid_hand:
                    return result
            except Exception as e:
                logger.warning(f"Provider {detector.provider.value} failed: {e}")
                continue
        
        # If no provider succeeded, return error
        return CardDetectionResult(
            cards=[],
            raw_response="",
            confidence=0.0,
            provider=AIProvider.GEMINI,
            processing_time=0,
            success=False,
            error="All AI providers failed. Please check API keys."
        )


# ============ STREAMLIT INTEGRATION ============

def create_ai_detector_ui():
    """Create Streamlit UI for AI detector configuration"""
    import streamlit as st
    
    st.markdown("### 🤖 AI Provider Settings")
    
    provider = st.selectbox(
        "Select AI Provider",
        options=["Google Gemini (Free)", "OpenAI GPT-4", "Claude"],
        index=0
    )
    
    provider_map = {
        "Google Gemini (Free)": AIProvider.GEMINI,
        "OpenAI GPT-4": AIProvider.OPENAI,
        "Claude": AIProvider.CLAUDE
    }
    
    selected_provider = provider_map[provider]
    
    # API Key input
    env_var_map = {
        AIProvider.GEMINI: "GEMINI_API_KEY",
        AIProvider.OPENAI: "OPENAI_API_KEY",
        AIProvider.CLAUDE: "ANTHROPIC_API_KEY"
    }
    
    env_var = env_var_map[selected_provider]
    existing_key = os.environ.get(env_var, "")
    
    if existing_key:
        st.success(f"✅ {env_var} found in environment")
        api_key = existing_key
    else:
        api_key = st.text_input(
            f"Enter {env_var}",
            type="password",
            help=f"Get your API key from the provider's website"
        )
        
        if api_key:
            os.environ[env_var] = api_key
    
    return selected_provider, api_key


# ============ TESTING ============

if __name__ == "__main__":
    import sys
    
    print("Testing AI Card Detector...")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No GEMINI_API_KEY found. Set it with:")
        print("   export GEMINI_API_KEY='your-api-key'")
        print("   Or on Windows:")
        print("   set GEMINI_API_KEY=your-api-key")
        sys.exit(1)
    
    print(f"✅ API key found")
    
    # Create test image (you would use a real image)
    detector = AICardDetector(AIProvider.GEMINI, api_key)
    
    # Test with a sample image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing with: {image_path}")
        
        result = detector.detect_from_file(image_path)
        
        print(f"\nResults:")
        print(f"  Success: {result.success}")
        print(f"  Cards: {result.card_string}")
        print(f"  Count: {len(result.cards)}")
        print(f"  Valid hand: {result.is_valid_hand}")
        print(f"  Time: {result.processing_time:.2f}s")
        print(f"  Raw response: {result.raw_response}")
    else:
        print("Usage: python card_detector_ai.py <image_path>")