"""
Debug script - Xem Gemini trả về gì
"""

import os
from google import genai
from google.genai import types
from PIL import Image
import sys

# ===== CONFIG =====
# Thay bằng key thật của bạn
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")

if API_KEY == "YOUR_KEY_HERE":
    print("❌ Set GEMINI_API_KEY environment variable!")
    sys.exit(1)

os.environ["GEMINI_API_KEY"] = API_KEY
client = genai.Client()

# ===== PROMPT =====
PROMPT = """You are a playing card recognition expert.

Identify the 13 playing cards in this image.

Output format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S

(Ranks: A K Q J 10 9 8 7 6 5 4 3 2, Suits: S H D C)

Your output:"""

# ===== TEST =====
if len(sys.argv) < 2:
    print("Usage: python debug_gemini.py <image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"📸 Testing with: {image_path}")
print("="*60)

try:
    # Load image
    img = Image.open(image_path)
    print(f"✅ Image loaded: {img.size}, mode={img.mode}")
    
    # Call Gemini
    print("\n🤖 Calling Gemini 2.5 Flash...")
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PROMPT, img],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        )
    )
    
    print("\n" + "="*60)
    print("RAW RESPONSE:")
    print("="*60)
    print(response.text)
    print("="*60)
    
    # Parse
    import re
    pattern = r'\b(10|[2-9JQKA])([SHDC])\b'
    matches = re.findall(response.text.upper(), pattern)
    cards = [f"{r}{s}" for r, s in matches if f"{r}{s}" not in [f"{r}{s}" for r, s in matches[:matches.index((r, s))]]]  # Remove duplicates
    
    print(f"\n📋 PARSED CARDS ({len(cards)}):")
    print(cards)
    print("="*60)
    
    if len(cards) == 13:
        print("✅ SUCCESS - Detected exactly 13 cards!")
    else:
        print(f"⚠️ WARNING - Expected 13 cards, got {len(cards)}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()