"""
Image Input Component for Streamlit
Sử dụng AI Vision APIs để nhận diện bài chính xác
"""
import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
from typing import Optional, List

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(src_dir, 'ml'))

# Try import AI detector
try:
    from card_detector_ai import AICardDetector, AIProvider, CardDetectionResult
    AI_DETECTOR_AVAILABLE = True
except ImportError as e:
    AI_DETECTOR_AVAILABLE = False
    print(f"⚠️ AI Card detector not available: {e}")


def image_input_component(key: str = "card_image") -> Optional[str]:
    """
    Complete image input component with AI detection
    
    Returns:
        Card string if detected successfully, None otherwise
    """
    
    # Custom CSS
    st.markdown("""
    <style>
        .detection-result {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .detected-cards-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        
        .detected-card {
            background: white;
            color: #333;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: bold;
            font-family: monospace;
            font-size: 1.1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .card-red { color: #e74c3c; border: 2px solid #e74c3c; }
        .card-black { color: #000; border: 2px solid #000; }
        
        .api-key-box {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if AI detector is available
    if not AI_DETECTOR_AVAILABLE:
        st.error("❌ AI Card Detector not available. Please install required packages.")
        st.code("pip install google-generativeai Pillow")
        return _manual_input_fallback(key)
    
    # AI Provider selection and API key
    with st.expander("🔧 AI Settings", expanded=False):
        provider, api_key = _setup_ai_provider(key)
    
    # Check API key
    if not api_key:
        st.warning("⚠️ Please enter your API key above to enable AI card detection.")
        st.markdown("""
        **How to get a FREE API key:**
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Click "Create API Key"
        3. Copy and paste the key above
        """)
        return _manual_input_fallback(key)
    
    # Tabs for input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Camera", "✍️ Manual Input"])
    
    detected_cards = None
    
    with tab1:
        detected_cards = _file_upload_tab(key, provider, api_key)
    
    with tab2:
        result = _camera_tab(key, provider, api_key)
        if result:
            detected_cards = result
    
    with tab3:
        result = _manual_input_fallback(key)
        if result:
            detected_cards = result
    
    return detected_cards


def _setup_ai_provider(key: str):
    """Setup AI provider and API key"""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        provider_name = st.selectbox(
            "AI Provider",
            ["🌟 Google Gemini (Free)", "🤖 OpenAI GPT-4", "🔮 Claude"],
            key=f"{key}_provider"
        )
    
    provider_map = {
        "🌟 Google Gemini (Free)": (AIProvider.GEMINI, "GEMINI_API_KEY"),
        "🤖 OpenAI GPT-4": (AIProvider.OPENAI, "OPENAI_API_KEY"),
        "🔮 Claude": (AIProvider.CLAUDE, "ANTHROPIC_API_KEY")
    }
    
    provider, env_var = provider_map[provider_name]
    
    with col2:
        # Check for existing key in environment or session
        existing_key = os.environ.get(env_var) or st.session_state.get(f"{key}_api_key", "")
        
        if existing_key:
            st.success(f"✅ API Key configured")
            api_key = existing_key
        else:
            api_key = st.text_input(
                f"Enter {env_var}",
                type="password",
                key=f"{key}_api_input",
                placeholder="Paste your API key here..."
            )
            
            if api_key:
                st.session_state[f"{key}_api_key"] = api_key
                os.environ[env_var] = api_key
    
    return provider, api_key


def _file_upload_tab(key: str, provider: AIProvider, api_key: str) -> Optional[str]:
    """File upload with AI detection"""
    
    st.markdown("""
    <div style="border: 2px dashed #667eea; border-radius: 12px; padding: 2rem; 
                text-align: center; background: #f8f9fa; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">📁 Upload Card Image</h3>
        <p style="color: #666; margin: 0.5rem 0 0 0;">
            Drag & drop or click to upload an image of your 13 cards
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        key=f"{key}_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        return _process_image(uploaded_file, key, provider, api_key)
    
    return None


def _camera_tab(key: str, provider: AIProvider, api_key: str) -> Optional[str]:
    """Camera capture with AI detection"""
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h3>📷 Take a Photo</h3>
        <p style="color: #666;">Capture your 13 cards with your camera</p>
    </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input(
        "Capture cards",
        key=f"{key}_camera",
        label_visibility="collapsed"
    )
    
    if camera_image:
        return _process_image(camera_image, key, provider, api_key)
    
    return None


def _process_image(uploaded_file, key: str, provider: AIProvider, api_key: str) -> Optional[str]:
    """Process uploaded image with AI detection"""
    
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Display image
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(image, caption="Your Cards", use_container_width=True)
        
        with col2:
            st.markdown("### 🎴 Detection Info")
            st.write(f"**Provider:** {provider.value}")
            st.write(f"**Image size:** {image.size[0]}x{image.size[1]}")
        
        # Detect button
        if st.button("🔍 Detect Cards with AI", type="primary", key=f"{key}_detect_btn", use_container_width=True):
            return _run_ai_detection(image, key, provider, api_key)
        
        # Check if we have previous results
        if st.session_state.get(f"{key}_last_detection"):
            result = st.session_state[f"{key}_last_detection"]
            _display_detection_result(result, key)
            
            if result.success and result.cards:
                return result.card_string
    
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
    
    return None


def _run_ai_detection(image: Image.Image, key: str, provider: AIProvider, api_key: str) -> Optional[str]:
    """Run AI detection on image"""
    
    with st.spinner(f"🤖 Analyzing with {provider.value}..."):
        try:
            # Create detector
            detector = AICardDetector(provider, api_key)
            
            # Detect
            result = detector.detect_from_image(image)
            
            # Store result
            st.session_state[f"{key}_last_detection"] = result
            
            # Display result
            _display_detection_result(result, key)
            
            if result.success and result.cards:
                return result.card_string
            
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")
            
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    return None


def _display_detection_result(result: CardDetectionResult, key: str):
    """Display detection results beautifully"""
    
    if result.success and result.cards:
        # Success header
        st.markdown(f"""
        <div class="detection-result">
            <h3 style="margin: 0;">✅ Cards Detected!</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Found {len(result.cards)} cards in {result.processing_time:.1f}s
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display cards visually
        st.markdown("### 🃏 Detected Cards")
        
        cards_html = '<div class="detected-cards-grid">'
        
        suit_symbols = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
        
        for card in result.cards:
            # Parse card
            if len(card) == 2:
                rank, suit = card[0], card[1]
            else:  # 10X
                rank, suit = card[:2], card[2]
            
            suit_symbol = suit_symbols.get(suit.upper(), suit)
            color_class = "card-red" if suit.upper() in ['H', 'D'] else "card-black"
            
            cards_html += f'<span class="detected-card {color_class}">{rank}{suit_symbol}</span>'
        
        cards_html += '</div>'
        
        st.markdown(cards_html, unsafe_allow_html=True)
        
        # Card string for copying
        st.markdown("**Card String (copy this):**")
        st.code(result.card_string, language=None)
        
        # Validation
        if result.is_valid_hand:
            st.success("✅ Valid Mau Binh hand: 13 unique cards!")
            
            if st.button("✅ Use These Cards", type="primary", key=f"{key}_use_cards", use_container_width=True):
                st.session_state['card_input'] = result.card_string
                st.session_state['cards_from_image'] = True
                return result.card_string
        else:
            if len(result.cards) != 13:
                st.warning(f"⚠️ Expected 13 cards, found {len(result.cards)}")
            
            duplicates = [c for c in result.cards if result.cards.count(c) > 1]
            if duplicates:
                st.warning(f"⚠️ Duplicate cards: {set(duplicates)}")
            
            # Allow manual correction
            st.markdown("### ✏️ Manual Correction")
            corrected = st.text_input(
                "Edit detected cards",
                value=result.card_string,
                key=f"{key}_correct"
            )
            
            if st.button("✅ Use Corrected Cards", key=f"{key}_use_corrected"):
                return corrected
        
        # Show raw response (debug)
        with st.expander("🔍 Raw AI Response"):
            st.text(result.raw_response)
    
    else:
        st.error(f"❌ Detection failed: {result.error or 'Unknown error'}")
        
        st.markdown("""
        **Tips for better detection:**
        - Make sure all 13 cards are clearly visible
        - Good lighting helps a lot
        - Avoid glare and shadows
        - Spread cards so they don't overlap
        """)
    
    return None


def _manual_input_fallback(key: str) -> Optional[str]:
    """Manual card input when AI is not available"""
    
    st.markdown("### ✍️ Manual Card Input")
    st.markdown("Enter your 13 cards manually:")
    
    # Initialize selected cards in session state
    if f'{key}_manual_cards' not in st.session_state:
        st.session_state[f'{key}_manual_cards'] = []
    
    selected = st.session_state[f'{key}_manual_cards']
    
    # Display current selection
    if selected:
        st.markdown(f"**Selected ({len(selected)}/13):** `{' '.join(selected)}`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All", key=f"{key}_clear"):
                st.session_state[f'{key}_manual_cards'] = []
                st.rerun()
        
        with col2:
            if len(selected) == 13:
                if st.button("✅ Use These Cards", type="primary", key=f"{key}_use_manual"):
                    return ' '.join(selected)
    
    # Card selection grid
    st.markdown("**Click to select cards:**")
    
    suits = [('♠', 'S'), ('♥', 'H'), ('♦', 'D'), ('♣', 'C')]
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    
    for suit_symbol, suit_code in suits:
        cols = st.columns(13)
        
        for i, rank in enumerate(ranks):
            card_id = f"{rank}{suit_code}"
            
            with cols[i]:
                is_selected = card_id in selected
                is_disabled = is_selected or (len(selected) >= 13 and not is_selected)
                
                button_label = f"{rank}{suit_symbol}"
                
                if is_selected:
                    # Show as selected (different style)
                    st.markdown(f"<div style='background: #667eea; color: white; padding: 0.3rem; border-radius: 4px; text-align: center; font-weight: bold;'>{button_label}</div>", unsafe_allow_html=True)
                else:
                    if st.button(
                        button_label,
                        key=f"{key}_card_{card_id}",
                        disabled=is_disabled,
                        use_container_width=True
                    ):
                        if card_id not in selected and len(selected) < 13:
                            st.session_state[f'{key}_manual_cards'].append(card_id)
                            st.rerun()
    
    # Also allow text input
    st.markdown("---")
    st.markdown("**Or paste card string directly:**")
    
    text_input = st.text_input(
        "Card string",
        placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
        key=f"{key}_text_manual"
    )
    
    if text_input:
        if st.button("✅ Use Text Input", key=f"{key}_use_text"):
            return text_input
    
    return None


# ============ TESTING ============

if __name__ == "__main__":
    import streamlit as st
    
    st.title("🎴 AI Card Detection Test")
    
    result = image_input_component("test")
    
    if result:
        st.success(f"✅ Final result: {result}")