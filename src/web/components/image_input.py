"""
Image Input - Upload/Paste ảnh và nhận diện bài
Version 2.1: FIXED all use_container_width warnings
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import sys
import os
from pathlib import Path
from typing import Optional
import io
import base64

# Safe rerun
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            st.warning("Please refresh (F5)")


# Paths
current_dir = Path(__file__).parent
ml_dir = current_dir.parent.parent / "ml"
sys.path.insert(0, str(ml_dir))

# Detector
DETECTOR_AVAILABLE = False
CardDetector = None
try:
    from card_detector_ai import CardDetector as CD
    CardDetector = CD
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CardDetector: {e}")

# Paste button
PASTE_AVAILABLE = False
try:
    from streamlit_paste_button import paste_image_button
    PASTE_AVAILABLE = True
except ImportError:
    pass


def image_input_component(key: str = "card_image") -> Optional[str]:
    """Component nhập ảnh và nhận diện bài"""
    
    # CSS
    st.markdown("""
    <style>
        .card-badge {
            display: inline-block;
            padding: 8px 12px;
            margin: 4px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1rem;
            background: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        .card-red { color: #e74c3c; border: 2px solid #e74c3c; }
        .card-black { color: #2c3e50; border: 2px solid #2c3e50; }
        
        .stImage {
            max-width: 100% !important;
        }
        .stImage > img {
            max-height: 500px !important;
            width: auto !important;
            object-fit: contain !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Nếu đã có kết quả
    if st.session_state.get(f'{key}_result'):
        return _show_result(key)
    
    # Nhập ảnh
    return _show_input(key)


def _show_input(key: str) -> Optional[str]:
    """Hiện giao diện nhập ảnh"""
    
    image_key = f'{key}_current_image'
    
    # Paste
    if PASTE_AVAILABLE:
        st.info("📸 **Cách nhanh:** Chụp màn hình (Win+Shift+S) → Click Paste")
        
        paste_result = paste_image_button(
            label="📋 Paste ảnh từ clipboard",
            key=f"{key}_paste",
            background_color="#667eea",
            hover_background_color="#764ba2",
        )
        
        if paste_result and paste_result.image_data:
            try:
                if isinstance(paste_result.image_data, str):
                    img_data = paste_result.image_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                else:
                    img = paste_result.image_data
                
                st.session_state[image_key] = img
                st.success("✅ Ảnh đã paste!")
            except Exception as e:
                st.error(f"Lỗi paste: {e}")
    else:
        st.warning("📦 Cài `pip install streamlit-paste-button` để paste!")
    
    st.markdown("---")
    
    # Upload
    st.markdown("**📁 Hoặc upload ảnh:**")
    uploaded = st.file_uploader(
        "Chọn ảnh",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        key=f"{key}_upload",
        label_visibility="collapsed"
    )
    
    if uploaded:
        try:
            st.session_state[image_key] = Image.open(uploaded)
        except Exception as e:
            st.error(f"Lỗi đọc ảnh: {e}")
    
    # Lấy image
    image = st.session_state.get(image_key)
    
    # Nếu có ảnh
    if image:
        display_img = _prepare_display_image(image)
        
        # ✅ FIX: Bỏ use_container_width
        st.image(
            display_img, 
            caption=f"Ảnh của bạn ({image.size[0]}x{image.size[1]})",
            width=600,
        )
        
        st.caption(f"📐 Kích thước gốc: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
        
        col1, col2, col3 = st.columns(3)
        
        # ✅ FIX: Bỏ use_container_width ở buttons
        with col1:
            if st.button("🔍 NHẬN DIỆN", type="primary", key=f"{key}_detect"):
                _run_detection(image, key)
        
        with col2:
            if st.button("✨ Enhance", key=f"{key}_enhance"):
                enhanced = _enhance_image_for_detection(image)
                st.session_state[image_key] = enhanced
                st.success("✅ Đã enhance ảnh!")
                safe_rerun()
        
        with col3:
            if st.button("🗑️ Xóa", key=f"{key}_clear"):
                _clear_image_state(key)
    
    return None


def _prepare_display_image(image: Image.Image, max_display_size: int = 800) -> Image.Image:
    """Chuẩn bị ảnh để hiển thị"""
    display = image.copy()
    
    if max(display.size) > max_display_size:
        ratio = max_display_size / max(display.size)
        new_size = (int(display.size[0] * ratio), int(display.size[1] * ratio))
        display = display.resize(new_size, Image.Resampling.LANCZOS)
    
    return display


def _enhance_image_for_detection(image: Image.Image) -> Image.Image:
    """Enhance ảnh để detect tốt hơn"""
    from PIL import ImageEnhance, ImageFilter
    
    img = image.copy()
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)
    
    img = img.filter(ImageFilter.SHARPEN)
    
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.15)
    
    return img


def _clear_image_state(key: str):
    """Clear state"""
    keys_to_clear = [
        f'{key}_current_image',
        f'{key}_result',
    ]
    
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    
    safe_rerun()


def _run_detection(image, key: str):
    """Chạy nhận diện"""
    
    if not DETECTOR_AVAILABLE or CardDetector is None:
        st.error("❌ Detector không khả dụng!")
        return
    
    with st.spinner("🤖 Đang nhận diện bài..."):
        try:
            detector = CardDetector()
            
            status = detector.get_status()
            if status['total_keys'] == 0:
                st.error("❌ Không có API key!")
                st.code("""
# Tạo file config/ai_config.yaml:
gemini:
  api_keys:
    - 'YOUR_KEY_HERE'
                """)
                return
            
            result = detector.detect(image)
            
            if result.success and result.cards:
                st.session_state[f'{key}_result'] = {
                    'cards': result.cards,
                    'card_string': result.card_string,
                    'provider': result.provider,
                    'time': result.time,
                    'is_valid': result.is_valid,
                    'raw_response': result.raw_response,
                }
                safe_rerun()
            else:
                st.error(f"❌ Nhận diện thất bại: {result.error}")
                
                if result.raw_response:
                    with st.expander("🔍 Raw Response"):
                        st.code(result.raw_response)
                
                st.info("💡 **Thử:**\n- Click nút **✨ Enhance** rồi detect lại\n- Chụp ảnh rõ hơn\n- Đảm bảo ánh sáng tốt")
                
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            
            import traceback
            with st.expander("🔍 Chi tiết"):
                st.code(traceback.format_exc())


def _show_result(key: str) -> Optional[str]:
    """Hiện kết quả"""
    
    result = st.session_state[f'{key}_result']
    cards = result['cards']
    
    st.success(f"✅ **{result['provider'].upper()}** phát hiện {len(cards)} lá ({result['time']:.1f}s)")
    
    _display_cards(cards)
    
    if result['is_valid']:
        st.success("🎉 Hoàn hảo! Đủ 13 lá bài duy nhất.")
    else:
        if len(cards) != 13:
            st.warning(f"⚠️ Cần 13 lá, phát hiện {len(cards)} lá")
        
        duplicates = [c for c in cards if cards.count(c) > 1]
        if duplicates:
            st.warning(f"⚠️ Lá trùng: {set(duplicates)}")
    
    card_string = st.text_input(
        "✏️ Chỉnh sửa nếu cần:",
        value=result['card_string'],
        key=f"{key}_edit",
        help="Format: AS KH QD JC 10S..."
    )
    
    if result.get('raw_response'):
        with st.expander("🔍 Raw AI Response"):
            st.code(result['raw_response'])
    
    col1, col2 = st.columns(2)
    
    # ✅ FIX: Bỏ use_container_width
    with col1:
        if st.button("🚀 GIẢI BÀI", type="primary", key=f"{key}_solve"):
            st.session_state[f'{key}_result'] = None
            st.session_state['card_input'] = card_string.upper()
            st.session_state['cards_from_image'] = True
            return card_string.upper()
    
    with col2:
        if st.button("📷 Chụp lại", key=f"{key}_retry"):
            st.session_state[f'{key}_result'] = None
            safe_rerun()
    
    return None


def _display_cards(cards: list):
    """Hiện cards dạng badge"""
    
    symbols = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    
    html = '<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:1rem 0;">'
    
    for card in cards:
        if len(card) >= 2:
            if card.startswith('10'):
                rank, suit = '10', card[2] if len(card) > 2 else ''
            else:
                rank, suit = card[0], card[1] if len(card) > 1 else ''
        else:
            rank, suit = card, ''
        
        symbol = symbols.get(suit.upper(), suit)
        color = "card-red" if suit.upper() in ['H', 'D'] else "card-black"
        
        html += f'<span class="card-badge {color}">{rank}{symbol}</span>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)