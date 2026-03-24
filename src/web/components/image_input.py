"""
Image Input - Upload/Paste ảnh và nhận diện bài
Đơn giản hóa - chỉ dùng Gemini + YOLO
"""

import streamlit as st
from PIL import Image
import sys
import os
from pathlib import Path
from typing import Optional

# Add paths
current_dir = Path(__file__).parent
ml_dir = current_dir.parent.parent / "ml"
sys.path.insert(0, str(ml_dir))

# Import detector
try:
    from card_detector_ai import CardDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CardDetector not available: {e}")
    DETECTOR_AVAILABLE = False

# Try paste button
try:
    from streamlit_paste_button import paste_image_button
    PASTE_AVAILABLE = True
except ImportError:
    PASTE_AVAILABLE = False


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
    </style>
    """, unsafe_allow_html=True)
    
    # Nếu đã có kết quả
    if st.session_state.get(f'{key}_result'):
        return _show_result(key)
    
    # Nhập ảnh
    return _show_input(key)


def _show_input(key: str) -> Optional[str]:
    """Hiện giao diện nhập ảnh"""
    
    image = None
    
    # Paste button
    if PASTE_AVAILABLE:
        st.info("📸 **Cách nhanh:** Chụp màn hình (Win+Shift+S) → Click nút Paste bên dưới")
        
        paste_result = paste_image_button(
            label="📋 Paste ảnh từ clipboard",
            key=f"{key}_paste",
            background_color="#667eea",
            hover_background_color="#764ba2",
        )
        
        if paste_result.image_data:
            image = paste_result.image_data
    
    st.markdown("---")
    
    # Upload
    st.markdown("**📁 Hoặc upload ảnh:**")
    uploaded = st.file_uploader(
        "Chọn ảnh",
        type=['jpg', 'jpeg', 'png', 'webp'],
        key=f"{key}_upload",
        label_visibility="collapsed"
    )
    
    if uploaded:
        try:
            image = Image.open(uploaded)
        except Exception as e:
            st.error(f"Lỗi đọc ảnh: {e}")
    
    # Nếu có ảnh
    if image:
        st.image(image, caption="Ảnh của bạn", use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 NHẬN DIỆN", type="primary", use_container_width=True, key=f"{key}_detect"):
                _run_detection(image, key)
        
        with col2:
            if st.button("🗑️ Xóa", use_container_width=True, key=f"{key}_clear"):
                st.rerun()
    
    return None


def _run_detection(image: Image.Image, key: str):
    """Chạy nhận diện"""
    
    if not DETECTOR_AVAILABLE:
        st.error("❌ Detector không khả dụng!")
        return
    
    with st.spinner("🤖 Đang nhận diện bài..."):
        try:
            detector = CardDetector()
            result = detector.detect(image)
            
            if result.success and result.cards:
                st.session_state[f'{key}_result'] = {
                    'cards': result.cards,
                    'card_string': result.card_string,
                    'provider': result.provider,
                    'time': result.time,
                    'is_valid': result.is_valid
                }
                st.rerun()
            else:
                st.error(f"❌ Nhận diện thất bại: {result.error}")
                st.info("💡 Thử chụp lại với ánh sáng tốt, bài rõ ràng")
                
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")


def _show_result(key: str) -> Optional[str]:
    """Hiện kết quả nhận diện"""
    
    result = st.session_state[f'{key}_result']
    cards = result['cards']
    
    # Status
    st.success(f"✅ **{result['provider'].upper()}** phát hiện {len(cards)} lá ({result['time']:.1f}s)")
    
    # Cards display
    _display_cards(cards)
    
    # Validation
    if result['is_valid']:
        st.success("🎉 Hoàn hảo! Đủ 13 lá bài duy nhất.")
    else:
        if len(cards) != 13:
            st.warning(f"⚠️ Cần 13 lá, phát hiện {len(cards)} lá")
        
        duplicates = [c for c in cards if cards.count(c) > 1]
        if duplicates:
            st.warning(f"⚠️ Lá trùng: {set(duplicates)}")
    
    # Edit
    card_string = st.text_input(
        "✏️ Chỉnh sửa nếu cần:",
        value=result['card_string'],
        key=f"{key}_edit"
    )
    
    # Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 GIẢI BÀI", type="primary", use_container_width=True, key=f"{key}_solve"):
            st.session_state[f'{key}_result'] = None
            st.session_state['card_input'] = card_string.upper()
            return card_string.upper()
    
    with col2:
        if st.button("📷 Chụp lại", use_container_width=True, key=f"{key}_retry"):
            st.session_state[f'{key}_result'] = None
            st.rerun()
    
    return None


def _display_cards(cards: list):
    """Hiện các lá bài dạng badge"""
    
    symbols = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    
    html = '<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:1rem 0;">'
    
    for card in cards:
        if len(card) == 2:
            rank, suit = card[0], card[1]
        else:
            rank, suit = card[:2], card[2]
        
        symbol = symbols.get(suit.upper(), suit)
        color = "card-red" if suit.upper() in ['H', 'D'] else "card-black"
        
        html += f'<span class="card-badge {color}">{rank}{symbol}</span>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# Alias
def quick_image_input(key: str = "quick") -> Optional[str]:
    return image_input_component(key)