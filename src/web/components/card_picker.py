"""
Interactive Card Picker Component for Streamlit
Allows both clicking and typing to select cards
"""
import streamlit as st
import random


def rerun_app():
    """Compatible rerun for all Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.warning("Please refresh the page manually")


def render_selected_cards_preview(selected_cards):
    """Render preview of selected cards"""
    
    if not selected_cards:
        return ""
    
    suit_symbols = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    suit_colors = {
        'S': '#1a1a2e',
        'H': '#e63946',
        'D': '#f4a261',
        'C': '#2a9d8f'
    }
    
    cards_html = ""
    for card in selected_cards:
        rank = card[:-1]
        suit = card[-1]
        symbol = suit_symbols.get(suit, suit)
        color = suit_colors.get(suit, '#333')
        
        cards_html += f'''<span style="display: inline-block; background: white; border: 2px solid {color}; border-radius: 6px; padding: 4px 8px; margin: 2px; color: {color}; font-weight: bold; font-size: 0.9rem;">{rank}{symbol}</span>'''
    
    return f'''<div style="background: #f8f9fa; border-radius: 10px; padding: 12px; margin: 10px 0; min-height: 50px;"><div style="color: #666; font-size: 0.85rem; margin-bottom: 8px;">Selected ({len(selected_cards)}/13):</div><div style="display: flex; flex-wrap: wrap; gap: 4px;">{cards_html}</div></div>'''


def interactive_card_picker():
    """
    Interactive card picker with click-to-select functionality
    
    Returns:
        str or None: Space-separated card string if 13 cards selected and USE button clicked
    """
    
    # Initialize session state
    if 'picker_selected_cards' not in st.session_state:
        st.session_state.picker_selected_cards = []
    
    selected = st.session_state.picker_selected_cards
    
    # Card data
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    suits = [
        ('♠', 'S', 'Spades', '#1a1a2e'),
        ('♥', 'H', 'Hearts', '#e63946'),
        ('♦', 'D', 'Diamonds', '#f4a261'),
        ('♣', 'C', 'Clubs', '#2a9d8f')
    ]
    
    # Header with count
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if len(selected) == 13:
            st.success(f"✅ **{len(selected)}/13 cards selected** - Ready to solve!")
        elif len(selected) > 0:
            st.info(f"🎴 **{len(selected)}/13 cards selected**")
        else:
            st.warning("👆 **Click cards to select (need 13)**")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True, disabled=len(selected)==0, key="picker_clear_btn"):
            st.session_state.picker_selected_cards = []
            rerun_app()
    
    with col3:
        if st.button("🎲 Random 13", use_container_width=True, key="picker_random_btn"):
            all_cards = [f"{r}{s[1]}" for r in ranks for s in suits]
            st.session_state.picker_selected_cards = random.sample(all_cards, 13)
            rerun_app()
    
    # Show selected cards preview
    if selected:
        st.markdown(render_selected_cards_preview(selected), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Card grid - 4 columns for 4 suits
    suit_cols = st.columns(4)
    
    for suit_idx, (symbol, code, name, color) in enumerate(suits):
        with suit_cols[suit_idx]:
            # Suit header
            st.markdown(
                f'<div style="text-align: center; font-size: 1.5rem; color: {color}; font-weight: bold; margin-bottom: 10px;">{symbol} {name}</div>',
                unsafe_allow_html=True
            )
            
            # Cards for this suit
            for rank in ranks:
                card = f"{rank}{code}"
                is_selected = card in selected
                is_disabled = len(selected) >= 13 and not is_selected
                
                # Button label
                if is_selected:
                    label = f"✓ {rank}{symbol}"
                else:
                    label = f"{rank}{symbol}"
                
                # Button type
                btn_type = "primary" if is_selected else "secondary"
                
                if st.button(
                    label,
                    key=f"pick_card_{card}",
                    use_container_width=True,
                    type=btn_type,
                    disabled=is_disabled
                ):
                    if is_selected:
                        st.session_state.picker_selected_cards.remove(card)
                    else:
                        st.session_state.picker_selected_cards.append(card)
                    rerun_app()
    
    st.markdown("---")
    
    # Return result if 13 cards selected
    if len(selected) == 13:
        result = " ".join(selected)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code(result, language=None)
        
        with col2:
            if st.button("🚀 USE THESE CARDS", type="primary", use_container_width=True, key="picker_use_btn"):
                return result
    
    return None


def quick_select_buttons():
    """Quick select common hands for testing"""
    
    st.markdown("##### 🎯 Quick Select (Examples)")
    
    examples = {
        "🔥 Premium": "AS AH AD KC KD QS QH JD 10C 9S 8H 7D 6C",
        "⚖️ Balanced": "KS QH JD 10C 9S 8H 7D 6C 5S 4H 3D 2C AS",
        "🎰 Flush": "AS KS QS JS 9S 7H 6H 4H 3H 2D 8C 5D 4C",
        "📈 Straight": "9S 8H 7D 6C 5S AH KD QC JH 10D 4S 3H 2C"
    }
    
    cols = st.columns(4)
    
    for i, (name, cards) in enumerate(examples.items()):
        with cols[i]:
            if st.button(name, key=f"quick_select_{i}", use_container_width=True):
                st.session_state.picker_selected_cards = cards.split()
                rerun_app()