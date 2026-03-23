"""
Beautiful card rendering component for Streamlit
Supports both formats: "AS" and "A♠"
"""

def parse_card_string(card_str):
    """
    Parse card string to (rank, suit_code)
    Supports both formats:
    - "AS", "KH", "10D", "JC"
    - "A♠", "K♥", "10♦", "J♣"
    
    Returns:
        tuple: (rank_str, suit_code) e.g., ("A", "S")
    """
    # Unicode symbol to code mapping
    unicode_to_code = {
        '♠': 'S',
        '♥': 'H',
        '♦': 'D',
        '♣': 'C'
    }
    
    card_str = card_str.strip()
    
    # Check if last char is Unicode suit symbol
    if card_str[-1] in unicode_to_code:
        suit_code = unicode_to_code[card_str[-1]]
        rank = card_str[:-1]
    # Check if last char is letter suit code
    elif card_str[-1].upper() in ['S', 'H', 'D', 'C']:
        suit_code = card_str[-1].upper()
        rank = card_str[:-1]
    else:
        # Fallback
        suit_code = 'S'
        rank = card_str
    
    return rank, suit_code


def get_card_html(card_str):
    """
    Render single card as beautiful HTML
    
    Args:
        card_str: Card string like "AS", "KH", "10D" or "A♠", "K♥", "10♦"
    
    Returns:
        HTML string for the card
    """
    
    suit_symbols = {
        'S': '♠',
        'H': '♥', 
        'D': '♦',
        'C': '♣'
    }
    
    suit_colors = {
        'S': '#1a1a2e',      # Dark navy for Spades
        'H': '#e63946',      # Red for Hearts
        'D': '#f4a261',      # Orange for Diamonds
        'C': '#2a9d8f'       # Teal for Clubs
    }
    
    # Parse card
    rank, suit_code = parse_card_string(card_str)
    
    color = suit_colors.get(suit_code, '#000')
    symbol = suit_symbols.get(suit_code, '?')
    
    # Beautiful 3D card HTML
    html = f'''<div style="
        display: inline-block;
        background: linear-gradient(145deg, #ffffff 0%, #f0f0f0 100%);
        border: 3px solid {color};
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.8);
        font-family: 'Arial Black', sans-serif;
        min-width: 55px;
        text-align: center;
    "><span style="
        color: {color};
        font-size: 1.5rem;
        font-weight: 900;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    ">{rank}{symbol}</span></div>'''
    
    return html


def render_hand_html(cards_list, title="", hand_rank="", chi_type="back"):
    """
    Render full hand with beautiful styling
    """
    
    # Gradient colors for different chi
    gradients = {
        'back': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'middle': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'front': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
    }
    
    gradient = gradients.get(chi_type, gradients['back'])
    
    # Build cards HTML
    cards_html = ""
    for card in cards_list:
        cards_html += get_card_html(card)
    
    # Title HTML
    title_html = ""
    if title:
        title_html = f'<div style="color: white; font-size: 1.3rem; font-weight: 800; margin-bottom: 14px; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">{title}</div>'
    
    # Hand rank HTML
    rank_html = ""
    if hand_rank:
        rank_html = f'<div style="color: #FFD700; font-size: 1.1rem; font-weight: 700; margin-top: 12px; text-align: center; text-shadow: 0 2px 4px rgba(0,0,0,0.5); background: rgba(0,0,0,0.25); padding: 8px 14px; border-radius: 8px;">⭐ {hand_rank}</div>'
    
    html = f'''<div style="
        background: {gradient};
        border-radius: 14px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    ">{title_html}<div style="
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: center;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 14px;
    ">{cards_html}</div>{rank_html}</div>'''
    
    return html


def render_comparison_cards(back, middle, front, evaluations):
    """
    Render all three chi in beautiful responsive grid
    """
    
    back_eval, mid_eval, front_eval = evaluations
    
    back_html = render_hand_html(back, "🔵 Chi 1 (Back - 5 cards)", back_eval, "back")
    middle_html = render_hand_html(middle, "🟢 Chi 2 (Middle - 5 cards)", mid_eval, "middle")
    front_html = render_hand_html(front, "🟡 Chi cuối (Front - 3 cards)", front_eval, "front")
    
    html = f'''<div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    ">{back_html}{middle_html}{front_html}</div>'''
    
    return html


def render_input_cards_preview(cards_list):
    """
    Preview of input cards (all 13)
    """
    
    cards_html = ""
    for card in cards_list:
        cards_html += get_card_html(card)
    
    html = f'''<div style="
        background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
        border-radius: 14px;
        padding: 18px;
        margin: 14px 0;
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    "><div style="
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    ">🃏 Your 13 Cards</div><div style="
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        justify-content: center;
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px;
    ">{cards_html}</div></div>'''
    
    return html