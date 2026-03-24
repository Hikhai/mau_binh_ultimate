"""
Professional Streamlit Web App for Mau Binh Solver
Production-ready with analytics, monetization, and BEAUTIFUL CARD DISPLAY
"""
import streamlit as st
import sys
import os
from datetime import datetime
import time
import json
from pathlib import Path

# ===== COMPATIBILITY FIX =====
def rerun():
    """Compatible rerun for all Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.warning("Please refresh the page manually")

# FIX: Add correct paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src/

sys.path.insert(0, os.path.join(parent_dir, 'core'))
sys.path.insert(0, os.path.join(parent_dir, 'engines'))
sys.path.insert(0, parent_dir)

# Import components path
sys.path.insert(0, os.path.join(current_dir, 'components'))

from card import Deck
from ultimate_solver import UltimateSolver, SolverMode
from evaluator import HandEvaluator

# Import visual components
try:
    from card_renderer import (
        render_comparison_cards, 
        render_input_cards_preview,
        get_card_html
    )
    VISUAL_CARDS_AVAILABLE = True
except ImportError:
    VISUAL_CARDS_AVAILABLE = False
    print("⚠️ Warning: Visual card components not found. Using text display.")


# ============== CONFIG ==============

st.set_page_config(
    page_title="Mau Binh AI Solver Pro",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.85rem;
        border: none;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem !important;
        }
        
        .sub-header {
            font-size: 1rem !important;
        }
        
        .stButton>button {
            font-size: 1rem !important;
            padding: 0.6rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============== SESSION STATE ==============

if 'solve_count' not in st.session_state:
    st.session_state.solve_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# Usage tracking
if 'daily_usage' not in st.session_state:
    st.session_state.daily_usage = {
        'date': datetime.now().date(),
        'count': 0
    }

# Reset daily
if st.session_state.daily_usage['date'] != datetime.now().date():
    st.session_state.daily_usage = {
        'date': datetime.now().date(),
        'count': 0
    }

# Free tier limits
FREE_DAILY_LIMIT = 99999
PREMIUM_MODES = []  # No premium restrictions

def check_usage_limit(mode):
    if st.session_state.daily_usage['count'] >= FREE_DAILY_LIMIT:
        return False, "Daily limit reached! Upgrade to Premium for unlimited solves."
    
    if mode in PREMIUM_MODES and st.session_state.daily_usage['count'] >= 3:
        return False, "Free tier limited to 3 Accurate/Ultimate solves per day. Upgrade to Premium!"
    
    return True, None


# ============== ANALYTICS ==============

def log_solve(mode, cards_input, result, error=None):
    """Log solve for analytics"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'cards': cards_input,
        'success': error is None,
        'error': str(error) if error else None,
        'ev': result.ev if result else None,
        'time': result.computation_time if result else None
    }
    
    st.session_state.history.append(log_entry)
    st.session_state.solve_count += 1
    
    # Save to file
    log_dir = Path("../../data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"web_app_{datetime.now().strftime('%Y%m%d')}.jsonl"
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Log error: {e}")


# ============== HEADER ==============

st.markdown('<h1 class="main-header">🃏 Mậu Binh AI Solver Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    '⚡ Professional AI-powered Chinese Poker solver with beautiful card visualization<br>'
    '🤖 Game Theory • Deep Learning • Monte Carlo Simulation'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============== SIDEBAR ==============

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/playing-cards.png", width=80)
    st.markdown("## ⚙️ Settings")
    
    # Mode selection
    mode_options = {
        "⚡ Fast": ("fast", "< 1 second", "Good for quick decisions"),
        "⚖️ Balanced": ("balanced", "2-5 seconds", "✨ Recommended for most cases"),
        "🎯 Accurate": ("accurate", "10-20 seconds", "Maximum accuracy"),
        "🤖 ML Agent": ("ml_only", "2-3 seconds", "AI Deep Learning model"),
        "🚀 Ultimate": ("ultimate", "30-60 seconds", "🔥 Best possible solution")
    }
    
    selected_mode = st.selectbox(
        "Solver Mode",
        list(mode_options.keys()),
        index=1,
        help="Choose speed vs accuracy tradeoff"
    )
    
    mode_value, mode_time, mode_desc = mode_options[selected_mode]
    
    st.info(f"⏱️ **{mode_time}**\n\n{mode_desc}")
    
    st.markdown("---")

    # ML Status
    st.markdown("## 🤖 AI Status")

    # Check if ML is available
    try:
        from ultimate_solver import ML_AVAILABLE
        
        if ML_AVAILABLE:
            st.success("✅ AI Model: Online")
            st.caption("Deep Q-Network trained on 100,000+ hands")
        else:
            st.warning("⚠️ AI Model: Offline")
            st.caption("Using traditional algorithms only")
    except Exception:
        st.info("ℹ️ AI Status: Unknown")
    
    # Visual Cards Status
    if VISUAL_CARDS_AVAILABLE:
        st.success("✅ Visual Cards: Enabled")
    else:
        st.warning("⚠️ Visual Cards: Disabled")
    
    # Stats
    st.markdown("## 📊 Statistics")
    st.metric("Total Solves", st.session_state.solve_count)
    st.metric("Today's Usage", f"{st.session_state.daily_usage['count']}/{FREE_DAILY_LIMIT}")
    
    if st.session_state.history:
        recent = st.session_state.history[-10:]
        success_rate = sum(1 for x in recent if x['success']) / len(recent) * 100
        st.metric("Success Rate", f"{success_rate:.0f}%")
    
    st.markdown("---")
    
    # Help
    with st.expander("❓ How to use"):
        st.markdown("""
        **Quick Start:**
        1. Enter your 13 cards (or use examples)
        2. Choose solver mode
        3. Click **SOLVE** 🚀
        4. Get optimal arrangement with beautiful card display!
        
        **Card Format:**
        - `AS` = Ace of Spades ♠
        - `KH` = King of Hearts ♥
        - `QD` = Queen of Diamonds ♦
        - `JC` = Jack of Clubs ♣
        - `10S` = Ten of Spades ♠
        
        **Example:** `AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S`
        """)
    
    with st.expander("🎯 Understanding Results"):
        st.markdown("""
        **Expected Value (EV):**
        - Average profit per hand
        - Positive = profitable
        - Higher = better
        
        **Scoop Chance:**
        - Probability to win all 3 chi
        - 30%+ = excellent hand
        
        **Win 2/3:**
        - Probability to win at least 2 chi
        - 60%+ = strong hand
        """)


# ============== MAIN CONTENT ==============

# Import image input component với error handling chi tiết
IMAGE_INPUT_AVAILABLE = False
image_input_component = None

try:
    import sys
    import os
    
    # Đảm bảo path đúng
    current_dir = os.path.dirname(os.path.abspath(__file__))
    components_dir = os.path.join(current_dir, 'components')
    
    if components_dir not in sys.path:
        sys.path.insert(0, components_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from components.image_input import image_input_component
    IMAGE_INPUT_AVAILABLE = True
    
except Exception as e:
    print(f"⚠️ Image input component not available: {e}")
    IMAGE_INPUT_AVAILABLE = False

# Import card picker
PICKER_AVAILABLE = False
interactive_card_picker = None

try:
    from card_picker import interactive_card_picker, quick_select_buttons
    PICKER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Card picker not available: {e}")
    PICKER_AVAILABLE = False

# Example hands
with st.expander("📝 Example Hands (Click to use)", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    examples = {
        "🔥 Premium": "AS AH AD KC KD QS QH JD 10C 9S 8H 7D 6C",
        "⚖️ Balanced": "KS QH JD 10C 9S 8H 7D 6C 5S 4H 3D 2C AS",
        "⚠️ Weak": "9S 8H 7D 6C 5S 4H 3D 2C KS QH JD 10C 9H"
    }
    
    for i, (name, cards) in enumerate(examples.items()):
        col = [col1, col2, col3][i]
        with col:
            if st.button(name, key=f"ex_{i}", use_container_width=True):
                st.session_state.card_input = cards
                st.session_state.cards_ready_from_picker = False
                st.session_state.cards_from_image = False
                rerun()


# ===== INPUT SECTION WITH TABS =====
st.markdown("## 📥 Input Your Cards")

# Initialize flags
if 'cards_ready_from_picker' not in st.session_state:
    st.session_state.cards_ready_from_picker = False

if 'cards_from_image' not in st.session_state:
    st.session_state.cards_from_image = False

# Initialize solve button state
solve_button = False
card_input = st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S")

# ===== CREATE TABS BASED ON AVAILABLE COMPONENTS =====
if IMAGE_INPUT_AVAILABLE and PICKER_AVAILABLE:
    # All 3 tabs
    tab1, tab2, tab3 = st.tabs(["✍️ Type Cards", "🎴 Click to Pick", "📷 From Image"])
    
    # TAB 1: Type Cards
    with tab1:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            typed_input = st.text_input(
                "Enter 13 cards (space-separated)",
                value=st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"),
                help="Format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
                key="main_input",
                placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S..."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 SOLVE", type="primary", use_container_width=True, key="solve_type"):
                card_input = typed_input
                solve_button = True
                st.session_state.cards_ready_from_picker = False
                st.session_state.cards_from_image = False
    
    # TAB 2: Click to Pick
    with tab2:
        if st.session_state.cards_ready_from_picker and st.session_state.get('card_input'):
            st.success(f"✅ **Cards selected!** Ready to solve.")
            st.code(st.session_state.card_input, language=None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 SOLVE NOW", type="primary", use_container_width=True, key="solve_pick"):
                    card_input = st.session_state.card_input
                    solve_button = True
            
            with col2:
                if st.button("🔄 Pick Different Cards", use_container_width=True, key="pick_different"):
                    st.session_state.cards_ready_from_picker = False
                    st.session_state.picker_selected_cards = []
                    rerun()
        else:
            picked_result = interactive_card_picker()
            
            if picked_result:
                st.session_state.card_input = picked_result
                st.session_state.cards_ready_from_picker = True
                st.session_state.cards_from_image = False
                rerun()
    
    # TAB 3: From Image (NEW!)
    with tab3:
        if st.session_state.get('cards_from_image') and st.session_state.get('card_input'):
            st.success(f"✅ **Cards detected from image!**")
            st.code(st.session_state.card_input, language=None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 SOLVE NOW", type="primary", use_container_width=True, key="solve_image"):
                    card_input = st.session_state.card_input
                    solve_button = True
            
            with col2:
                if st.button("📷 Detect Different Image", use_container_width=True, key="new_image"):
                    st.session_state.cards_from_image = False
                    rerun()
        else:
            detected_cards = image_input_component(key="main_image_input")
            
            if detected_cards:
                st.session_state.card_input = detected_cards
                st.session_state.cards_from_image = True
                st.session_state.cards_ready_from_picker = False
                rerun()

elif IMAGE_INPUT_AVAILABLE and not PICKER_AVAILABLE:
    # 2 tabs: Type + Image
    tab1, tab3 = st.tabs(["✍️ Type Cards", "📷 From Image"])
    
    with tab1:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            typed_input = st.text_input(
                "Enter 13 cards (space-separated)",
                value=st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"),
                help="Format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
                key="main_input",
                placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S..."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 SOLVE", type="primary", use_container_width=True, key="solve_type"):
                card_input = typed_input
                solve_button = True
                st.session_state.cards_from_image = False
    
    with tab3:
        if st.session_state.get('cards_from_image') and st.session_state.get('card_input'):
            st.success(f"✅ **Cards detected from image!**")
            st.code(st.session_state.card_input, language=None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 SOLVE NOW", type="primary", use_container_width=True, key="solve_image"):
                    card_input = st.session_state.card_input
                    solve_button = True
            
            with col2:
                if st.button("📷 Detect Different Image", use_container_width=True, key="new_image"):
                    st.session_state.cards_from_image = False
                    rerun()
        else:
            detected_cards = image_input_component(key="main_image_input")
            
            if detected_cards:
                st.session_state.card_input = detected_cards
                st.session_state.cards_from_image = True
                rerun()

elif PICKER_AVAILABLE and not IMAGE_INPUT_AVAILABLE:
    # 2 tabs: Type + Pick
    tab1, tab2 = st.tabs(["✍️ Type Cards", "🎴 Click to Pick"])
    
    with tab1:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            typed_input = st.text_input(
                "Enter 13 cards (space-separated)",
                value=st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"),
                help="Format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
                key="main_input",
                placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S..."
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 SOLVE", type="primary", use_container_width=True, key="solve_type"):
                card_input = typed_input
                solve_button = True
                st.session_state.cards_ready_from_picker = False
    
    with tab2:
        if st.session_state.cards_ready_from_picker and st.session_state.get('card_input'):
            st.success(f"✅ **Cards selected!** Ready to solve.")
            st.code(st.session_state.card_input, language=None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 SOLVE NOW", type="primary", use_container_width=True, key="solve_pick"):
                    card_input = st.session_state.card_input
                    solve_button = True
            
            with col2:
                if st.button("🔄 Pick Different Cards", use_container_width=True, key="pick_different"):
                    st.session_state.cards_ready_from_picker = False
                    st.session_state.picker_selected_cards = []
                    rerun()
        else:
            picked_result = interactive_card_picker()
            
            if picked_result:
                st.session_state.card_input = picked_result
                st.session_state.cards_ready_from_picker = True
                rerun()

else:
    # Fallback: only text input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        card_input = st.text_input(
            "Enter 13 cards (space-separated)",
            value=st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"),
            help="Format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
            key="main_input_fallback",
            placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        solve_button = st.button("🚀 SOLVE NOW", type="primary", use_container_width=True)


# ============== SOLVE ==============

if solve_button and card_input:
    # Usage limit check
    can_solve, error_msg = check_usage_limit(mode_value)
    if not can_solve:
        st.error(f"❌ {error_msg}")
        st.info("💎 **Upgrade to Premium:** $9.99/month for unlimited access!")
        st.stop()

    error = None
    result = None
    
    try:
        # Validate input
        cards = Deck.parse_hand(card_input)
        
        if len(cards) != 13:
            error = f"Need exactly 13 cards, got {len(cards)}"
            st.error(f"❌ {error}")
        else:
            # Show input cards preview (if visual available)
            if VISUAL_CARDS_AVAILABLE:
                st.markdown("### 🎴 Your Input Cards")
                cards_strs = [str(c) for c in cards]
                st.markdown(
                    render_input_cards_preview(cards_strs),
                    unsafe_allow_html=True
                )
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 Initializing AI solver...")
            progress_bar.progress(20)
            time.sleep(0.1)
            
            # Solve
            solver_mode = SolverMode(mode_value)
            solver = UltimateSolver(cards, mode=solver_mode, verbose=False)
            
            status_text.text(f"🧠 Computing with {selected_mode} mode...")
            progress_bar.progress(50)
            
            result = solver.solve()
            
            progress_bar.progress(100)
            status_text.text("✅ Solution found!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            # Log
            log_solve(mode_value, card_input, result, error)
            
            # Display results
            st.success("✅ **OPTIMAL SOLUTION FOUND!**")
            
            st.markdown("---")
            
            # ===== ARRANGEMENT WITH VISUAL CARDS =====
            st.markdown("## 🎯 Optimal Arrangement")
            
            # Prepare evaluations
            back_eval = HandEvaluator.evaluate(result.back)
            middle_eval = HandEvaluator.evaluate(result.middle)
            front_eval = HandEvaluator.evaluate(result.front)
            
            # Convert cards to strings
            back_strs = [str(c) for c in result.back]
            middle_strs = [str(c) for c in result.middle]
            front_strs = [str(c) for c in result.front]
            
            # Render beautiful cards (or fallback to text)
            if VISUAL_CARDS_AVAILABLE:
                st.markdown(
                    render_comparison_cards(
                        back_strs,
                        middle_strs,
                        front_strs,
                        (str(back_eval), str(middle_eval), str(front_eval))
                    ),
                    unsafe_allow_html=True
                )
            else:
                # Fallback: Text display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🔵 Chi 1 (Back)")
                    st.code(Deck.cards_to_string(result.back))
                    st.info(f"**{back_eval}**")
                
                with col2:
                    st.markdown("### 🟢 Chi 2 (Middle)")
                    st.code(Deck.cards_to_string(result.middle))
                    st.info(f"**{middle_eval}**")
                
                with col3:
                    st.markdown("### 🟡 Chi cuối (Front)")
                    st.code(Deck.cards_to_string(result.front))
                    st.info(f"**{front_eval}**")
            
            st.markdown("---")
            
            # ===== METRICS =====
            st.markdown("## 📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "Expected Value",
                f"{result.ev:+.2f}",
                help="Average profit per hand"
            )
            
            col2.metric(
                "Bonus Points",
                f"+{result.bonus}",
                help="Special combination bonuses"
            )
            
            col3.metric(
                "Scoop Chance",
                f"{result.p_scoop*100:.1f}%",
                help="Probability to win all 3 chi",
                delta=f"{(result.p_scoop - 0.3)*100:+.1f}%" if result.p_scoop > 0.3 else None
            )
            
            col4.metric(
                "Win 2/3 Chi",
                f"{result.p_win_2_of_3*100:.1f}%",
                help="Probability to win at least 2 chi",
                delta=f"{(result.p_win_2_of_3 - 0.6)*100:+.1f}%" if result.p_win_2_of_3 > 0.6 else None
            )
            
            st.markdown("---")
            
            # ===== WIN PROBABILITY BREAKDOWN =====
            st.markdown("## 📈 Win Probability Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔵 Back Win Rate**")
                back_wr = result.p_win_back * 100
                st.progress(result.p_win_back)
                st.metric("Back", f"{back_wr:.1f}%", label_visibility="collapsed")
            
            with col2:
                st.markdown("**🟢 Middle Win Rate**")
                mid_wr = result.p_win_middle * 100
                st.progress(result.p_win_middle)
                st.metric("Middle", f"{mid_wr:.1f}%", label_visibility="collapsed")
            
            with col3:
                st.markdown("**🟡 Front Win Rate**")
                front_wr = result.p_win_front * 100
                st.progress(result.p_win_front)
                st.metric("Front", f"{front_wr:.1f}%", label_visibility="collapsed")
            
            # Overall assessment
            st.markdown("### 🎯 Overall Assessment")
            
            # Weighted score (front most important in Mau Binh)
            total_score = (front_wr * 0.4 + mid_wr * 0.3 + back_wr * 0.3)
            
            if total_score >= 70:
                st.success(f"🔥 **EXCELLENT HAND!** (Score: {total_score:.0f}/100)")
                st.write("✨ This is a very strong arrangement. Play aggressively and expect to win!")
            elif total_score >= 55:
                st.info(f"✅ **GOOD HAND** (Score: {total_score:.0f}/100)")
                st.write("👍 Above average hand. Standard confident play recommended.")
            elif total_score >= 40:
                st.warning(f"⚠️ **AVERAGE HAND** (Score: {total_score:.0f}/100)")
                st.write("🤔 Moderate hand. Defensive play may be wise against strong opponents.")
            else:
                st.error(f"❌ **WEAK HAND** (Score: {total_score:.0f}/100)")
                st.write("🛡️ Difficult hand. Focus on minimizing losses. Consider folding if game rules allow.")
            
            st.markdown("---")
            
            # ===== DETAILS =====
            with st.expander("📈 Detailed Analysis & Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎲 Win Probabilities:**")
                    st.write(f"- Front:  {result.p_win_front*100:.1f}%")
                    st.write(f"- Middle: {result.p_win_middle*100:.1f}%")
                    st.write(f"- Back:   {result.p_win_back*100:.1f}%")
                    st.write(f"- Scoop:  {result.p_scoop*100:.1f}%")
                    st.write(f"- Win 2+: {result.p_win_2_of_3*100:.1f}%")
                
                with col2:
                    st.markdown("**⚙️ Computation Info:**")
                    st.write(f"- Mode: **{result.mode.value.upper()}**")
                    st.write(f"- Time: {result.computation_time:.2f}s")
                    st.write(f"- Evaluated: {result.num_arrangements_evaluated:,} arrangements")
                    st.write(f"- Bonus: +{result.bonus} points")
                    st.write(f"- Expected Value: {result.ev:+.2f}")
            
            
            
            # ===== RECOMMENDATIONS =====
            st.markdown("## 💡 Strategic Recommendations")
            
            recommendations = []
            
            if result.p_scoop > 0.3:
                recommendations.append("🎯 **High scoop chance!** (30%+) Play aggressively and bet confidently.")
            
            if result.ev > 1.5:
                recommendations.append("💰 **Very positive EV!** (>1.5) This is a highly profitable hand.")
            elif result.ev > 0.5:
                recommendations.append("💵 **Positive EV** (>0.5) This hand has profit potential.")
            
            if result.bonus >= 6:
                recommendations.append(f"⭐ **Big bonus hand!** You have {result.bonus} bonus points - maximize value!")
            elif result.bonus >= 3:
                recommendations.append(f"✨ **Bonus hand!** You have {result.bonus} bonus points.")
            
            if result.p_win_front > 0.8:
                recommendations.append("💪 **Dominant front!** (80%+ win rate) You have a major advantage in chi cuối.")
            
            if result.p_win_2_of_3 > 0.7:
                recommendations.append("✅ **High overall win probability!** (70%+) Confident play strongly recommended.")
            
            if result.ev < -0.5:
                recommendations.append("⚠️ **Negative EV hand** - Play defensively and minimize exposure.")
            
            if recommendations:
                for rec in recommendations:
                    st.success(rec)
            else:
                st.info("🤔 **Standard hand.** Play according to basic strategy. No special adjustments needed.")
            
            # Show ML-specific info when ML-only mode was used
            if result.mode.value == "ml_only":
                st.markdown("---")
                st.info("""
                🤖 **AI Deep Learning Mode Active**
                
                This solution was generated using our Deep Q-Network (DQN) trained on 100,000+ expert hands.
                The AI model learned optimal strategies through reinforcement learning and expert gameplay patterns.
                
                **Model Details:**
                - Architecture: Deep Q-Network with experience replay
                - Training: 100,000 hands from expert players
                - Accuracy: 85%+ match rate with expert solutions
                """)
            
            st.markdown("---")
            
            # ===== EXPORT & SHARE =====
            st.markdown("### 📤 Export & Share Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Screenshot-friendly format
                export_text = f"""
🃏 Mau Binh AI Solver Pro - Result
{'='*60}

INPUT:
Cards: {card_input}

OPTIMAL ARRANGEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔵 Chi 1 (Back - 5 cards):
   {Deck.cards_to_string(result.back)}
   {back_eval}

🟢 Chi 2 (Middle - 5 cards):
   {Deck.cards_to_string(result.middle)}
   {middle_eval}

🟡 Chi cuối (Front - 3 cards):
   {Deck.cards_to_string(result.front)}
   {front_eval}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE METRICS:
├─ Expected Value (EV):    {result.ev:+.2f}
├─ Bonus Points:           +{result.bonus}
├─ Scoop Chance:           {result.p_scoop*100:.1f}%
├─ Win 2/3 Chi:            {result.p_win_2_of_3*100:.1f}%
│
├─ Front Win Rate:         {result.p_win_front*100:.1f}%
├─ Middle Win Rate:        {result.p_win_middle*100:.1f}%
└─ Back Win Rate:          {result.p_win_back*100:.1f}%

COMPUTATION:
├─ Mode:                   {selected_mode}
├─ Time:                   {result.computation_time:.2f}s
└─ Arrangements Evaluated: {result.num_arrangements_evaluated:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by Mau Binh AI Solver Pro
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="📥 Download as TXT",
                    data=export_text,
                    file_name=f"maubinh_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Copy to clipboard
                if st.button("📋 Copy Result", use_container_width=True, key="copy_result"):
                    st.code(export_text, language=None)
                    st.success("✅ Result displayed above - Ctrl+A, Ctrl+C to copy!")
            
            with col3:
                # Share on social
                share_text = f"I got EV {result.ev:+.2f} with {result.p_scoop*100:.0f}% scoop chance using Mau Binh AI Solver Pro! 🃏🔥"
                
                twitter_url = f"https://twitter.com/intent/tweet?text={share_text.replace(' ', '%20')}"
                
                st.markdown(
                    f'<a href="{twitter_url}" target="_blank" style="text-decoration: none;"><button style="width:100%; padding:0.6rem; background:#1DA1F2; color:white; border:none; border-radius:0.5rem; cursor:pointer; font-weight:700; font-size:1rem;">🐦 Tweet Result</button></a>',
                    unsafe_allow_html=True
                )
            
            # After successful solve, increment usage
            st.session_state.daily_usage['count'] += 1
    
    except Exception as e:
        error = str(e)
        st.error(f"❌ **Error:** {error}")
        
        with st.expander("🔍 Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())
        
        log_solve(mode_value, card_input, None, error)


# ============== FOOTER ==============

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [📖 How to Play Mau Binh](https://en.wikipedia.org/wiki/Chinese_poker)
    - [🎯 Strategy Guide](#)
    - [🤖 About the AI](#)
    - [📊 Performance Metrics](#)
    """)

with col2:
    st.markdown("### 🔗 Connect")
    st.markdown("""
    - [💻 GitHub Repository](#)
    - [🐦 Twitter/X](#)
    - [💬 Discord Community](#)
    - [📧 Contact Us](#)
    """)

with col3:
    st.markdown("### ⚡ Technology")
    st.markdown("""
    - Deep Q-Learning (DQN)
    - Game Theory Optimal
    - Monte Carlo Simulation
    - Streamlit Framework
    """)

st.markdown(
    '<p style="text-align: center; color: #999; margin-top: 2rem; font-size: 0.9rem;">'
    'Made with ❤️ using Python, PyTorch & Streamlit<br>'
    '© 2024 Mau Binh AI Solver Pro • All rights reserved'
    '</p>',
    unsafe_allow_html=True
)

# Debug info (only in development)
if os.getenv('DEBUG', 'false').lower() == 'true':
    with st.expander("🔧 Debug Info"):
        st.write("**Session State:**")
        st.json({
            'solve_count': st.session_state.solve_count,
            'daily_usage': str(st.session_state.daily_usage),
            'history_length': len(st.session_state.history),
            'cards_ready_from_picker': st.session_state.get('cards_ready_from_picker', False)
        })
        
        st.write("**Environment:**")
        st.write(f"- Visual Cards: {VISUAL_CARDS_AVAILABLE}")
        st.write(f"- Picker Available: {PICKER_AVAILABLE}")
        st.write(f"- Python: {sys.version}")
        st.write(f"- Streamlit: {st.__version__}")