"""
Professional Streamlit Web App for Mau Binh Solver
Production-ready with analytics and monetization
"""
import streamlit as st
import sys
import os
from datetime import datetime
import time
import json
from pathlib import Path

# FIX: Add correct paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src/

sys.path.insert(0, os.path.join(parent_dir, 'core'))
sys.path.insert(0, os.path.join(parent_dir, 'engines'))
sys.path.insert(0, parent_dir)

from card import Deck
from ultimate_solver import UltimateSolver, SolverMode
from evaluator import HandEvaluator


# ============== CONFIG ==============

st.set_page_config(
    page_title="Mau Binh AI Solver Pro",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
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
FREE_DAILY_LIMIT = 10
PREMIUM_MODES = ['accurate', 'ultimate']

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
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============== HEADER ==============

st.markdown('<h1 class="main-header">🃏 Mậu Binh AI Solver Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 1.1rem;">'
    'Professional AI-powered Chinese Poker solver using Game Theory & Deep Learning'
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
        "⚖️ Balanced": ("balanced", "2-5 seconds", "Recommended for most cases"),
        "🎯 Accurate": ("accurate", "10-20 seconds", "Maximum accuracy"),
        "🤖 ML Agent": ("ml_only", "2-3 seconds", "AI Deep Learning model"),
        "🚀 Ultimate": ("ultimate", "30-60 seconds", "Best possible solution")
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
            st.caption("Deep Q-Network trained on 10,000+ hands")
        else:
            st.warning("⚠️ AI Model: Offline")
            st.caption("Using traditional algorithms only")
    except Exception:
        st.info("ℹ️ AI Status: Unknown")
    
    # Stats
    st.markdown("## 📊 Statistics")
    st.metric("Total Solves", st.session_state.solve_count)
    
    if st.session_state.history:
        recent = st.session_state.history[-10:]
        success_rate = sum(1 for x in recent if x['success']) / len(recent) * 100
        st.metric("Success Rate", f"{success_rate:.0f}%")
    
    st.markdown("---")
    
    # Pricing info
    with st.expander("💰 Pricing"):
        st.markdown("""
        **Free Tier:**
        - 10 solves/day
        - Fast mode only
        
        **Premium ($9.99/mo):**
        - Unlimited solves
        - All modes
        - No ads
        
        **Pro ($49.99/mo):**
        - Everything + API
        - Advanced analytics
        - Priority support
        """)
    
    # Help
    with st.expander("❓ How to use"):
        st.markdown("""
        1. Enter your 13 cards
        2. Choose solver mode
        3. Click SOLVE
        4. Get optimal arrangement!
        
        **Card format:**
        - AS = Ace of Spades ♠
        - KH = King of Hearts ♥
        - QD = Queen of Diamonds ♦
        - JC = Jack of Clubs ♣
        """)

    st.markdown("---")
    st.markdown("## 🎓 Learn")

    with st.expander("📖 Common Scenarios"):
        scenarios = {
            "🔥 Premium Hand": {
                "cards": "AS AH AD KC KD QS QH JD 10C 9S 8H 7D 6C",
                "tip": "Three Aces! Look for trip in front (+6 bonus)"
            },
            "🎯 Straight Draw": {
                "cards": "9S 8H 7D 6C 5S AH KD QC JH 10D 4S 3H 2C",
                "tip": "Multiple straight possibilities!"
            },
            "💎 Flush Draw": {
                "cards": "AS KS QS JS 9S 7H 6H 4H 3H 2D 8C 5D 4C",
                "tip": "Spade flush available"
            },
            "⚠️ Weak Hand": {
                "cards": "KS QH JD 9C 7S 6H 5D 4C 3H 2S 10D 8C 4H",
                "tip": "No pairs - maximize high cards in front"
            }
        }
        
        for name, data in scenarios.items():
            if st.button(name, key=f"scenario_{name}"):
                st.session_state.card_input = data['cards']
                st.info(f"💡 {data['tip']}")
                st.rerun()


# ============== MAIN CONTENT ==============

# Example hands
with st.expander("📝 Example Hands (Click to use)"):
    col1, col2, col3 = st.columns(3)
    
    examples = {
        "Strong Hand": "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S",
        "Medium Hand": "KS QH JD 10C 9S 8H 7D 6C 5S 4H 3D 2C AS",
        "Weak Hand": "9S 8H 7D 6C 5S 4H 3D 2C KS QH JD 10C 9H"
    }
    
    for i, (name, cards) in enumerate(examples.items()):
        col = [col1, col2, col3][i]
        with col:
            if st.button(name, key=f"ex_{i}"):
                st.session_state.card_input = cards


# Input section
st.markdown("## 📥 Input Your Cards")

col1, col2 = st.columns([3, 1])

with col1:
    card_input = st.text_input(
        "Enter 13 cards (space-separated)",
        value=st.session_state.get('card_input', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"),
        help="Format: AS KH QD JC 10S...",
        key="main_input"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    solve_button = st.button("🚀 SOLVE", type="primary", use_container_width=True)


# ============== SOLVE ==============

if solve_button:
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
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 Initializing solver...")
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
            st.success("✅ **SOLUTION FOUND!**")
            
            # ===== ARRANGEMENT =====
            st.markdown("## 🎯 Optimal Arrangement")
            
            col1, col2, col3 = st.columns(3)
            
            cards_style = "background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; text-align: center; font-size: 1.2rem; font-family: monospace;"
            
            with col1:
                st.markdown("### 🔵 Chi 1 (Back)")
                st.markdown(
                    f'<div style="{cards_style}">{Deck.cards_to_string(result.back)}</div>',
                    unsafe_allow_html=True
                )
                back_eval = HandEvaluator.evaluate(result.back)
                st.info(f"**{back_eval}**")
            
            with col2:
                st.markdown("### 🟢 Chi 2 (Middle)")
                st.markdown(
                    f'<div style="{cards_style}">{Deck.cards_to_string(result.middle)}</div>',
                    unsafe_allow_html=True
                )
                middle_eval = HandEvaluator.evaluate(result.middle)
                st.info(f"**{middle_eval}**")
            
            with col3:
                st.markdown("### 🟡 Chi cuối (Front)")
                st.markdown(
                    f'<div style="{cards_style}">{Deck.cards_to_string(result.front)}</div>',
                    unsafe_allow_html=True
                )
                front_eval = HandEvaluator.evaluate(result.front)
                st.info(f"**{front_eval}**")
            
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
                help="Probability to win all 3 chi"
            )
            
            col4.metric(
                "Win 2/3 Chi",
                f"{result.p_win_2_of_3*100:.1f}%",
                help="Probability to win at least 2 chi"
            )
            
            # ===== DETAILS =====
            with st.expander("📈 Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Win Probabilities:**")
                    st.write(f"- Front:  {result.p_win_front*100:.1f}%")
                    st.write(f"- Middle: {result.p_win_middle*100:.1f}%")
                    st.write(f"- Back:   {result.p_win_back*100:.1f}%")
                
                with col2:
                    st.markdown("**Computation Info:**")
                    st.write(f"- Mode: {result.mode.value}")
                    st.write(f"- Time: {result.computation_time:.2f}s")
                    st.write(f"- Evaluated: {result.num_arrangements_evaluated} arrangements")
            
            # ===== RECOMMENDATIONS =====
            st.markdown("## 💡 Strategic Recommendations")
            
            recommendations = []
            
            if result.p_scoop > 0.3:
                recommendations.append("🎯 **High scoop chance!** Play aggressively.")
            
            if result.ev > 1.0:
                recommendations.append("💰 **Positive EV!** This is a profitable hand.")
            
            if result.bonus >= 4:
                recommendations.append(f"⭐ **Bonus hand!** You have {result.bonus} bonus points.")
            
            if result.p_win_front > 0.8:
                recommendations.append("💪 **Strong front!** You have a big advantage.")
            
            if result.p_win_2_of_3 > 0.7:
                recommendations.append("✅ **High win probability!** Confident play recommended.")
            
            if recommendations:
                for rec in recommendations:
                    st.success(rec)
            else:
                st.info("🤔 Standard hand. Play according to basic strategy.")

            # Show ML-specific info when ML-only mode was used
            if result.mode.value == "ml_only":
                st.info("""
                🤖 **AI Mode Active**
                
                This solution was generated using our Deep Q-Network trained on expert games.
                The AI model learned optimal strategies through 10,000+ training hands.
                """)

                # Show ML confidence
                if hasattr(result, 'ml_confidence'):
                    st.metric("AI Confidence", f"{result.ml_confidence*100:.0f}%")

            # ===== COMPARE WITH OTHER MODES =====
            with st.expander("🔀 Compare with other solver modes"):
                st.info("💡 Try different modes to see alternative solutions!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Try Fast Mode", use_container_width=True):
                        st.session_state.card_input = card_input
                        st.rerun()
                
                with col2:
                    if st.button("Try Ultimate Mode", use_container_width=True):
                        st.session_state.card_input = card_input
                        st.rerun()

            # ===== SAVE TO HISTORY =====
            if 'saved_solutions' not in st.session_state:
                st.session_state.saved_solutions = []

            if st.button("💾 Save This Solution"):
                st.session_state.saved_solutions.append({
                    'cards': card_input,
                    'mode': selected_mode,
                    'arrangement': (result.back, result.middle, result.front),
                    'ev': result.ev,
                    'timestamp': datetime.now()
                })
                st.success("✅ Solution saved!")

            # Show saved solutions
            if st.session_state.saved_solutions:
                with st.expander(f"📚 Saved Solutions ({len(st.session_state.saved_solutions)})"):
                    for i, saved in enumerate(reversed(st.session_state.saved_solutions[-5:])):
                        st.text(f"{i+1}. {saved['cards']} - EV: {saved['ev']:+.2f}")

            # ===== SHARE & COPY =====
            st.markdown("---")
            st.markdown("### 📢 Share Your Results")

            col1, col2, col3 = st.columns(3)

            result_text = f"I got EV +{result.ev:.2f} with {result.p_scoop*100:.0f}% scoop chance using Mau Binh AI Solver!"

            with col1:
                twitter_url = f"https://twitter.com/intent/tweet?text={result_text}"
                st.markdown(f"[🐦 Tweet]({twitter_url})")

            with col2:
                st.markdown("📋 Copy Result")
                if st.button("Copy", key="copy_btn"):
                    st.code(f"""
Cards: {card_input}
Back:   {Deck.cards_to_string(result.back)}
Middle: {Deck.cards_to_string(result.middle)}
Front:  {Deck.cards_to_string(result.front)}
EV: {result.ev:+.2f}
                    """)

            with col3:
                st.markdown("⭐ [Rate on GitHub](https://github.com/YOUR_REPO)")

            # After successful solve, increment usage
            st.session_state.daily_usage['count'] += 1
    
    except Exception as e:
        error = str(e)
        st.error(f"❌ **Error:** {error}")
        log_solve(mode_value, card_input, None, error)


# ============== FOOTER ==============

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [How to Play](https://example.com)
    - [Strategy Guide](https://example.com)
    - [API Docs](https://example.com)
    """)

with col2:
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [GitHub](https://github.com)
    - [Twitter](https://twitter.com)
    - [Discord](https://discord.com)
    """)

with col3:
    st.markdown("### ⚡ Powered By")
    st.markdown("""
    - Deep Q-Learning
    - Game Theory
    - Monte Carlo Simulation
    """)

st.markdown(
    '<p style="text-align: center; color: #999; margin-top: 2rem;">'
    'Made with ❤️ | © 2024 Mau Binh AI Solver Pro'
    '</p>',
    unsafe_allow_html=True
)