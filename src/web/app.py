"""
Professional Streamlit Web App for Mau Binh Solver V2.2
Production-ready with ML Agent V2, Hybrid Mode, Analytics, Beautiful Card Display
✅ FIXED: All use_container_width warnings
"""
import streamlit as st
import sys
import os
from datetime import datetime
import time
import json
from pathlib import Path
import base64
from io import BytesIO


# ===== COMPATIBILITY FIX =====
def rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.warning("Please refresh the page manually")


# ===== PATH SETUP =====
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, os.path.join(parent_dir, 'core'))
sys.path.insert(0, os.path.join(parent_dir, 'engines'))
sys.path.insert(0, os.path.join(parent_dir, 'ml'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(current_dir, 'components'))


# ===== CORE IMPORTS =====
from card import Deck
from evaluator import HandEvaluator
from ultimate_solver import (
    UltimateSolver, SolverMode,
    ML_AVAILABLE, REWARD_CALC_AVAILABLE,
    get_available_modes, get_ml_status
)


# ===== VISUAL COMPONENTS =====
VISUAL_CARDS_AVAILABLE = False
try:
    from card_renderer import render_comparison_cards, render_input_cards_preview, get_card_html
    VISUAL_CARDS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Visual card components not found: {e}")


# ===== IMAGE INPUT =====
IMAGE_INPUT_AVAILABLE = False
_image_import_error = None

try:
    from components.image_input import image_input_component
    IMAGE_INPUT_AVAILABLE = True
except ImportError:
    pass

if not IMAGE_INPUT_AVAILABLE:
    try:
        from image_input import image_input_component
        IMAGE_INPUT_AVAILABLE = True
    except ImportError:
        pass

if not IMAGE_INPUT_AVAILABLE:
    try:
        import importlib.util
        image_input_path = os.path.join(current_dir, 'components', 'image_input.py')
        if os.path.exists(image_input_path):
            spec = importlib.util.spec_from_file_location("image_input", image_input_path)
            image_input_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(image_input_mod)
            image_input_component = image_input_mod.image_input_component
            IMAGE_INPUT_AVAILABLE = True
    except Exception as e:
        _image_import_error = str(e)

if not IMAGE_INPUT_AVAILABLE:
    print(f"⚠️ Image input component not available. Error: {_image_import_error}")


# ===== CARD PICKER =====
PICKER_AVAILABLE = False
try:
    from card_picker import interactive_card_picker, quick_select_buttons
    PICKER_AVAILABLE = True
except ImportError:
    pass
if not PICKER_AVAILABLE:
    try:
        from components.card_picker import interactive_card_picker, quick_select_buttons
        PICKER_AVAILABLE = True
    except ImportError:
        pass


# ===== FALLBACK IMAGE INPUT =====
def fallback_image_input(key="fallback_image"):
    """Fallback khi image_input_component không available."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    ">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">📷 Upload or Paste Screenshot</h3>
        <p style="color: #666;">
            Upload ảnh bài hoặc chụp màn hình rồi paste (Ctrl+V)<br>
            <small>Hỗ trợ: PNG, JPG, JPEG</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload ảnh bài",
        type=['png', 'jpg', 'jpeg', 'webp'],
        key=f"{key}_uploader",
        help="Kéo thả ảnh hoặc click để chọn file.",
        label_visibility="collapsed",
    )

    use_camera = st.checkbox("📸 Dùng camera", key=f"{key}_camera_toggle")
    camera_image = None
    if use_camera:
        camera_image = st.camera_input("Chụp ảnh bài", key=f"{key}_camera")

    # Paste zone HTML
    st.components.v1.html("""
    <div id="paste-zone" class="paste-zone" tabindex="0" onclick="this.focus()"
         style="border: 2px dashed #aaa; border-radius: 12px; padding: 40px 20px;
                text-align: center; cursor: pointer; background: #fafafa; margin: 10px 0; outline: none;">
        <div style="font-size: 3rem; margin-bottom: 10px;">📋</div>
        <p><b>Click here</b>, then press <b>Ctrl+V</b> to paste screenshot</p>
        <p style="font-size: 0.85rem; color: #aaa; margin-top: 8px;">
            Chụp màn hình game → Ctrl+V paste vào đây
        </p>
        <div id="paste-preview" style="margin-top: 15px;"></div>
    </div>
    <script>
    const pasteZone = document.getElementById('paste-zone');
    const preview = document.getElementById('paste-preview');
    pasteZone.addEventListener('paste', function(e) {
        e.preventDefault();
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const reader = new FileReader();
                reader.onload = function(event) {
                    preview.innerHTML = '<img src="' + event.target.result +
                        '" style="max-width:100%;max-height:300px;border-radius:8px;margin-top:10px;">';
                    pasteZone.style.borderColor = '#11998e';
                    pasteZone.style.background = '#11998e11';
                };
                reader.readAsDataURL(blob);
                break;
            }
        }
    });
    pasteZone.addEventListener('focus', function() {
        pasteZone.style.borderColor = '#667eea';
        pasteZone.style.background = '#667eea11';
    });
    pasteZone.addEventListener('blur', function() {
        pasteZone.style.borderColor = '#aaa';
        pasteZone.style.background = '#fafafa';
    });
    </script>
    """, height=250)

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = uploaded_file
        st.image(uploaded_file, caption="📷 Ảnh đã upload", width="stretch")
    elif camera_image is not None:
        image_to_process = camera_image
        st.image(camera_image, caption="📸 Ảnh từ camera", width="stretch")

    if image_to_process is not None:
        detected = _try_detect_cards_from_image(image_to_process)
        if detected:
            return detected

    return None


def _try_detect_cards_from_image(image_data):
    """Thử detect cards từ image data."""
    try:
        from card_detector import CardDetector
        detector = CardDetector()

        from PIL import Image
        import numpy as np

        if hasattr(image_data, 'read'):
            img = Image.open(image_data)
        else:
            img = Image.open(BytesIO(image_data))

        img_array = np.array(img)
        cards = detector.detect(img_array)

        if cards and len(cards) == 13:
            card_str = " ".join(str(c) for c in cards)
            st.success(f"✅ Detected 13 cards: {card_str}")
            return card_str
        elif cards:
            card_str = " ".join(str(c) for c in cards)
            st.warning(f"⚠️ Detected {len(cards)}/13 cards: {card_str}")
            st.info("Hãy nhập thêm các lá còn thiếu")
            corrected = st.text_input("Sửa kết quả detect:", value=card_str, key="correct_detection")
            if st.button("✅ Confirm", key="confirm_detection"):
                return corrected
        else:
            st.warning("⚠️ Không detect được lá bài nào")

    except ImportError:
        st.info("🔍 ML Card Detector chưa cài. Dùng manual input:")
    except Exception as e:
        st.warning(f"⚠️ Detection error: {e}")

    st.markdown("---")
    st.markdown("**✍️ Nhập cards thủ công từ ảnh:**")
    manual_cards = st.text_input(
        "Nhập 13 lá bài nhìn thấy trong ảnh:",
        placeholder="AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
        key="manual_from_image"
    )
    if manual_cards and st.button("✅ Use these cards", key="use_manual_cards"):
        return manual_cards

    return None


# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Mau Binh AI Solver Pro",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .ml-badge-online {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .ml-badge-offline {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .stButton>button {
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
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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
    @media (max-width: 768px) {
        .main-header { font-size: 2.2rem !important; }
        .sub-header { font-size: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ============== SESSION STATE ==============
if 'solve_count' not in st.session_state:
    st.session_state.solve_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'card_input' not in st.session_state:
    st.session_state.card_input = "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"
if 'cards_ready_from_picker' not in st.session_state:
    st.session_state.cards_ready_from_picker = False
if 'cards_from_image' not in st.session_state:
    st.session_state.cards_from_image = False
if 'daily_usage' not in st.session_state:
    st.session_state.daily_usage = {'date': datetime.now().date(), 'count': 0}

if st.session_state.daily_usage['date'] != datetime.now().date():
    st.session_state.daily_usage = {'date': datetime.now().date(), 'count': 0}

FREE_DAILY_LIMIT = 99999


# ============== HELPER FUNCTIONS ==============
def check_usage_limit(mode):
    if st.session_state.daily_usage['count'] >= FREE_DAILY_LIMIT:
        return False, "Daily limit reached!"
    return True, None


def log_solve(mode, cards_input, result, error=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'cards': cards_input,
        'success': error is None,
        'error': str(error) if error else None,
        'ev': getattr(result, 'ev', 0) if result else None,
        'time': getattr(result, 'computation_time', 0) if result else None
    }
    st.session_state.history.append(log_entry)
    st.session_state.solve_count += 1

    try:
        log_dir = Path(parent_dir) / ".." / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"web_app_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass


# ============== HEADER ==============
st.markdown('<h1 class="main-header">🃏 Mậu Binh AI Solver Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    '⚡ Professional AI-powered Chinese Poker solver with ML Agent V2<br>'
    '🤖 Deep Learning • Game Theory • Monte Carlo • Hybrid AI'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")


# ============== SIDEBAR ==============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/playing-cards.png", width=80)
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🎯 Solver Mode")

    mode_options = {}

    if ML_AVAILABLE:
        mode_options["🔥 ML Hybrid (Best!)"] = ("ml_hybrid", "2-5s", "🔥 SmartSolver + AI scoring = Best!")
        mode_options["🤖 ML Agent (Best)"] = ("ml_best", "~50ms", "AI Ensemble")
        mode_options["⚡ ML Agent (Fast)"] = ("ml_fast", "~20ms", "AI DQN only")
        mode_options["🔍 ML + Beam Search"] = ("ml_beam", "2-3s", "AI + Search")
    elif REWARD_CALC_AVAILABLE:
        mode_options["🔥 ML Hybrid"] = ("ml_hybrid", "2-5s", "🔥 SmartSolver + Bonus scoring")

    mode_options["⚡ Fast"] = ("fast", "<1s", "Quick decisions")
    mode_options["⚖️ Balanced"] = ("balanced", "2-5s", "✨ Recommended")
    mode_options["🎯 Accurate"] = ("accurate", "10-20s", "Maximum accuracy")
    mode_options["🚀 Ultimate"] = ("ultimate", "30-60s", "Best traditional")

    default_idx = 0 if (ML_AVAILABLE or REWARD_CALC_AVAILABLE) else len(mode_options) - 3

    selected_mode = st.selectbox(
        "Choose Mode",
        list(mode_options.keys()),
        index=default_idx,
        help="ML Hybrid = SmartSolver + AI scoring (best results)"
    )

    mode_value, mode_time, mode_desc = mode_options[selected_mode]

    if mode_value.startswith('ml_'):
        st.info(f"⏱️ **{mode_time}**\n\n🤖 {mode_desc}")
    else:
        st.info(f"⏱️ **{mode_time}**\n\n{mode_desc}")

    st.markdown("---")

    st.markdown("### 🤖 AI Status")
    ml_status = get_ml_status()

    if ml_status.get('model_loaded'):
        st.markdown('<span class="ml-badge-online">✅ ML Agent Online</span>', unsafe_allow_html=True)
        st.caption(f"Model: {ml_status.get('model_path', 'Unknown')}")
    elif ml_status.get('reward_calc_available'):
        st.markdown('<span class="ml-badge-online">✅ Hybrid Ready</span>', unsafe_allow_html=True)
        st.caption("RewardCalculator active (no ML model needed)")
    else:
        st.markdown('<span class="ml-badge-offline">⚠️ ML Offline</span>', unsafe_allow_html=True)
        st.caption("Using traditional algorithms")

    st.markdown("---")

    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Solves", st.session_state.solve_count)
    with col2:
        st.metric("Today", st.session_state.daily_usage['count'])

    if st.session_state.history:
        recent = st.session_state.history[-10:]
        success_rate = sum(1 for x in recent if x['success']) / len(recent) * 100
        st.metric("Success Rate", f"{success_rate:.0f}%")

    st.markdown("---")

    st.markdown("### 📡 Components")
    status_items = [
        ("ML Agent", ML_AVAILABLE),
        ("Reward Calc", REWARD_CALC_AVAILABLE),
        ("Visual Cards", VISUAL_CARDS_AVAILABLE),
        ("Image Input", IMAGE_INPUT_AVAILABLE),
        ("Card Picker", PICKER_AVAILABLE),
    ]
    for name, status in status_items:
        st.write(f"{'✅' if status else '⚠️'} {name}")

    st.markdown("---")

    with st.expander("❓ How to use"):
        st.markdown("""
        **Quick Start:**
        1. Choose solver mode (🔥 Hybrid recommended!)
        2. Enter 13 cards
        3. Click **SOLVE** 🚀
        
        **Modes:**
        - **🔥 Hybrid**: SmartSolver + AI scoring (BEST!)
        - **🤖 ML Agent**: Pure AI (fast)
        - **⚖️ Balanced**: Traditional (reliable)
        
        **Card Format:** `AS KH QD JC 10S`
        """)


# ============== MAIN CONTENT ==============
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
            if st.button(name, key=f"ex_{i}"):
                st.session_state.card_input = cards
                st.session_state.cards_ready_from_picker = False
                st.session_state.cards_from_image = False
                rerun()


# ===== INPUT SECTION =====
st.markdown("## 📥 Input Your Cards")

solve_button = False
card_input = st.session_state.card_input

tab1, tab2, tab3 = st.tabs([
    "✍️ Type Cards",
    "🎴 Click to Pick",
    "📷 Screenshot (Ctrl+V)"
])


# TAB 1: Type Cards
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        typed_input = st.text_input(
            "Enter 13 cards (space-separated)",
            value=st.session_state.card_input,
            help="Format: AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
            key="main_input",
            placeholder="e.g., AS KH QD JC 10S 9H 8D 7C 6S..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 SOLVE", type="primary", key="solve_type"):
            card_input = typed_input
            solve_button = True
            st.session_state.cards_ready_from_picker = False
            st.session_state.cards_from_image = False


# TAB 2: Pick Cards
with tab2:
    if PICKER_AVAILABLE:
        if st.session_state.cards_ready_from_picker and st.session_state.card_input:
            st.success("✅ Cards selected!")
            st.code(st.session_state.card_input, language=None)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 SOLVE NOW", type="primary", key="solve_pick"):
                    card_input = st.session_state.card_input
                    solve_button = True
            with col2:
                if st.button("🔄 Pick Different", key="pick_different"):
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
    else:
        st.info("🎴 Card Picker component chưa load được. Dùng text input:")
        st.markdown("""
        **Quick reference:**
        | Suit | Code | | Rank | Code |
        |------|------|-|------|------|
        | ♠ Spades | S | | Ace | A |
        | ♥ Hearts | H | | King | K |
        | ♦ Diamonds | D | | Queen | Q |
        | ♣ Clubs | C | | Jack | J |
        """)
        fallback_pick = st.text_input(
            "Enter 13 cards:",
            placeholder="AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
            key="fallback_picker_input"
        )
        if fallback_pick and st.button("🚀 SOLVE", key="solve_fallback_pick"):
            card_input = fallback_pick
            solve_button = True


# TAB 3: Screenshot / Paste Image
with tab3:
    if st.session_state.cards_from_image and st.session_state.card_input:
        st.success("✅ 13 cards detected!")
        st.code(st.session_state.card_input, language=None)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 SOLVE NOW", type="primary", key="solve_image"):
                card_input = st.session_state.card_input
                solve_button = True
        with col2:
            if st.button("📷 Detect Different", key="new_image"):
                st.session_state.cards_from_image = False
                rerun()
    else:
        detected_cards = None

        if IMAGE_INPUT_AVAILABLE:
            try:
                detected_cards = image_input_component(key="main_image_input")
            except Exception as e:
                st.warning(f"⚠️ Image component error: {e}")
                detected_cards = fallback_image_input(key="main_fallback")
        else:
            detected_cards = fallback_image_input(key="main_fallback")

        if detected_cards:
            st.session_state.card_input = detected_cards
            st.session_state.cards_from_image = True
            st.session_state.cards_ready_from_picker = False
            rerun()


# ============== SOLVE ==============
if solve_button and card_input:
    can_solve, error_msg = check_usage_limit(mode_value)
    if not can_solve:
        st.error(f"❌ {error_msg}")
        st.stop()

    error = None
    result = None

    try:
        cards = Deck.parse_hand(card_input)

        if len(cards) != 13:
            st.error(f"❌ Need exactly 13 cards, got {len(cards)}")
        else:
            if VISUAL_CARDS_AVAILABLE:
                st.markdown("### 🎴 Your Input Cards")
                st.markdown(render_input_cards_preview([str(c) for c in cards]), unsafe_allow_html=True)

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text(f"🤖 Solving with {selected_mode}...")
            progress_bar.progress(30)

            solver_mode = SolverMode(mode_value)
            solver = UltimateSolver(cards, mode=solver_mode, verbose=False)
            result = solver.solve()

            progress_bar.progress(100)
            status_text.text("✅ Solution found!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

            log_solve(mode_value, card_input, result)

            st.success("✅ **OPTIMAL SOLUTION FOUND!**")
            st.markdown("---")

            st.markdown("## 🎯 Optimal Arrangement")

            back_eval = HandEvaluator.evaluate(result.back)
            middle_eval = HandEvaluator.evaluate(result.middle)
            front_eval = HandEvaluator.evaluate(result.front)

            if VISUAL_CARDS_AVAILABLE:
                st.markdown(render_comparison_cards(
                    [str(c) for c in result.back],
                    [str(c) for c in result.middle],
                    [str(c) for c in result.front],
                    (str(back_eval), str(middle_eval), str(front_eval))
                ), unsafe_allow_html=True)
            else:
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

            st.markdown("## 📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Value", f"{result.ev:+.2f}")
            col2.metric("Bonus Points", f"+{result.bonus}")
            col3.metric("Scoop Chance", f"{result.p_scoop*100:.1f}%",
                        delta=f"{(result.p_scoop-0.25)*100:+.1f}%" if result.p_scoop > 0.25 else None)
            col4.metric("Win 2/3 Chi", f"{result.p_win_2_of_3*100:.1f}%",
                        delta=f"{(result.p_win_2_of_3-0.55)*100:+.1f}%" if result.p_win_2_of_3 > 0.55 else None)

            st.markdown("---")

            st.markdown("## 📈 Win Probability Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔵 Back Win Rate**")
                st.progress(min(result.p_win_back, 1.0))
                st.write(f"{result.p_win_back*100:.1f}%")
            with col2:
                st.markdown("**🟢 Middle Win Rate**")
                st.progress(min(result.p_win_middle, 1.0))
                st.write(f"{result.p_win_middle*100:.1f}%")
            with col3:
                st.markdown("**🟡 Front Win Rate**")
                st.progress(min(result.p_win_front, 1.0))
                st.write(f"{result.p_win_front*100:.1f}%")

            total_score = result.p_win_front * 40 + result.p_win_middle * 30 + result.p_win_back * 30
            st.markdown("### 🎯 Overall Assessment")
            if total_score >= 70:
                st.success(f"🔥 **EXCELLENT!** (Score: {total_score:.0f}/100)")
            elif total_score >= 55:
                st.info(f"✅ **GOOD** (Score: {total_score:.0f}/100)")
            elif total_score >= 40:
                st.warning(f"⚠️ **AVERAGE** (Score: {total_score:.0f}/100)")
            else:
                st.error(f"❌ **WEAK** (Score: {total_score:.0f}/100)")

            st.markdown("---")

            if result.mode.value.startswith('ml_') and result.ml_metrics:
                st.markdown("### 🤖 AI Analysis")
                m = result.ml_metrics
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("Mode", m.get('mode', 'unknown'))
                with info_cols[1]:
                    st.metric("Candidates", m.get('num_candidates', 0))
                with info_cols[2]:
                    st.metric("Valid", m.get('num_valid', 0))
                with info_cols[3]:
                    st.metric("Combined Score", f"{m.get('combined_score', 0):.2f}")

                with st.expander("🔍 Scoring Breakdown"):
                    st.write(f"- SmartSolver score: {m.get('smart_score', 0):.2f}")
                    st.write(f"- RewardCalc reward: {m.get('reward', 0):.2f}")
                    st.write(f"- ML Agent reward: {m.get('ml_reward', 0):.2f}")
                    st.write(f"- Combined: {m.get('combined_score', 0):.2f}")
                    st.write(f"- Bonus: +{m.get('bonus', 0)}")

            st.markdown("---")

            st.markdown("## 💡 Strategic Recommendations")
            recs = []
            if result.p_scoop > 0.3:
                recs.append("🎯 **High scoop chance!** Play aggressively.")
            if result.ev > 1.5:
                recs.append("💰 **Very positive EV!** Highly profitable.")
            elif result.ev > 0.5:
                recs.append("💵 **Positive EV.** Good profit potential.")
            if result.bonus >= 6:
                recs.append(f"⭐ **Big bonus!** +{result.bonus} points!")
            elif result.bonus >= 3:
                recs.append(f"✨ **Bonus hand!** +{result.bonus} points.")
            if result.ev < -0.5:
                recs.append("⚠️ **Negative EV.** Play defensively.")
            for rec in (recs or ["🤔 **Standard hand.** Play basic strategy."]):
                st.success(rec)

            st.markdown("---")

            with st.expander("📈 Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎲 Win Probabilities:**")
                    st.write(f"- Front: {result.p_win_front*100:.1f}%")
                    st.write(f"- Middle: {result.p_win_middle*100:.1f}%")
                    st.write(f"- Back: {result.p_win_back*100:.1f}%")
                    st.write(f"- Scoop: {result.p_scoop*100:.1f}%")
                with col2:
                    st.markdown("**⚙️ Computation:**")
                    st.write(f"- Mode: **{selected_mode}**")
                    st.write(f"- Time: {result.computation_time:.3f}s")
                    st.write(f"- Bonus: +{result.bonus}")
                    st.write(f"- EV: {result.ev:+.2f}")

            st.markdown("---")

            st.markdown("### 📤 Actions")
            col1, col2, col3 = st.columns(3)
            with col1:
                export_text = (
                    f"🃏 Mau Binh AI Solver Pro\n{'='*60}\n"
                    f"INPUT: {card_input}\n\n"
                    f"Back:   {Deck.cards_to_string(result.back)} → {back_eval}\n"
                    f"Middle: {Deck.cards_to_string(result.middle)} → {middle_eval}\n"
                    f"Front:  {Deck.cards_to_string(result.front)} → {front_eval}\n\n"
                    f"EV: {result.ev:+.2f} | Bonus: +{result.bonus} | Mode: {selected_mode}\n"
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                st.download_button("📥 Download", data=export_text,
                                   file_name=f"maubinh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                   mime="text/plain")
            with col2:
                if st.button("🔄 Solve Another", key="solve_another"):
                    st.session_state.cards_from_image = False
                    st.session_state.cards_ready_from_picker = False
                    rerun()
            with col3:
                if st.button("🔀 Try Different Mode"):
                    st.info("👈 Select a different mode in sidebar")

            st.session_state.daily_usage['count'] += 1

    except Exception as e:
        error = str(e)
        st.error(f"❌ **Error:** {error}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())
        log_solve(mode_value, card_input, None, error)


# ============== FOOTER ==============
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Resources")
    st.markdown("- 📖 How to Play")

with col2:
    st.markdown("### ⚡ Technology")
    st.markdown("- DQN + Transformer\n- Game Theory\n- Monte Carlo\n- Hybrid AI")

with col3:
    st.markdown("### 📊 Available Modes")
    for m in get_available_modes():
        st.write(f"✅ {m}")

st.markdown(
    '<p style="text-align:center;color:#999;margin-top:2rem;font-size:0.9rem;">'
    'Made with ❤️ using Python, PyTorch & Streamlit<br>'
    '© 2024 Mau Binh AI Solver Pro • ML Agent V2.2'
    '</p>', unsafe_allow_html=True
)