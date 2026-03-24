"""
Compare Solver Modes - So sánh kết quả giữa TẤT CẢ các mode
Bao gồm: Fast, Balanced, Accurate, ML Agent, Ultimate
"""
import streamlit as st
import sys
import os
from datetime import datetime
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ===== PATH SETUP =====
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(web_dir)

sys.path.insert(0, os.path.join(src_dir, 'core'))
sys.path.insert(0, os.path.join(src_dir, 'engines'))
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(web_dir, 'components'))

from card import Deck, Card
from ultimate_solver import UltimateSolver, SolverMode
from evaluator import HandEvaluator

# Try import visual components
try:
    from card_renderer import render_comparison_cards, get_card_html
    VISUAL_CARDS_AVAILABLE = True
except ImportError:
    VISUAL_CARDS_AVAILABLE = False

# Check ML availability
try:
    from ultimate_solver import ML_AVAILABLE
except ImportError:
    ML_AVAILABLE = False

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Compare All Modes - Mau Binh AI",
    page_icon="🔄",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .compare-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .mode-fast { background: #38ef7d; color: #000; }
    .mode-balanced { background: #667eea; color: #fff; }
    .mode-accurate { background: #f39c12; color: #000; }
    .mode-ml { background: #9b59b6; color: #fff; }
    .mode-ultimate { background: #e74c3c; color: #fff; }
    
    .winner-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .ml-vs-traditional {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        border: 2px solid #9b59b6;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .speed-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .speed-fast { background: #d4edda; color: #155724; }
    .speed-medium { background: #fff3cd; color: #856404; }
    .speed-slow { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<h1 class="compare-header">🔄 Compare All Solver Modes</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 1.1rem;">'
    'Compare <b>Fast</b>, <b>Balanced</b>, <b>Accurate</b>, <b>ML Agent</b>, and <b>Ultimate</b> modes side-by-side<br>'
    '🤖 See how AI Deep Learning compares to traditional algorithms!'
    '</p>',
    unsafe_allow_html=True
)

# Mode status indicators
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<span class="mode-badge mode-fast">⚡ Fast</span>', unsafe_allow_html=True)
    st.caption("< 1 second")

with col2:
    st.markdown('<span class="mode-badge mode-balanced">⚖️ Balanced</span>', unsafe_allow_html=True)
    st.caption("2-5 seconds")

with col3:
    st.markdown('<span class="mode-badge mode-accurate">🎯 Accurate</span>', unsafe_allow_html=True)
    st.caption("10-20 seconds")

with col4:
    if ML_AVAILABLE:
        st.markdown('<span class="mode-badge mode-ml">🤖 ML Agent ✓</span>', unsafe_allow_html=True)
        st.caption("AI Online")
    else:
        st.markdown('<span class="mode-badge" style="background:#ccc;">🤖 ML Agent ✗</span>', unsafe_allow_html=True)
        st.caption("AI Offline")

with col5:
    st.markdown('<span class="mode-badge mode-ultimate">🚀 Ultimate</span>', unsafe_allow_html=True)
    st.caption("30-60 seconds")

st.markdown("---")

# ===== MODE DEFINITIONS =====
MODES = {
    "⚡ Fast": {
        "mode": SolverMode.FAST,
        "color": "#38ef7d",
        "desc": "Greedy algorithm, instant results",
        "icon": "⚡",
        "category": "traditional",
        "time_class": "fast"
    },
    "⚖️ Balanced": {
        "mode": SolverMode.BALANCED,
        "color": "#667eea",
        "desc": "Optimized search with pruning",
        "icon": "⚖️",
        "category": "traditional",
        "time_class": "medium"
    },
    "🎯 Accurate": {
        "mode": SolverMode.ACCURATE,
        "color": "#f39c12",
        "desc": "Exhaustive search, high accuracy",
        "icon": "🎯",
        "category": "traditional",
        "time_class": "slow"
    },
    "🤖 ML Agent": {
        "mode": SolverMode.ML_ONLY,
        "color": "#9b59b6",
        "desc": "Deep Q-Network (DQN) trained on 100K+ hands",
        "icon": "🤖",
        "category": "ml",
        "time_class": "medium"
    },
    "🚀 Ultimate": {
        "mode": SolverMode.ULTIMATE,
        "color": "#e74c3c",
        "desc": "Ensemble: ML + Monte Carlo + Game Theory",
        "icon": "🚀",
        "category": "hybrid",
        "time_class": "slow"
    }
}

# ===== INPUT SECTION =====
st.markdown("## 📥 Input Cards")

col1, col2 = st.columns([3, 1])

with col1:
    examples = {
        "🔥 Premium Hand": "AS AH AD KC KD QS QH JD 10C 9S 8H 7D 6C",
        "⚖️ Balanced Hand": "KS QH JD 10C 9S 8H 7D 6C 5S 4H 3D 2C AS",
        "🎲 Random Mix": "AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
        "⚠️ Weak Hand": "9S 8H 7D 6C 5S 4H 3D 2C KS QH JD 10C 9H",
        "🃏 Flush Potential": "AS KS QS JS 10S 9H 8H 7D 6C 5S 4H 3D 2C",
        "🎯 Straight Draw": "AS 2H 3D 4C 5S 6H 7D 8C 9S 10H JD QC KS",
        "💎 Full House Setup": "AS AH AD KS KH KC QD JC 10S 9H 8D 7C 6S"
    }
    
    selected_example = st.selectbox(
        "📝 Quick Examples",
        ["Custom Input"] + list(examples.keys()),
        index=0
    )
    
    if selected_example != "Custom Input":
        default_cards = examples[selected_example]
    else:
        default_cards = st.session_state.get('compare_cards', "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S")
    
    card_input = st.text_input(
        "Enter 13 cards (space-separated)",
        value=default_cards,
        help="Format: AS KH QD JC 10S...",
        key="compare_input"
    )

with col2:
    st.markdown("### ⚙️ Options")
    
    # Mode selection - DEFAULT ALL 5 MODES
    selected_modes = st.multiselect(
        "Modes to compare",
        list(MODES.keys()),
        default=list(MODES.keys()),  # 🔥 ALL 5 MODES BY DEFAULT
        help="Select which modes to compare"
    )
    
    # Check if ML is available
    if "🤖 ML Agent" in selected_modes and not ML_AVAILABLE:
        st.warning("⚠️ ML Agent offline, will skip")

# Advanced options
with st.expander("⚙️ Advanced Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        parallel_execution = st.checkbox(
            "🚀 Parallel Execution",
            value=True,
            help="Run modes simultaneously (faster)"
        )
    
    with col2:
        show_cards_visual = st.checkbox(
            "🎴 Visual Cards",
            value=VISUAL_CARDS_AVAILABLE,
            disabled=not VISUAL_CARDS_AVAILABLE
        )
    
    with col3:
        show_ml_analysis = st.checkbox(
            "🤖 ML vs Traditional Analysis",
            value=True,
            help="Show detailed comparison between ML and traditional methods"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        show_time_analysis = st.checkbox(
            "⏱️ Time Analysis",
            value=True
        )
    
    with col5:
        show_arrangement_diff = st.checkbox(
            "🔍 Show Differences",
            value=True
        )
    
    with col6:
        timeout_seconds = st.number_input(
            "⏰ Timeout (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            help="Max time for Ultimate mode"
        )

st.markdown("---")

# ===== COMPARE BUTTON =====
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    compare_button = st.button(
        "🔄 COMPARE ALL MODES",
        type="primary",
        use_container_width=True
    )

# ===== COMPARISON LOGIC =====
def solve_with_mode(cards, mode_name, mode_info, timeout=120):
    """Solve hand with specific mode and return results"""
    try:
        start_time = time.time()
        solver = UltimateSolver(cards, mode=mode_info["mode"], verbose=False)
        result = solver.solve()
        actual_time = time.time() - start_time
        
        return {
            "mode_name": mode_name,
            "mode_info": mode_info,
            "result": result,
            "actual_time": actual_time,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "mode_name": mode_name,
            "mode_info": mode_info,
            "result": None,
            "actual_time": time.time() - start_time if 'start_time' in dir() else 0,
            "success": False,
            "error": str(e)
        }


def get_arrangement_hash(result):
    """Get hash of arrangement for comparison"""
    if result is None:
        return None
    
    front = tuple(sorted(str(c) for c in result.front))
    middle = tuple(sorted(str(c) for c in result.middle))
    back = tuple(sorted(str(c) for c in result.back))
    
    return (front, middle, back)


def compare_arrangements_detailed(results):
    """Detailed comparison between all arrangements"""
    comparisons = []
    
    successful = [r for r in results if r["success"]]
    
    for i, r1 in enumerate(successful):
        for j, r2 in enumerate(successful):
            if i >= j:
                continue
            
            hash1 = get_arrangement_hash(r1["result"])
            hash2 = get_arrangement_hash(r2["result"])
            
            same_front = hash1[0] == hash2[0] if hash1 and hash2 else False
            same_middle = hash1[1] == hash2[1] if hash1 and hash2 else False
            same_back = hash1[2] == hash2[2] if hash1 and hash2 else False
            
            comparisons.append({
                "mode1": r1["mode_name"],
                "mode2": r2["mode_name"],
                "category1": r1["mode_info"]["category"],
                "category2": r2["mode_info"]["category"],
                "same_arrangement": same_front and same_middle and same_back,
                "ev_diff": r1["result"].ev - r2["result"].ev,
                "scoop_diff": (r1["result"].p_scoop - r2["result"].p_scoop) * 100,
                "same_front": same_front,
                "same_middle": same_middle,
                "same_back": same_back,
                "time_ratio": r1["actual_time"] / r2["actual_time"] if r2["actual_time"] > 0 else 0
            })
    
    return comparisons


# ===== EXECUTE COMPARISON =====
if compare_button and card_input and selected_modes:
    try:
        # Validate input
        cards = Deck.parse_hand(card_input)
        
        if len(cards) != 13:
            st.error(f"❌ Need exactly 13 cards, got {len(cards)}")
            st.stop()
        
        # Check for duplicates
        card_strs = [str(c) for c in cards]
        if len(set(card_strs)) != 13:
            st.error("❌ Duplicate cards detected!")
            st.stop()
        
        # Store for later
        st.session_state.compare_cards = card_input
        
        # Filter out ML if not available
        active_modes = selected_modes.copy()
        if "🤖 ML Agent" in active_modes and not ML_AVAILABLE:
            active_modes.remove("🤖 ML Agent")
            st.warning("⚠️ ML Agent is offline, skipping...")
        
        if not active_modes:
            st.error("❌ No valid modes selected!")
            st.stop()
        
        # Show input preview
        st.markdown("### 🎴 Input Cards")
        if VISUAL_CARDS_AVAILABLE and show_cards_visual:
            from card_renderer import render_input_cards_preview
            st.markdown(
                render_input_cards_preview([str(c) for c in cards]),
                unsafe_allow_html=True
            )
        else:
            st.code(card_input)
        
        st.markdown("---")
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_estimates = st.empty()
        
        results = []
        total_modes = len(active_modes)
        start_total = time.time()
        
        # Estimate total time
        time_est = sum([
            1 if MODES[m]["time_class"] == "fast" else 
            5 if MODES[m]["time_class"] == "medium" else 30 
            for m in active_modes
        ])
        time_estimates.info(f"⏱️ Estimated total time: {time_est}-{time_est*2} seconds")
        
        if parallel_execution and total_modes > 1:
            # Parallel execution
            status_text.text("🚀 Running all modes in parallel...")
            
            with ThreadPoolExecutor(max_workers=min(total_modes, 5)) as executor:
                futures = {
                    executor.submit(
                        solve_with_mode, 
                        cards.copy(), 
                        mode_name, 
                        MODES[mode_name],
                        timeout_seconds
                    ): mode_name
                    for mode_name in active_modes
                }
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    progress_bar.progress(completed / total_modes)
                    
                    # Show status with time
                    if result["success"]:
                        status_text.text(
                            f"✅ {result['mode_name']}: EV={result['result'].ev:+.3f} "
                            f"({result['actual_time']:.2f}s)"
                        )
                    else:
                        status_text.text(f"❌ {result['mode_name']}: {result['error']}")
        else:
            # Sequential execution
            for i, mode_name in enumerate(active_modes):
                status_text.text(f"🧠 Running {mode_name}...")
                result = solve_with_mode(
                    cards.copy(), 
                    mode_name, 
                    MODES[mode_name],
                    timeout_seconds
                )
                results.append(result)
                progress_bar.progress((i + 1) / total_modes)
                
                if result["success"]:
                    status_text.text(
                        f"✅ {mode_name}: EV={result['result'].ev:+.3f} "
                        f"({result['actual_time']:.2f}s)"
                    )
        
        total_time = time.time() - start_total
        
        # Sort by mode order
        mode_order = list(MODES.keys())
        results.sort(key=lambda x: mode_order.index(x["mode_name"]) if x["mode_name"] in mode_order else 999)
        
        progress_bar.empty()
        status_text.empty()
        time_estimates.empty()
        
        # ===== SUCCESS MESSAGE =====
        successful_count = sum(1 for r in results if r["success"])
        st.success(f"✅ Compared {successful_count}/{len(results)} modes in {total_time:.1f}s!")
        
        st.markdown("---")
        
        # ===== SUMMARY TABLE =====
        st.markdown("## 📊 Results Summary")
        
        # Create summary dataframe
        summary_data = []
        for r in results:
            if r["success"]:
                res = r["result"]
                summary_data.append({
                    "Mode": r["mode_name"],
                    "Category": r["mode_info"]["category"].upper(),
                    "EV": res.ev,
                    "Bonus": res.bonus,
                    "Scoop %": res.p_scoop * 100,
                    "Win 2/3 %": res.p_win_2_of_3 * 100,
                    "Front %": res.p_win_front * 100,
                    "Middle %": res.p_win_middle * 100,
                    "Back %": res.p_win_back * 100,
                    "Time (s)": r["actual_time"],
                    "Arrangements": res.num_arrangements_evaluated,
                    "Color": r["mode_info"]["color"]
                })
            else:
                summary_data.append({
                    "Mode": r["mode_name"],
                    "Category": r["mode_info"]["category"].upper(),
                    "EV": None,
                    "Error": r["error"][:50] + "..." if len(r.get("error", "")) > 50 else r.get("error", "")
                })
        
        df = pd.DataFrame(summary_data)
        
        # Find best values
        valid_df = df[df["EV"].notna()]
        
        if len(valid_df) > 0:
            best_ev_idx = valid_df["EV"].idxmax()
            best_scoop_idx = valid_df["Scoop %"].idxmax() if "Scoop %" in valid_df.columns else None
            fastest_idx = valid_df["Time (s)"].idxmin() if "Time (s)" in valid_df.columns else None
            best_win23_idx = valid_df["Win 2/3 %"].idxmax() if "Win 2/3 %" in valid_df.columns else None
        else:
            best_ev_idx = best_scoop_idx = fastest_idx = best_win23_idx = None
        
        # Display styled dataframe
        display_cols = ["Mode", "Category", "EV", "Bonus", "Scoop %", "Win 2/3 %", "Time (s)", "Arrangements"]
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols].style.format({
                "EV": lambda x: f"{x:+.3f}" if pd.notna(x) else "-",
                "Bonus": lambda x: f"+{x:.0f}" if pd.notna(x) else "-",
                "Scoop %": lambda x: f"{x:.1f}%" if pd.notna(x) else "-",
                "Win 2/3 %": lambda x: f"{x:.1f}%" if pd.notna(x) else "-",
                "Time (s)": lambda x: f"{x:.3f}s" if pd.notna(x) else "-",
                "Arrangements": lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            }).apply(lambda row: [
                'background-color: #d4edda; font-weight: bold' if row.name == best_ev_idx and i == available_cols.index("EV") else
                'background-color: #cce5ff; font-weight: bold' if row.name == fastest_idx and i == available_cols.index("Time (s)") else
                ''
                for i in range(len(row))
            ], axis=1),
            use_container_width=True
        )
        
        # ===== WINNER CARDS =====
        st.markdown("### 🏆 Best Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if best_ev_idx is not None:
                best_ev_mode = df.loc[best_ev_idx, "Mode"]
                best_ev_val = df.loc[best_ev_idx, "EV"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 1.2rem; border-radius: 12px; text-align: center; color: white;">
                    <h4 style="margin:0;">💰 Best EV</h4>
                    <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">{best_ev_val:+.3f}</p>
                    <p style="margin:0; font-size: 1rem;">{best_ev_mode}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if best_scoop_idx is not None:
                best_scoop_mode = df.loc[best_scoop_idx, "Mode"]
                best_scoop_val = df.loc[best_scoop_idx, "Scoop %"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.2rem; border-radius: 12px; text-align: center; color: white;">
                    <h4 style="margin:0;">🎯 Best Scoop</h4>
                    <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">{best_scoop_val:.1f}%</p>
                    <p style="margin:0; font-size: 1rem;">{best_scoop_mode}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if fastest_idx is not None:
                fastest_mode = df.loc[fastest_idx, "Mode"]
                fastest_val = df.loc[fastest_idx, "Time (s)"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.2rem; border-radius: 12px; text-align: center; color: white;">
                    <h4 style="margin:0;">⚡ Fastest</h4>
                    <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">{fastest_val:.3f}s</p>
                    <p style="margin:0; font-size: 1rem;">{fastest_mode}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if best_win23_idx is not None:
                best_win23_mode = df.loc[best_win23_idx, "Mode"]
                best_win23_val = df.loc[best_win23_idx, "Win 2/3 %"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%); 
                            padding: 1.2rem; border-radius: 12px; text-align: center; color: white;">
                    <h4 style="margin:0;">✅ Best Win 2/3</h4>
                    <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">{best_win23_val:.1f}%</p>
                    <p style="margin:0; font-size: 1rem;">{best_win23_mode}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== ML VS TRADITIONAL ANALYSIS =====
        if show_ml_analysis:
            st.markdown("## 🤖 ML Agent vs Traditional Algorithms")
            
            ml_results = [r for r in results if r["success"] and r["mode_info"]["category"] == "ml"]
            trad_results = [r for r in results if r["success"] and r["mode_info"]["category"] == "traditional"]
            hybrid_results = [r for r in results if r["success"] and r["mode_info"]["category"] == "hybrid"]
            
            if ml_results and trad_results:
                ml_r = ml_results[0]
                best_trad = max(trad_results, key=lambda x: x["result"].ev)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="ml-vs-traditional">
                        <h3 style="color: #9b59b6; margin-top:0;">🤖 ML Agent (Deep Learning)</h3>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Expected Value", f"{ml_r['result'].ev:+.3f}")
                    st.metric("Scoop Chance", f"{ml_r['result'].p_scoop*100:.1f}%")
                    st.metric("Computation Time", f"{ml_r['actual_time']:.2f}s")
                    
                    st.markdown("""
                    **How it works:**
                    - Deep Q-Network (DQN) architecture
                    - Trained on 100,000+ expert hands
                    - Learns patterns from experience
                    - Uses neural network inference
                    """)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2d3436 0%, #636e72 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; border: 2px solid #f39c12;">
                        <h3 style="color: #f39c12; margin-top:0;">📐 Best Traditional ({best_trad['mode_name']})</h3>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Expected Value", f"{best_trad['result'].ev:+.3f}")
                    st.metric("Scoop Chance", f"{best_trad['result'].p_scoop*100:.1f}%")
                    st.metric("Computation Time", f"{best_trad['actual_time']:.2f}s")
                    
                    st.markdown("""
                    **How it works:**
                    - Rule-based algorithms
                    - Exhaustive/pruned search
                    - Mathematical optimization
                    - Hand-crafted heuristics
                    """)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Comparison insight
                ev_diff = ml_r['result'].ev - best_trad['result'].ev
                scoop_diff = (ml_r['result'].p_scoop - best_trad['result'].p_scoop) * 100
                time_ratio = ml_r['actual_time'] / best_trad['actual_time'] if best_trad['actual_time'] > 0 else 0
                
                st.markdown("### 📈 Comparison Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if ev_diff > 0.01:
                        st.success(f"🤖 ML wins by EV: **+{ev_diff:.3f}**")
                    elif ev_diff < -0.01:
                        st.warning(f"📐 Traditional wins by EV: **{-ev_diff:.3f}**")
                    else:
                        st.info("🤝 **Tie on EV** (within 0.01)")
                
                with col2:
                    if scoop_diff > 1:
                        st.success(f"🤖 ML better scoop: **+{scoop_diff:.1f}%**")
                    elif scoop_diff < -1:
                        st.warning(f"📐 Traditional better scoop: **{-scoop_diff:.1f}%**")
                    else:
                        st.info("🤝 **Similar scoop chance**")
                
                with col3:
                    if time_ratio < 0.5:
                        st.success(f"🤖 ML is **{1/time_ratio:.1f}x faster**")
                    elif time_ratio > 2:
                        st.warning(f"📐 Traditional is **{time_ratio:.1f}x faster**")
                    else:
                        st.info("🤝 **Similar speed**")
                
                # Same arrangement check
                ml_hash = get_arrangement_hash(ml_r["result"])
                trad_hash = get_arrangement_hash(best_trad["result"])
                
                if ml_hash == trad_hash:
                    st.success("✅ **ML and Traditional produced the SAME arrangement!**")
                else:
                    st.warning("⚠️ **Different arrangements** - see detailed comparison below")
            
            else:
                if not ml_results:
                    st.info("🤖 ML Agent not included in comparison. Enable it to see ML vs Traditional analysis.")
                if not trad_results:
                    st.info("📐 No traditional modes selected.")
            
            # Ultimate mode analysis
            if hybrid_results:
                st.markdown("### 🚀 Ultimate Mode Analysis")
                
                ultimate_r = hybrid_results[0]
                
                st.markdown(f"""
                <div class="insight-box">
                    <h4>🚀 Ultimate Mode Performance</h4>
                    <p><b>Ultimate</b> combines the best of both worlds:</p>
                    <ul>
                        <li>✅ ML Agent for initial suggestion</li>
                        <li>✅ Monte Carlo simulation for validation</li>
                        <li>✅ Game Theory optimization</li>
                        <li>✅ Ensemble voting from multiple strategies</li>
                    </ul>
                    <p><b>Result:</b> EV = {ultimate_r['result'].ev:+.3f}, Time = {ultimate_r['actual_time']:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Compare Ultimate vs others
                if len(valid_df) > 1:
                    ultimate_ev = ultimate_r['result'].ev
                    other_evs = [r['result'].ev for r in results if r['success'] and r != ultimate_r]
                    
                    if other_evs:
                        avg_other_ev = sum(other_evs) / len(other_evs)
                        best_other_ev = max(other_evs)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            diff_vs_avg = ultimate_ev - avg_other_ev
                            st.metric(
                                "Ultimate vs Average",
                                f"{ultimate_ev:+.3f}",
                                delta=f"{diff_vs_avg:+.3f}",
                                delta_color="normal" if diff_vs_avg >= 0 else "inverse"
                            )
                        
                        with col2:
                            diff_vs_best = ultimate_ev - best_other_ev
                            st.metric(
                                "Ultimate vs Best Other",
                                f"{ultimate_ev:+.3f}",
                                delta=f"{diff_vs_best:+.3f}",
                                delta_color="normal" if diff_vs_best >= 0 else "inverse"
                            )
        
        st.markdown("---")
        
        # ===== VISUAL CHARTS =====
        st.markdown("## 📈 Visual Comparison")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EV & Bonus", "⏱️ Performance", "🎯 Win Rates", "🔬 Detailed"])
        
        with tab1:
            # EV Comparison Bar Chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Expected Value (EV)", "Bonus Points"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            colors = [MODES.get(m, {}).get("color", "#667eea") for m in valid_df["Mode"]]
            
            fig.add_trace(
                go.Bar(
                    x=valid_df["Mode"],
                    y=valid_df["EV"],
                    marker_color=colors,
                    text=valid_df["EV"].apply(lambda x: f"{x:+.3f}"),
                    textposition="outside",
                    name="EV"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=valid_df["Mode"],
                    y=valid_df["Bonus"],
                    marker_color=colors,
                    text=valid_df["Bonus"].apply(lambda x: f"+{x:.0f}"),
                    textposition="outside",
                    name="Bonus"
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Time & Efficiency
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Computation Time (log scale)", "EV per Second (Efficiency)"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Time (log scale for better visualization)
            fig.add_trace(
                go.Bar(
                    x=valid_df["Mode"],
                    y=valid_df["Time (s)"],
                    marker_color=colors,
                    text=valid_df["Time (s)"].apply(lambda x: f"{x:.2f}s"),
                    textposition="outside",
                    name="Time"
                ),
                row=1, col=1
            )
            
            # EV per second (efficiency)
            efficiency = valid_df["EV"] / valid_df["Time (s)"].replace(0, 0.001)
            
            fig.add_trace(
                go.Bar(
                    x=valid_df["Mode"],
                    y=efficiency,
                    marker_color=colors,
                    text=efficiency.apply(lambda x: f"{x:.2f}"),
                    textposition="outside",
                    name="EV/s"
                ),
                row=1, col=2
            )
            
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed ranking
            st.markdown("### ⚡ Speed Ranking")
            speed_df = valid_df[["Mode", "Time (s)", "EV"]].copy()
            speed_df["Efficiency"] = speed_df["EV"] / speed_df["Time (s)"].replace(0, 0.001)
            speed_df = speed_df.sort_values("Time (s)")
            
            for i, row in speed_df.iterrows():
                time_class = "fast" if row["Time (s)"] < 2 else "medium" if row["Time (s)"] < 15 else "slow"
                st.markdown(
                    f'<span class="speed-indicator speed-{time_class}">{row["Time (s)"]:.2f}s</span> '
                    f'**{row["Mode"]}** (EV: {row["EV"]:+.3f}, Efficiency: {row["Efficiency"]:.2f} EV/s)',
                    unsafe_allow_html=True
                )
        
        with tab3:
            # Win rates radar chart
            if "Front %" in valid_df.columns:
                categories = ['Front', 'Middle', 'Back', 'Scoop', 'Win 2/3']
                
                fig = go.Figure()
                
                for _, row in valid_df.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            row["Front %"],
                            row["Middle %"],
                            row["Back %"],
                            row["Scoop %"],
                            row["Win 2/3 %"]
                        ],
                        theta=categories,
                        fill='toself',
                        name=row["Mode"],
                        line_color=MODES.get(row["Mode"], {}).get("color", "#667eea")
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title_text="Win Rate Comparison (Radar)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Win rate heatmap
            st.markdown("### 🎯 Win Rate Heatmap")
            
            heatmap_data = valid_df[["Mode", "Front %", "Middle %", "Back %"]].set_index("Mode")
            
            fig = px.imshow(
                heatmap_data.T,
                labels=dict(x="Mode", y="Chi", color="Win %"),
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Detailed metrics comparison
            st.markdown("### 🔬 All Metrics Comparison")
            
            # Normalize metrics for comparison
            metrics_to_compare = ["EV", "Scoop %", "Win 2/3 %", "Front %", "Middle %", "Back %"]
            
            normalized_data = []
            for _, row in valid_df.iterrows():
                for metric in metrics_to_compare:
                    if metric in row and pd.notna(row[metric]):
                        normalized_data.append({
                            "Mode": row["Mode"],
                            "Metric": metric,
                            "Value": row[metric]
                        })
            
            norm_df = pd.DataFrame(normalized_data)
            
            if not norm_df.empty:
                fig = px.bar(
                    norm_df,
                    x="Metric",
                    y="Value",
                    color="Mode",
                    barmode="group",
                    color_discrete_map={m: MODES.get(m, {}).get("color", "#667eea") for m in valid_df["Mode"]}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.markdown("### 📊 Metric Correlations")
            
            if len(valid_df) >= 3:
                corr_cols = ["EV", "Scoop %", "Win 2/3 %", "Time (s)"]
                corr_data = valid_df[[c for c in corr_cols if c in valid_df.columns]].corr()
                
                fig = px.imshow(
                    corr_data,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== ARRANGEMENT DIFFERENCES =====
        if show_arrangement_diff:
            st.markdown("## 🔍 Arrangement Differences")
            
            comparisons = compare_arrangements_detailed(results)
            
            same_count = sum(1 for c in comparisons if c["same_arrangement"])
            diff_count = len(comparisons) - same_count
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Same Arrangements", f"{same_count}/{len(comparisons)} pairs")
            
            with col2:
                st.metric("Different Arrangements", f"{diff_count}/{len(comparisons)} pairs")
            
            if diff_count > 0:
                st.warning(f"⚠️ **{diff_count} mode pairs** produced different arrangements!")
                
                # Show differences
                for comp in comparisons:
                    if not comp["same_arrangement"]:
                        with st.expander(f"🔄 {comp['mode1']} vs {comp['mode2']}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("EV Difference", f"{comp['ev_diff']:+.3f}")
                            
                            with col2:
                                st.metric("Scoop % Diff", f"{comp['scoop_diff']:+.1f}%")
                            
                            with col3:
                                st.write("**Chi Differences:**")
                                st.write(f"- Front: {'✅ Same' if comp['same_front'] else '❌ Different'}")
                                st.write(f"- Middle: {'✅ Same' if comp['same_middle'] else '❌ Different'}")
                                st.write(f"- Back: {'✅ Same' if comp['same_back'] else '❌ Different'}")
            else:
                st.success("✅ **All modes produced the SAME arrangement!** This indicates high confidence in the solution.")
        
        st.markdown("---")
        
        # ===== DETAILED ARRANGEMENTS =====
        st.markdown("## 🎴 Detailed Arrangements by Mode")
        
        for r in results:
            if not r["success"]:
                with st.expander(f"❌ {r['mode_name']} - FAILED", expanded=False):
                    st.error(f"Error: {r['error']}")
                continue
            
            res = r["result"]
            mode_name = r["mode_name"]
            mode_info = r["mode_info"]
            
            # Evaluate hands
            back_eval = HandEvaluator.evaluate(res.back)
            middle_eval = HandEvaluator.evaluate(res.middle)
            front_eval = HandEvaluator.evaluate(res.front)
            
            with st.expander(
                f"{mode_name} | EV: {res.ev:+.3f} | Scoop: {res.p_scoop*100:.1f}% | ⏱️ {r['actual_time']:.2f}s",
                expanded=False
            ):
                # Visual cards
                if VISUAL_CARDS_AVAILABLE and show_cards_visual:
                    st.markdown(
                        render_comparison_cards(
                            [str(c) for c in res.back],
                            [str(c) for c in res.middle],
                            [str(c) for c in res.front],
                            (str(back_eval), str(middle_eval), str(front_eval))
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🔵 Back (5 cards)**")
                        st.code(Deck.cards_to_string(res.back))
                        st.info(str(back_eval))
                    
                    with col2:
                        st.markdown("**🟢 Middle (5 cards)**")
                        st.code(Deck.cards_to_string(res.middle))
                        st.info(str(middle_eval))
                    
                    with col3:
                        st.markdown("**🟡 Front (3 cards)**")
                        st.code(Deck.cards_to_string(res.front))
                        st.info(str(front_eval))
                
                # Metrics row
                st.markdown("##### 📊 Detailed Metrics")
                
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                mcol1.metric("EV", f"{res.ev:+.3f}")
                mcol2.metric("Bonus", f"+{res.bonus}")
                mcol3.metric("Scoop", f"{res.p_scoop*100:.1f}%")
                mcol4.metric("Win 2/3", f"{res.p_win_2_of_3*100:.1f}%")
                mcol5.metric("Arrangements", f"{res.num_arrangements_evaluated:,}")
                
                # Win rates
                wcol1, wcol2, wcol3 = st.columns(3)
                wcol1.metric("Front Win", f"{res.p_win_front*100:.1f}%")
                wcol2.metric("Middle Win", f"{res.p_win_middle*100:.1f}%")
                wcol3.metric("Back Win", f"{res.p_win_back*100:.1f}%")
        
        st.markdown("---")
        
        # ===== FINAL RECOMMENDATION =====
        st.markdown("## 💡 Final Recommendation")
        
        # Calculate scores
        scores = []
        for r in results:
            if not r["success"]:
                continue
            
            res = r["result"]
            category = r["mode_info"]["category"]
            
            # Weighted score
            ev_score = (res.ev + 2) / 4 * 35  # 0-35 points
            scoop_score = res.p_scoop * 25  # 0-25 points
            speed_score = max(0, (60 - r["actual_time"]) / 60 * 20)  # 0-20 points
            win23_score = res.p_win_2_of_3 * 20  # 0-20 points
            
            total = ev_score + scoop_score + speed_score + win23_score
            
            scores.append({
                "mode": r["mode_name"],
                "category": category,
                "total_score": total,
                "ev": res.ev,
                "time": r["actual_time"],
                "scoop": res.p_scoop
            })
        
        if scores:
            # Sort by score
            scores.sort(key=lambda x: x["total_score"], reverse=True)
            
            best_overall = scores[0]
            fastest = min(scores, key=lambda x: x["time"])
            best_ev = max(scores, key=lambda x: x["ev"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                    <h3 style="margin:0;">🏆 OVERALL BEST</h3>
                    <p style="font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0;">{best_overall['mode']}</p>
                    <p style="margin:0;">Score: {best_overall['total_score']:.0f}/100</p>
                    <p style="margin:0; font-size: 0.9rem;">EV: {best_overall['ev']:+.3f} | Time: {best_overall['time']:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                    <h3 style="margin:0;">⚡ FASTEST</h3>
                    <p style="font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0;">{fastest['mode']}</p>
                    <p style="margin:0;">Time: {fastest['time']:.3f}s</p>
                    <p style="margin:0; font-size: 0.9rem;">EV: {fastest['ev']:+.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                    <h3 style="margin:0;">💰 HIGHEST EV</h3>
                    <p style="font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0;">{best_ev['mode']}</p>
                    <p style="margin:0;">EV: {best_ev['ev']:+.3f}</p>
                    <p style="margin:0; font-size: 0.9rem;">Time: {best_ev['time']:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Usage recommendations table
            st.markdown("### 📝 When to Use Each Mode")
            
            rec_data = {
                "Situation": [
                    "🎮 Live game, time pressure",
                    "🏠 Practice, learning",
                    "💰 High stakes, important hand",
                    "🤖 Testing AI capabilities",
                    "🔬 Research, maximum accuracy",
                    "📊 Comparing strategies"
                ],
                "Recommended Mode": [
                    "⚡ Fast",
                    "⚖️ Balanced",
                    "🚀 Ultimate",
                    "🤖 ML Agent",
                    "🎯 Accurate or 🚀 Ultimate",
                    "🔄 Compare All (this page!)"
                ],
                "Why": [
                    "Instant results, good enough for quick decisions",
                    "Best trade-off between speed and quality",
                    "Combines all methods for best possible result",
                    "See how Deep Learning approaches the problem",
                    "Exhaustive search guarantees optimal solution",
                    "Understand trade-offs between methods"
                ]
            }
            
            st.table(pd.DataFrame(rec_data))
            
            # Key insights
            st.markdown("### 🔑 Key Insights from This Comparison")
            
            insights = []
            
            # Check if all modes agree
            unique_arrangements = len(set(
                get_arrangement_hash(r["result"]) 
                for r in results if r["success"]
            ))
            
            if unique_arrangements == 1:
                insights.append("✅ **High Confidence:** All modes agree on the same arrangement!")
            else:
                insights.append(f"⚠️ **{unique_arrangements} different arrangements** found - review each carefully")
            
            # ML vs Traditional insight
            ml_res = next((r for r in results if r["success"] and r["mode_info"]["category"] == "ml"), None)
            trad_res = [r for r in results if r["success"] and r["mode_info"]["category"] == "traditional"]
            
            if ml_res and trad_res:
                best_trad = max(trad_res, key=lambda x: x["result"].ev)
                ev_diff = ml_res["result"].ev - best_trad["result"].ev
                
                if abs(ev_diff) < 0.01:
                    insights.append("🤝 **ML matches Traditional:** AI learned well from expert data!")
                elif ev_diff > 0:
                    insights.append(f"🤖 **ML outperforms Traditional** by {ev_diff:.3f} EV - AI found better solution!")
                else:
                    insights.append(f"📐 **Traditional outperforms ML** by {-ev_diff:.3f} EV - rule-based wins here")
            
            # Speed insight
            if fastest["time"] < 1:
                insights.append(f"⚡ **{fastest['mode']} is blazing fast** at {fastest['time']*1000:.0f}ms!")
            
            # Ultimate insight
            ultimate_res = next((r for r in results if r["success"] and "Ultimate" in r["mode_name"]), None)
            if ultimate_res:
                if ultimate_res["result"].ev == best_ev["ev"]:
                    insights.append("🚀 **Ultimate achieved best EV** - worth the extra time for important hands!")
                else:
                    insights.append(f"🤔 **Ultimate EV ({ultimate_res['result'].ev:+.3f}) is not the best** - simpler modes suffice here")
            
            for insight in insights:
                st.markdown(f"- {insight}")
        
        # Export comparison
        st.markdown("---")
        st.markdown("### 📥 Export Comparison")
        
        export_text = f"""
MAUBINH AI SOLVER - MODE COMPARISON REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT CARDS:
{card_input}

RESULTS SUMMARY:
{'-'*60}
"""
        for r in results:
            if r["success"]:
                res = r["result"]
                export_text += f"""
{r['mode_name']}:
  EV: {res.ev:+.3f}
  Bonus: +{res.bonus}
  Scoop: {res.p_scoop*100:.1f}%
  Win 2/3: {res.p_win_2_of_3*100:.1f}%
  Time: {r['actual_time']:.2f}s
  Arrangements: {res.num_arrangements_evaluated:,}
  
  Back:   {Deck.cards_to_string(res.back)}
  Middle: {Deck.cards_to_string(res.middle)}
  Front:  {Deck.cards_to_string(res.front)}
"""
            else:
                export_text += f"\n{r['mode_name']}: FAILED - {r['error']}\n"
        
        export_text += f"""
{'='*60}
RECOMMENDATION: {best_overall['mode']} (Score: {best_overall['total_score']:.0f}/100)
"""
        
        st.download_button(
            "📥 Download Comparison Report",
            data=export_text,
            file_name=f"maubinh_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())

# ===== FOOTER =====
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Mode Categories")
    st.markdown("""
    - **Traditional:** Rule-based algorithms
    - **ML:** Deep Learning (DQN)
    - **Hybrid:** Combines multiple approaches
    """)

with col2:
    st.markdown("### ⏱️ Time Estimates")
    st.markdown("""
    - ⚡ Fast: < 1 second
    - ⚖️ Balanced: 2-5 seconds
    - 🎯 Accurate: 10-20 seconds
    - 🤖 ML Agent: 2-3 seconds
    - 🚀 Ultimate: 30-60 seconds
    """)

with col3:
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use **parallel execution** for faster comparison
    - **Ultimate** is best for important hands
    - **Fast** is good enough for practice
    """)

st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.9rem; margin-top: 2rem;">'
    '🔄 Compare All Modes | Part of Mau Binh AI Solver Pro'
    '</p>',
    unsafe_allow_html=True
)