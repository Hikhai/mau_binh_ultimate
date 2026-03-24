"""
Mode Comparison Component - Reusable component for comparing solver modes
Can be embedded in main app or used standalone
"""
import streamlit as st
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ModeResult:
    """Result from a single mode"""
    mode_name: str
    mode_value: str
    color: str
    result: Any  # SolverResult
    computation_time: float
    success: bool
    error: Optional[str] = None


class ModeComparisonEngine:
    """Engine for comparing multiple solver modes"""
    
    MODES_CONFIG = {
        "⚡ Fast": {
            "value": "fast",
            "color": "#38ef7d",
            "desc": "Quick decisions",
            "time_estimate": "< 1s"
        },
        "⚖️ Balanced": {
            "value": "balanced",
            "color": "#667eea",
            "desc": "Best trade-off",
            "time_estimate": "2-5s"
        },
        "🎯 Accurate": {
            "value": "accurate",
            "color": "#f39c12",
            "desc": "High accuracy",
            "time_estimate": "10-20s"
        },
        "🤖 ML Agent": {
            "value": "ml_only",
            "color": "#9b59b6",
            "desc": "Deep Learning",
            "time_estimate": "2-3s"
        },
        "🚀 Ultimate": {
            "value": "ultimate",
            "color": "#e74c3c",
            "desc": "Best solution",
            "time_estimate": "30-60s"
        }
    }
    
    def __init__(self, cards: List, parallel: bool = True):
        self.cards = cards
        self.parallel = parallel
        self.results: List[ModeResult] = []
    
    def _solve_single(self, mode_name: str, config: Dict) -> ModeResult:
        """Solve with a single mode"""
        from ultimate_solver import UltimateSolver, SolverMode
        
        try:
            start = time.time()
            solver = UltimateSolver(
                self.cards.copy(),
                mode=SolverMode(config["value"]),
                verbose=False
            )
            result = solver.solve()
            elapsed = time.time() - start
            
            return ModeResult(
                mode_name=mode_name,
                mode_value=config["value"],
                color=config["color"],
                result=result,
                computation_time=elapsed,
                success=True
            )
        except Exception as e:
            return ModeResult(
                mode_name=mode_name,
                mode_value=config["value"],
                color=config["color"],
                result=None,
                computation_time=0,
                success=False,
                error=str(e)
            )
    
    def compare(self, modes: List[str] = None, progress_callback=None) -> List[ModeResult]:
        """
        Compare multiple modes
        
        Args:
            modes: List of mode names to compare. If None, compare all.
            progress_callback: Optional callback(completed, total, mode_name)
        
        Returns:
            List of ModeResult
        """
        if modes is None:
            modes = list(self.MODES_CONFIG.keys())
        
        self.results = []
        total = len(modes)
        
        if self.parallel and total > 1:
            with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
                futures = {
                    executor.submit(
                        self._solve_single,
                        mode_name,
                        self.MODES_CONFIG[mode_name]
                    ): mode_name
                    for mode_name in modes
                }
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total, result.mode_name)
        else:
            for i, mode_name in enumerate(modes):
                result = self._solve_single(mode_name, self.MODES_CONFIG[mode_name])
                self.results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, mode_name)
        
        # Sort by mode order
        mode_order = list(self.MODES_CONFIG.keys())
        self.results.sort(
            key=lambda x: mode_order.index(x.mode_name) if x.mode_name in mode_order else 999
        )
        
        return self.results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        
        for r in self.results:
            if r.success:
                res = r.result
                data.append({
                    "Mode": r.mode_name,
                    "EV": res.ev,
                    "Bonus": res.bonus,
                    "Scoop %": res.p_scoop * 100,
                    "Win 2/3 %": res.p_win_2_of_3 * 100,
                    "Front Win %": res.p_win_front * 100,
                    "Middle Win %": res.p_win_middle * 100,
                    "Back Win %": res.p_win_back * 100,
                    "Time (s)": r.computation_time,
                    "Arrangements": res.num_arrangements_evaluated,
                    "Color": r.color
                })
            else:
                data.append({
                    "Mode": r.mode_name,
                    "Error": r.error,
                    "Color": r.color
                })
        
        return pd.DataFrame(data)
    
    def get_best(self, metric: str = "ev") -> Optional[ModeResult]:
        """Get best result by metric"""
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return None
        
        if metric == "ev":
            return max(successful, key=lambda r: r.result.ev)
        elif metric == "scoop":
            return max(successful, key=lambda r: r.result.p_scoop)
        elif metric == "speed":
            return min(successful, key=lambda r: r.computation_time)
        elif metric == "win_2_3":
            return max(successful, key=lambda r: r.result.p_win_2_of_3)
        else:
            return successful[0]
    
    def find_differences(self) -> List[Dict]:
        """Find differences between mode arrangements"""
        differences = []
        successful = [r for r in self.results if r.success]
        
        for i, r1 in enumerate(successful):
            for j, r2 in enumerate(successful):
                if i >= j:
                    continue
                
                same_front = set(str(c) for c in r1.result.front) == set(str(c) for c in r2.result.front)
                same_middle = set(str(c) for c in r1.result.middle) == set(str(c) for c in r2.result.middle)
                same_back = set(str(c) for c in r1.result.back) == set(str(c) for c in r2.result.back)
                
                if not (same_front and same_middle and same_back):
                    differences.append({
                        "mode1": r1.mode_name,
                        "mode2": r2.mode_name,
                        "ev_diff": r1.result.ev - r2.result.ev,
                        "same_front": same_front,
                        "same_middle": same_middle,
                        "same_back": same_back
                    })
        
        return differences


def render_comparison_summary(engine: ModeComparisonEngine):
    """Render comparison summary in Streamlit"""
    df = engine.to_dataframe()
    
    if df.empty:
        st.warning("No results to display")
        return
    
    # Best results
    best_ev = engine.get_best("ev")
    best_scoop = engine.get_best("scoop")
    fastest = engine.get_best("speed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if best_ev:
            st.metric(
                "💰 Best EV",
                f"{best_ev.result.ev:+.3f}",
                delta=best_ev.mode_name
            )
    
    with col2:
        if best_scoop:
            st.metric(
                "🎯 Best Scoop",
                f"{best_scoop.result.p_scoop*100:.1f}%",
                delta=best_scoop.mode_name
            )
    
    with col3:
        if fastest:
            st.metric(
                "⚡ Fastest",
                f"{fastest.computation_time:.2f}s",
                delta=fastest.mode_name
            )
    
    # Table
    display_cols = ["Mode", "EV", "Bonus", "Scoop %", "Win 2/3 %", "Time (s)"]
    available_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[available_cols].style.format({
            "EV": "{:+.3f}",
            "Bonus": "{:.0f}",
            "Scoop %": "{:.1f}%",
            "Win 2/3 %": "{:.1f}%",
            "Time (s)": "{:.3f}s"
        }, na_rep="-"),
        use_container_width=True
    )
    
    # Differences
    diffs = engine.find_differences()
    if diffs:
        st.warning(f"⚠️ {len(diffs)} mode pairs have different arrangements!")
    else:
        st.success("✅ All modes produced the same arrangement!")


def quick_compare_widget(cards_input: str, modes: List[str] = None):
    """
    Quick comparison widget that can be embedded anywhere
    
    Usage:
        from mode_comparison import quick_compare_widget
        quick_compare_widget("AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S")
    """
    from card import Deck
    
    try:
        cards = Deck.parse_hand(cards_input)
        
        if len(cards) != 13:
            st.error(f"Need 13 cards, got {len(cards)}")
            return
        
        if modes is None:
            modes = ["⚡ Fast", "⚖️ Balanced", "🎯 Accurate"]
        
        engine = ModeComparisonEngine(cards, parallel=True)
        
        with st.spinner("Comparing modes..."):
            engine.compare(modes)
        
        render_comparison_summary(engine)
        
        return engine
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    st.title("Mode Comparison Test")
    
    cards_input = st.text_input(
        "Enter 13 cards",
        value="AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"
    )
    
    if st.button("Compare"):
        quick_compare_widget(cards_input)