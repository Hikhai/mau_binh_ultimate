"""
Mode Comparison Component - SIMPLIFIED
"""
import streamlit as st
import time


def run_comparison(cards, solver_class, mode_enum, evaluator, deck_class):
    """Run comparison and return results"""
    
    modes = [
        ('fast', '⚡ Fast'),
        ('balanced', '⚖️ Balanced'),
        ('accurate', '🎯 Accurate')
    ]
    
    results = {}
    
    for mode_key, mode_name in modes:
        st.write(f"🔄 Computing {mode_name}...")
        
        try:
            mode = mode_enum(mode_key)
            solver = solver_class(cards, mode=mode, verbose=False)
            
            start = time.time()
            result = solver.solve()
            elapsed = time.time() - start
            
            results[mode_key] = {
                'name': mode_name,
                'back': deck_class.cards_to_string(result.back),
                'middle': deck_class.cards_to_string(result.middle),
                'front': deck_class.cards_to_string(result.front),
                'back_eval': str(evaluator.evaluate(result.back)),
                'middle_eval': str(evaluator.evaluate(result.middle)),
                'front_eval': str(evaluator.evaluate(result.front)),
                'ev': result.ev,
                'bonus': result.bonus,
                'p_scoop': result.p_scoop,
                'p_win_2_of_3': result.p_win_2_of_3,
                'p_win_front': result.p_win_front,
                'p_win_middle': result.p_win_middle,
                'p_win_back': result.p_win_back,
                'time': elapsed,
                'success': True
            }
            st.write(f"✅ {mode_name} done! EV: {result.ev:+.2f}")
            
        except Exception as e:
            results[mode_key] = {
                'name': mode_name,
                'error': str(e),
                'success': False
            }
            st.write(f"❌ {mode_name} failed: {e}")
    
    return results


def display_results(results):
    """Display comparison results"""
    
    st.markdown("### 📊 Comparison Results")
    
    # Check success
    if not all(r.get('success', False) for r in results.values()):
        st.error("Some modes failed!")
        return
    
    # Summary table header
    st.markdown("#### Quick Summary")
    
    header_cols = st.columns([2, 1, 1, 1, 1, 1])
    header_cols[0].markdown("**Mode**")
    header_cols[1].markdown("**EV**")
    header_cols[2].markdown("**Bonus**")
    header_cols[3].markdown("**Scoop**")
    header_cols[4].markdown("**Win 2/3**")
    header_cols[5].markdown("**Time**")
    
    # Data rows
    for mode_key in ['fast', 'balanced', 'accurate']:
        r = results[mode_key]
        cols = st.columns([2, 1, 1, 1, 1, 1])
        cols[0].write(r['name'])
        cols[1].write(f"{r['ev']:+.2f}")
        cols[2].write(f"+{r['bonus']}")
        cols[3].write(f"{r['p_scoop']*100:.1f}%")
        cols[4].write(f"{r['p_win_2_of_3']*100:.1f}%")
        cols[5].write(f"{r['time']:.2f}s")
    
    # Best mode
    best = max(results.items(), key=lambda x: x[1].get('ev', -999))
    st.success(f"🏆 **Best:** {best[1]['name']} with EV = {best[1]['ev']:+.2f}")
    
    # Detailed tabs
    st.markdown("---")
    st.markdown("#### Detailed View")
    
    tabs = st.tabs(["⚡ Fast", "⚖️ Balanced", "🎯 Accurate"])
    
    for idx, mode_key in enumerate(['fast', 'balanced', 'accurate']):
        with tabs[idx]:
            r = results[mode_key]
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Back**")
                st.code(r['back'])
                st.caption(r['back_eval'])
            with c2:
                st.markdown("**Middle**")
                st.code(r['middle'])
                st.caption(r['middle_eval'])
            with c3:
                st.markdown("**Front**")
                st.code(r['front'])
                st.caption(r['front_eval'])


def show_comparison_ui(cards, solver_class, mode_enum, evaluator, deck_class, card_input_str):
    """Main comparison UI"""
    
    # Simple fixed key
    key = "mode_comparison_results"
    
    st.markdown("""
    **Compare Fast vs Balanced vs Accurate:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run Comparison", type="primary", use_container_width=True, key="compare_btn"):
            # Run and save immediately
            st.session_state[key] = run_comparison(
                cards, solver_class, mode_enum, evaluator, deck_class
            )
    
    with col2:
        if key in st.session_state:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
                del st.session_state[key]
    
    # Display if we have results
    if key in st.session_state and st.session_state[key]:
        st.markdown("---")
        display_results(st.session_state[key])