"""
Risk Analyzer - Phân tích rủi ro chi tiết
"""
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, '../core')
from card import Card, Deck
from evaluator import HandEvaluator

from game_theory import GameTheoryEngine, EVResult


@dataclass
class RiskMetrics:
    """Các chỉ số rủi ro"""
    # Volatility
    std_dev: float              # Độ lệch chuẩn
    variance: float             # Phương sai
    
    # Downside risk
    downside_deviation: float   # Độ lệch chuẩn downside
    max_loss: float            # Thua tối đa có thể
    
    # Risk-adjusted returns
    sharpe_ratio: float        # Sharpe ratio
    sortino_ratio: float       # Sortino ratio (dùng downside dev)
    
    # Percentiles
    percentile_5: float        # 5th percentile outcome
    percentile_25: float       # 25th percentile
    percentile_75: float       # 75th percentile
    percentile_95: float       # 95th percentile
    
    def __str__(self):
        return f"""
Risk Metrics:
─────────────────────────────────────────
Volatility:
  • Standard Deviation: {self.std_dev:.3f}
  • Variance:           {self.variance:.3f}

Downside Risk:
  • Downside Deviation: {self.downside_deviation:.3f}
  • Max Loss:           {self.max_loss:.3f}

Risk-Adjusted Returns:
  • Sharpe Ratio:       {self.sharpe_ratio:.3f}
  • Sortino Ratio:      {self.sortino_ratio:.3f}

Outcome Percentiles:
  •  5th percentile:    {self.percentile_5:+.3f}
  • 25th percentile:    {self.percentile_25:+.3f}
  • 75th percentile:    {self.percentile_75:+.3f}
  • 95th percentile:    {self.percentile_95:+.3f}
"""


class RiskAnalyzer:
    """Phân tích rủi ro chi tiết"""
    
    @staticmethod
    def calculate_risk_metrics(ev_result: EVResult) -> RiskMetrics:
        """
        Tính các chỉ số rủi ro từ EVResult
        """
        # Tạo phân phối outcomes
        outcomes = [
            (ev_result.p_win_3_0, ev_result.payoff_3_0),
            (ev_result.p_win_2_1, ev_result.payoff_2_1),
            (ev_result.p_lose_1_2, ev_result.payoff_1_2),
            (ev_result.p_lose_0_3, ev_result.payoff_0_3),
        ]
        
        # Tính EV (mean)
        mean = ev_result.ev_no_bonus
        
        # Tính variance và std_dev
        variance = sum(p * (payoff - mean)**2 for p, payoff in outcomes)
        std_dev = np.sqrt(variance)
        
        # Downside deviation (chỉ tính outcomes < mean)
        downside_outcomes = [(p, payoff) for p, payoff in outcomes if payoff < mean]
        if downside_outcomes:
            downside_variance = sum(
                p * (payoff - mean)**2 
                for p, payoff in downside_outcomes
            )
            downside_deviation = np.sqrt(downside_variance)
        else:
            downside_deviation = 0
        
        # Max loss
        max_loss = min(payoff for _, payoff in outcomes)
        
        # Sharpe ratio
        sharpe = ev_result.sharpe_ratio
        
        # Sortino ratio (dùng downside deviation thay vì std dev)
        if downside_deviation > 0:
            sortino = mean / downside_deviation
        else:
            sortino = float('inf') if mean > 0 else 0
        
        # Percentiles (approximate từ outcomes)
        # Sắp xếp outcomes theo payoff
        sorted_outcomes = sorted(outcomes, key=lambda x: x[1])
        
        # Cumulative probability
        cumulative_p = 0
        percentiles = {5: None, 25: None, 75: None, 95: None}
        
        for p, payoff in sorted_outcomes:
            cumulative_p += p
            
            for pct in [5, 25, 75, 95]:
                if percentiles[pct] is None and cumulative_p >= pct/100:
                    percentiles[pct] = payoff
        
        # Fill missing percentiles
        for pct in percentiles:
            if percentiles[pct] is None:
                percentiles[pct] = sorted_outcomes[-1][1]
        
        return RiskMetrics(
            std_dev=std_dev,
            variance=variance,
            downside_deviation=downside_deviation,
            max_loss=max_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            percentile_5=percentiles[5],
            percentile_25=percentiles[25],
            percentile_75=percentiles[75],
            percentile_95=percentiles[95]
        )
    
    @staticmethod
    def compare_risk_profiles(
        ev_results: List[Tuple[str, EVResult]]
    ) -> str:
        """
        So sánh risk profile của nhiều arrangements
        
        Args:
            ev_results: List of (name, EVResult)
        """
        report = "\n" + "="*70 + "\n"
        report += "RISK PROFILE COMPARISON\n"
        report += "="*70 + "\n\n"
        
        for name, ev_result in ev_results:
            report += f"📊 {name}\n"
            report += "─"*70 + "\n"
            
            metrics = RiskAnalyzer.calculate_risk_metrics(ev_result)
            report += str(metrics)
            report += "\n"
        
        return report


# ==================== TESTS ====================

def test_risk_analyzer():
    """Test RiskAnalyzer"""
    print("Testing RiskAnalyzer...")
    
    # Create mock EVResult
    ev_result = EVResult(
        ev=0.5,
        ev_no_bonus=0.3,
        bonus=2,
        p_win_3_0=0.2,
        p_win_2_1=0.4,
        p_lose_1_2=0.3,
        p_lose_0_3=0.1,
        payoff_3_0=3.0,
        payoff_2_1=1.0,
        payoff_1_2=-1.0,
        payoff_0_3=-3.0,
        risk=0.1,
        upside=0.2,
        sharpe_ratio=5.0
    )
    
    metrics = RiskAnalyzer.calculate_risk_metrics(ev_result)
    print(metrics)
    
    assert metrics.std_dev > 0
    assert metrics.max_loss == -3.0
    assert metrics.sharpe_ratio == 5.0
    
    print("✅ RiskAnalyzer tests passed!")


if __name__ == "__main__":
    test_risk_analyzer()
    print("\n✅ All risk_analyzer.py tests passed!")