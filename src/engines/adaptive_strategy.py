"""
Adaptive Strategy - Điều chỉnh chiến thuật theo ngữ cảnh
"""
import sys
from typing import Dict, Optional
from enum import Enum

sys.path.insert(0, '../core')
from multi_objective import ObjectiveWeights


class GameStage(Enum):
    """Giai đoạn game"""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class PlayerStyle(Enum):
    """Style chơi của đối thủ"""
    TIGHT = "tight"          # Chơi chặt, ít mạo hiểm
    LOOSE = "loose"          # Chơi lỏng, nhiều bài
    AGGRESSIVE = "aggressive"  # Chơi hung hăng
    PASSIVE = "passive"      # Chơi thụ động


class StackSize(Enum):
    """Kích thước stack"""
    SHORT = "short"    # < 30% chip ban đầu
    MEDIUM = "medium"  # 30-70%
    DEEP = "deep"      # > 70%


class GameContext:
    """Ngữ cảnh game"""
    
    def __init__(
        self,
        stage: GameStage = GameStage.MIDDLE,
        stack_size: StackSize = StackSize.MEDIUM,
        opponent_style: PlayerStyle = PlayerStyle.TIGHT,
        num_opponents: int = 3,
        position: Optional[str] = None
    ):
        self.stage = stage
        self.stack_size = stack_size
        self.opponent_style = opponent_style
        self.num_opponents = num_opponents
        self.position = position  # 'early', 'middle', 'late' position at table


class AdaptiveStrategySelector:
    """
    Chọn strategy (weights) dựa trên game context
    """
    
    @staticmethod
    def select_weights(context: GameContext) -> ObjectiveWeights:
        """
        Chọn objective weights dựa trên context
        
        Strategy matrix:
        ┌───────────┬──────────┬──────────┬──────────┐
        │ Context   │ EV       │ Scoop    │ Balance  │
        ├───────────┼──────────┼──────────┼──────────┤
        │ Early     │ Medium   │ Low      │ High     │
        │ Mid       │ High     │ Medium   │ Medium   │
        │ Late      │ Medium   │ High     │ Low      │
        ├───────────┼──────────┼──────────┼──────────┤
        │ Short $   │ Low      │ High     │ Low      │
        │ Deep $    │ High     │ Medium   │ High     │
        ├───────────┼──────────┼──────────┼──────────┤
        │ vs Tight  │ High     │ High     │ Medium   │
        │ vs Loose  │ High     │ Low      │ High     │
        └───────────┴──────────┴──────────┴──────────┘
        """
        
        # Base weights
        weights = ObjectiveWeights()
        
        # Adjust by game stage
        if context.stage == GameStage.EARLY:
            # Early: Chơi an toàn, cân bằng
            weights.ev = 0.25
            weights.scoop = 0.15
            weights.bonus = 0.15
            weights.front_strength = 0.25
            weights.balance = 0.20
        
        elif context.stage == GameStage.MIDDLE:
            # Middle: Standard play
            weights.ev = 0.30
            weights.scoop = 0.25
            weights.bonus = 0.15
            weights.front_strength = 0.20
            weights.balance = 0.10
        
        elif context.stage == GameStage.LATE:
            # Late: Aggressive, tìm scoop
            weights.ev = 0.20
            weights.scoop = 0.40
            weights.bonus = 0.20
            weights.front_strength = 0.15
            weights.balance = 0.05
        
        # Adjust by stack size
        if context.stack_size == StackSize.SHORT:
            # Short stack: Phải all-in vào potential cao
            weights.scoop += 0.15
            weights.bonus += 0.10
            weights.ev -= 0.10
            weights.balance -= 0.15
        
        elif context.stack_size == StackSize.DEEP:
            # Deep stack: Chơi patient, EV-focused
            weights.ev += 0.10
            weights.balance += 0.10
            weights.scoop -= 0.10
            weights.bonus -= 0.10
        
        # Adjust by opponent style
        if context.opponent_style == PlayerStyle.TIGHT:
            # Vs tight: Aggressive để pressure
            weights.scoop += 0.10
            weights.front_strength += 0.10
            weights.balance -= 0.10
        
        elif context.opponent_style == PlayerStyle.LOOSE:
            # Vs loose: Solid play, tận dụng mistakes
            weights.ev += 0.15
            weights.balance += 0.10
            weights.scoop -= 0.15
        
        elif context.opponent_style == PlayerStyle.AGGRESSIVE:
            # Vs aggressive: Patient, wait for premium
            weights.balance += 0.15
            weights.ev += 0.10
            weights.scoop -= 0.15
        
        # Normalize
        weights.normalize()
        
        return weights
    
    @staticmethod
    def get_strategy_explanation(context: GameContext) -> str:
        """
        Giải thích strategy được chọn
        """
        weights = AdaptiveStrategySelector.select_weights(context)
        
        explanation = f"""
╔════════════════════════════════════════════════════════════╗
║  ADAPTIVE STRATEGY SELECTION                               ║
╠════════════════════════════════════════════════════════════╣
║  Game Context:                                             ║
║    • Stage:          {context.stage.value:15s}                  ║
║    • Stack Size:     {context.stack_size.value:15s}             ║
║    • Opponent Style: {context.opponent_style.value:15s}         ║
║    • # Opponents:    {context.num_opponents:15d}                ║
╠════════════════════════════════════════════════════════════╣
{weights}╠════════════════════════════════════════════════════════════╣
║  Strategic Reasoning:                                      ║
"""
        
        # Add reasoning based on context
        if context.stage == GameStage.EARLY:
            explanation += "║    → Early game: Focus on balance and safety          ║\n"
        elif context.stage == GameStage.LATE:
            explanation += "║    → Late game: Aggressive, maximize scoop chance     ║\n"
        
        if context.stack_size == StackSize.SHORT:
            explanation += "║    → Short stack: Need high variance plays            ║\n"
        elif context.stack_size == StackSize.DEEP:
            explanation += "║    → Deep stack: Patient EV-maximizing play           ║\n"
        
        if context.opponent_style == PlayerStyle.TIGHT:
            explanation += "║    → Vs tight players: Apply pressure with aggression ║\n"
        elif context.opponent_style == PlayerStyle.LOOSE:
            explanation += "║    → Vs loose players: Solid value-oriented play      ║\n"
        
        explanation += "╚════════════════════════════════════════════════════════════╝\n"
        
        return explanation


# ==================== TESTS ====================

def test_adaptive_strategy():
    """Test adaptive strategy selection"""
    print("Testing Adaptive Strategy Selection...")
    
    # Test case 1: Early game, medium stack
    print("\n" + "="*60)
    print("TEST 1: Early game, medium stack, vs tight opponent")
    context1 = GameContext(
        stage=GameStage.EARLY,
        stack_size=StackSize.MEDIUM,
        opponent_style=PlayerStyle.TIGHT
    )
    
    explanation1 = AdaptiveStrategySelector.get_strategy_explanation(context1)
    print(explanation1)
    
    weights1 = AdaptiveStrategySelector.select_weights(context1)
    assert abs(sum([weights1.ev, weights1.scoop, weights1.bonus, 
                    weights1.front_strength, weights1.balance]) - 1.0) < 0.01
    
    # Test case 2: Late game, short stack
    print("\n" + "="*60)
    print("TEST 2: Late game, short stack, vs loose opponent")
    context2 = GameContext(
        stage=GameStage.LATE,
        stack_size=StackSize.SHORT,
        opponent_style=PlayerStyle.LOOSE
    )
    
    explanation2 = AdaptiveStrategySelector.get_strategy_explanation(context2)
    print(explanation2)
    
    weights2 = AdaptiveStrategySelector.select_weights(context2)
    
    # Late + short stack should have high scoop weight
    assert weights2.scoop > 0.3, f"Scoop weight should be high, got {weights2.scoop}"
    
    # Test case 3: Deep stack vs aggressive
    print("\n" + "="*60)
    print("TEST 3: Middle game, deep stack, vs aggressive opponent")
    context3 = GameContext(
        stage=GameStage.MIDDLE,
        stack_size=StackSize.DEEP,
        opponent_style=PlayerStyle.AGGRESSIVE
    )
    
    explanation3 = AdaptiveStrategySelector.get_strategy_explanation(context3)
    print(explanation3)
    
    weights3 = AdaptiveStrategySelector.select_weights(context3)
    
    # Deep stack should have high EV and balance
    assert weights3.ev > 0.25, f"EV weight should be high, got {weights3.ev}"
    
    print("\n✅ Adaptive strategy tests passed!")


if __name__ == "__main__":
    test_adaptive_strategy()
    print("\n" + "="*60)
    print("✅ All adaptive_strategy.py tests passed!")
    print("="*60)