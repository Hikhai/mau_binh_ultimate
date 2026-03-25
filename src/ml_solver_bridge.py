"""
ML Solver Bridge V2.0 - Bridge to ML Agent v2.0
Kết nối UltimateSolver với ML Agent mới
"""
import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'core'))
sys.path.insert(0, os.path.join(current_dir, 'ml'))

from card import Card

# Import ML Agent
ML_AGENT_AVAILABLE = False

try:
    from ml.agent import MauBinhAgent, BeamSearch
    from ml.core import RewardCalculator
    ML_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Agent import error: {e}")
except Exception as e:
    print(f"⚠️ ML Agent error: {e}")


class MLSolverBridge:
    """
    Bridge between UltimateSolver and ML Agent v2.0
    
    Features:
    - Auto-detect model file
    - Multiple solving modes (best, fast, beam)
    - Fallback to traditional methods
    - Caching for performance
    """
    
    # Default model paths to search
    DEFAULT_MODEL_PATHS = [
        # Relative to src/
        "data/models/production_v1/best_model.pth",
        "data/models/maubinh_v3/best_model.pth",
        # Relative to project root
        "../data/models/production_v1/best_model.pth",
        "../data/models/maubinh_v3/best_model.pth",
        "../../data/models/production_v1/best_model.pth",
        # With timestamps
        "../data/models/maubinh_v3_20260325_211706/best_model.pth",
        "data/models/maubinh_v3_20260325_211706/best_model.pth",
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML Bridge
        
        Args:
            model_path: Path to model file. If None, auto-detect.
        """
        self.agent = None
        self.beam_searcher = None
        self.is_loaded = False
        self.model_path = None
        self.reward_calc = None
        
        if not ML_AGENT_AVAILABLE:
            print("⚠️ ML Agent module not available")
            return
        
        # Initialize reward calculator
        try:
            self.reward_calc = RewardCalculator()
        except:
            pass
        
        # Find and load model
        if model_path:
            self._load_model(model_path)
        else:
            self._auto_load_model()
    
    def _auto_load_model(self):
        """Auto-detect and load model from default paths"""
        for path in self.DEFAULT_MODEL_PATHS:
            # Try relative to current dir
            full_path = os.path.join(current_dir, path)
            if Path(full_path).exists():
                self._load_model(full_path)
                if self.is_loaded:
                    return
            
            # Try as absolute path
            if Path(path).exists():
                self._load_model(path)
                if self.is_loaded:
                    return
            
            # Try from project root
            project_root = os.path.dirname(current_dir)
            root_path = os.path.join(project_root, path)
            if Path(root_path).exists():
                self._load_model(root_path)
                if self.is_loaded:
                    return
        
        print("⚠️ No ML model found in default paths. ML modes will use fallback.")
    
    def _load_model(self, model_path: str):
        """Load model from path"""
        try:
            print(f"🔄 Loading ML model from: {model_path}")
            self.agent = MauBinhAgent(model_path=model_path, device='cpu')
            self.model_path = model_path
            self.is_loaded = True
            print(f"✅ ML Model V2 loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            self.is_loaded = False
    
    def load_model(self, model_path: str) -> Tuple[bool, str]:
        """
        Load model manually
        
        Returns:
            (success, message)
        """
        if not ML_AGENT_AVAILABLE:
            return False, "ML Agent module not available"
        
        if not Path(model_path).exists():
            return False, f"Model file not found: {model_path}"
        
        try:
            self.agent = MauBinhAgent(model_path=model_path, device='cpu')
            self.model_path = model_path
            self.is_loaded = True
            return True, f"Model loaded: {model_path}"
        except Exception as e:
            self.is_loaded = False
            return False, f"Failed to load: {e}"
    
    def solve(
        self,
        cards: List[Card],
        mode: str = 'best'
    ) -> Tuple[Optional[List[Card]], Optional[List[Card]], Optional[List[Card]], dict]:
        """
        Solve using ML Agent
        
        Args:
            cards: 13 cards
            mode: 
                - 'best' or 'ensemble': Use ensemble (highest quality)
                - 'fast' or 'dqn': Use DQN only (faster)
                - 'beam': Use beam search (thorough)
            
        Returns:
            (back, middle, front, metrics)
            metrics = {
                'reward': float,
                'bonus': int,
                'strength': float,
                'is_valid': bool,
                'mode': str,
                'error': str (if failed)
            }
        """
        if not self.is_loaded or self.agent is None:
            return None, None, None, {
                'error': 'Model not loaded',
                'reward': 0,
                'bonus': 0,
                'is_valid': False
            }
        
        try:
            # Solve based on mode
            if mode in ['best', 'ensemble']:
                back, middle, front = self.agent.solve(cards, mode='ensemble')
            
            elif mode in ['fast', 'dqn']:
                back, middle, front = self.agent.solve(cards, mode='best')
            
            elif mode == 'beam':
                if self.beam_searcher is None:
                    self.beam_searcher = BeamSearch(beam_width=10)
                back, middle, front = self.beam_searcher.search(cards, depth=2)
            
            else:
                # Default to best
                back, middle, front = self.agent.solve(cards, mode='best')
            
            # Evaluate arrangement
            eval_result = self.agent.evaluate_arrangement((back, middle, front))
            
            metrics = {
                'reward': eval_result.get('reward', 0),
                'bonus': eval_result.get('bonus', 0),
                'strength': eval_result.get('strength', 0),
                'is_valid': eval_result.get('is_valid', False),
                'mode': mode
            }
            
            return back, middle, front, metrics
        
        except Exception as e:
            import traceback
            return None, None, None, {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'reward': 0,
                'bonus': 0,
                'is_valid': False
            }
    def solve_hybrid(
        self,
        cards: List[Card],
        smart_solver=None
    ) -> Tuple[Optional[List[Card]], Optional[List[Card]], Optional[List[Card]], dict]:
        """
        HYBRID: SmartSolver tìm candidates, ML Agent chấm điểm
        
        Best of both worlds!
        """
        # Import SmartSolver
        if smart_solver is None:
            try:
                from smart_solver import SmartSolver
                smart_solver = SmartSolver()
            except:
                # Fallback to pure ML
                return self.solve(cards, mode='best')
        
        # Step 1: SmartSolver tìm TOP 10 arrangements
        try:
            smart_results = smart_solver.find_best_arrangement(cards, top_k=10)
        except:
            return self.solve(cards, mode='best')
        
        if not smart_results or smart_results[0][0] is None:
            return self.solve(cards, mode='best')
        
        # Step 2: ML Agent chấm điểm từng arrangement
        best_arr = None
        best_score = -float('inf')
        best_metrics = {}
        
        for back, middle, front, smart_score in smart_results:
            if back is None:
                continue
            
            # ML evaluation
            if self.is_loaded and self.agent:
                ml_eval = self.agent.evaluate_arrangement((back, middle, front))
                ml_reward = ml_eval.get('reward', 0)
            else:
                ml_reward = 0
            
            # RewardCalculator evaluation
            if self.reward_calc:
                calc_reward = self.reward_calc.calculate_reward(back, middle, front)
            else:
                calc_reward = 0
            
            # Combined score:
            # - SmartSolver score: cách xếp cơ bản
            # - RewardCalculator: bonus + strength
            # - ML Agent: learned patterns
            combined_score = (
                smart_score * 0.3 +      # SmartSolver weight
                calc_reward * 0.5 +       # RewardCalculator weight (bonus-aware!)
                ml_reward * 0.2           # ML Agent weight
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_arr = (back, middle, front)
                best_metrics = {
                    'reward': calc_reward,
                    'bonus': self.reward_calc._calculate_bonus(back, middle, front) if self.reward_calc and calc_reward > -50 else 0,
                    'strength': self.reward_calc._calculate_strength(back, middle, front) if self.reward_calc and calc_reward > -50 else 0,
                    'is_valid': calc_reward > -50,
                    'smart_score': smart_score,
                    'ml_reward': ml_reward,
                    'combined_score': combined_score,
                    'mode': 'hybrid',
                    'num_candidates': len(smart_results)
                }
        
        if best_arr:
            return best_arr[0], best_arr[1], best_arr[2], best_metrics
        else:
            return None, None, None, {'error': 'No valid arrangement found'}
    def evaluate(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> dict:
        """
        Evaluate an arrangement
        
        Returns:
            {
                'reward': float,
                'bonus': int,
                'strength': float,
                'is_valid': bool
            }
        """
        if not self.is_loaded or self.agent is None:
            # Fallback to reward calculator if available
            if self.reward_calc:
                try:
                    reward = self.reward_calc.calculate_reward(back, middle, front)
                    bonus = self.reward_calc._calculate_bonus(back, middle, front) if reward > -50 else 0
                    strength = self.reward_calc._calculate_strength(back, middle, front) if reward > -50 else 0
                    
                    return {
                        'reward': reward,
                        'bonus': bonus,
                        'strength': strength,
                        'is_valid': reward > -50
                    }
                except:
                    pass
            
            return {
                'reward': 0,
                'bonus': 0,
                'strength': 0,
                'is_valid': False,
                'error': 'Model not loaded'
            }
        
        try:
            return self.agent.evaluate_arrangement((back, middle, front))
        except Exception as e:
            return {
                'reward': 0,
                'bonus': 0,
                'strength': 0,
                'is_valid': False,
                'error': str(e)
            }
    
    def get_status(self) -> dict:
        """Get current status"""
        return {
            'ml_available': ML_AGENT_AVAILABLE,
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
        }


# Singleton instance
_bridge_instance = None

def get_ml_bridge() -> MLSolverBridge:
    """Get singleton instance of MLSolverBridge"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MLSolverBridge()
    return _bridge_instance


# ==================== TEST ====================

def test_ml_bridge():
    """Test MLSolverBridge"""
    print("="*60)
    print("Testing MLSolverBridge...")
    print("="*60)
    
    bridge = MLSolverBridge()
    
    print(f"\n📊 Status:")
    print(f"  ML Module Available: {ML_AGENT_AVAILABLE}")
    print(f"  Model Loaded: {bridge.is_loaded}")
    print(f"  Model Path: {bridge.model_path}")
    
    if bridge.is_loaded:
        print(f"\n🧪 Testing solve...")
        from card import Deck
        
        cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
        
        # Test best mode
        print("\n  Testing mode='best'...")
        back, middle, front, metrics = bridge.solve(cards, mode='best')
        
        if back:
            print(f"    ✅ Solve OK")
            print(f"    Reward: {metrics['reward']:.2f}")
            print(f"    Bonus: {metrics['bonus']}")
            print(f"    Valid: {metrics['is_valid']}")
            print(f"    Back:   {Deck.cards_to_string(back)}")
            print(f"    Middle: {Deck.cards_to_string(middle)}")
            print(f"    Front:  {Deck.cards_to_string(front)}")
        else:
            print(f"    ❌ Solve failed: {metrics.get('error', 'Unknown')}")
        
        # Test fast mode
        print("\n  Testing mode='fast'...")
        back2, middle2, front2, metrics2 = bridge.solve(cards, mode='fast')
        
        if back2:
            print(f"    ✅ Fast mode OK, reward: {metrics2['reward']:.2f}")
        else:
            print(f"    ❌ Fast mode failed")
        
        # Test beam mode
        print("\n  Testing mode='beam'...")
        back3, middle3, front3, metrics3 = bridge.solve(cards, mode='beam')
        
        if back3:
            print(f"    ✅ Beam mode OK, reward: {metrics3['reward']:.2f}")
        else:
            print(f"    ❌ Beam mode failed")
    
    else:
        print("\n⚠️ Model not loaded - skipping solve tests")
        print("   To test, first train a model and place it in data/models/")
    
    print("\n" + "="*60)
    print("✅ MLSolverBridge test completed!")
    print("="*60)


if __name__ == "__main__":
    test_ml_bridge()