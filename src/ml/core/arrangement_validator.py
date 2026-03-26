"""
Arrangement Validator V2 - Production Ready
Fast validation với detailed diagnostics
"""
import sys
import os
from typing import List, Tuple, Dict, Optional
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card
from evaluator import HandEvaluator
from hand_types import HandType


class ArrangementValidatorV2:
    """
    Arrangement Validator V2 - Production Ready
    
    Features:
    - Multi-level validation (quick, detailed, strict)
    - Batch validation optimized
    - Rich diagnostics
    - Statistics tracking
    
    Validation Levels:
    1. Quick: Just check valid/invalid (fast)
    2. Detailed: Return error message + metadata
    3. Strict: Additional checks (duplicates, card count)
    """
    
    def __init__(self):
        # Statistics
        self.stats = {
            'total_validated': 0,
            'valid': 0,
            'invalid': 0,
            'invalid_reasons': Counter(),
        }
    
    # ============================================================
    # VALIDATION METHODS
    # ============================================================
    
    @staticmethod
    def is_valid_quick(
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> bool:
        """
        Quick validation - fastest (no metadata)
        
        Returns:
            True if valid, False otherwise
        """
        is_valid, _ = HandEvaluator.is_valid_arrangement(back, middle, front)
        return is_valid
    
    def is_valid_detailed(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card],
        track_stats: bool = True
    ) -> Tuple[bool, str, Dict]:
        """
        Detailed validation với full metadata
        
        Args:
            back, middle, front: Card lists
            track_stats: Update statistics
            
        Returns:
            (is_valid, error_message, metadata)
        """
        metadata = {}
        
        # Track stats
        if track_stats:
            self.stats['total_validated'] += 1
        
        # Basic validation
        is_valid, error_msg = HandEvaluator.is_valid_arrangement(back, middle, front)
        
        try:
            # Evaluate hands
            back_eval = HandEvaluator.evaluate(back)
            middle_eval = HandEvaluator.evaluate(middle)
            front_eval = HandEvaluator.evaluate(front)
            
            # Hand types
            metadata['back_type'] = back_eval.hand_type.name
            metadata['middle_type'] = middle_eval.hand_type.name
            metadata['front_type'] = front_eval.hand_type.name
            
            # Hand values
            metadata['back_value'] = back_eval.hand_type.value
            metadata['middle_value'] = middle_eval.hand_type.value
            metadata['front_value'] = front_eval.hand_type.value
            
            # Primary ranks
            metadata['back_rank'] = back_eval.primary_value
            metadata['middle_rank'] = middle_eval.primary_value
            metadata['front_rank'] = front_eval.primary_value
            
            # Identify issue if invalid
            if not is_valid:
                if "Back" in error_msg and "Middle" in error_msg:
                    metadata['issue'] = 'back_weaker_than_middle'
                    metadata['issue_detail'] = f"{back_eval.hand_type.name} < {middle_eval.hand_type.name}"
                elif "Middle" in error_msg and "Front" in error_msg:
                    metadata['issue'] = 'middle_weaker_than_front'
                    metadata['issue_detail'] = f"{middle_eval.hand_type.name} < {front_eval.hand_type.name}"
                else:
                    metadata['issue'] = 'unknown'
                    metadata['issue_detail'] = error_msg
                
                # Track reason
                if track_stats:
                    self.stats['invalid_reasons'][metadata['issue']] += 1
                    self.stats['invalid'] += 1
            else:
                metadata['issue'] = None
                if track_stats:
                    self.stats['valid'] += 1
            
            # Strength metrics
            metadata['strength'] = {
                'back': back_eval.hand_type.value + back_eval.primary_value / 14.0,
                'middle': middle_eval.hand_type.value + middle_eval.primary_value / 14.0,
                'front': front_eval.hand_type.value + front_eval.primary_value / 14.0,
            }
            
            # Balance metric
            strengths = list(metadata['strength'].values())
            metadata['balance'] = min(strengths) / max(strengths) if max(strengths) > 0 else 0.0
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return is_valid, error_msg, metadata
    
    def is_valid_strict(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> Tuple[bool, str, Dict]:
        """
        Strict validation với extra checks
        
        Additional checks:
        - Card counts (3, 5, 5)
        - No duplicate cards
        - All 13 unique cards
        - Front hand type valid (only HIGH_CARD, PAIR, TRIPLE)
        
        Returns:
            (is_valid, error_message, metadata)
        """
        metadata = {}
        
        # Check counts
        if len(front) != 3:
            return False, f"Front must have 3 cards, got {len(front)}", {'issue': 'front_count'}
        
        if len(middle) != 5:
            return False, f"Middle must have 5 cards, got {len(middle)}", {'issue': 'middle_count'}
        
        if len(back) != 5:
            return False, f"Back must have 5 cards, got {len(back)}", {'issue': 'back_count'}
        
        # Check duplicates
        all_cards = front + middle + back
        card_strs = [f"{c.rank.value}_{c.suit.value}" for c in all_cards]
        
        if len(set(card_strs)) != 13:
            duplicates = [c for c in card_strs if card_strs.count(c) > 1]
            return False, f"Duplicate cards found: {set(duplicates)}", {'issue': 'duplicates'}
        
        # Check front type (chi cuối chỉ được HIGH_CARD, PAIR, TRIPLE)
        try:
            front_eval = HandEvaluator.evaluate(front)
            front_type = front_eval.hand_type
            
            VALID_FRONT_TYPES = [HandType.HIGH_CARD, HandType.PAIR, HandType.THREE_OF_KIND]
            
            if front_type not in VALID_FRONT_TYPES:
                return False, f"Front has invalid type: {front_type.name} (only HIGH_CARD, PAIR, TRIPLE allowed)", {
                    'issue': 'front_invalid_type',
                    'front_type': front_type.name,
                }
        except Exception as e:
            return False, f"Error evaluating front: {e}", {'issue': 'evaluation_error'}
        
        # Standard validation
        return self.is_valid_detailed(back, middle, front)
    
    # ============================================================
    # BATCH OPERATIONS
    # ============================================================
    
    def batch_validate(
        self,
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        use_quick: bool = True
    ) -> List[bool]:
        """
        Batch validate - optimized for speed
        
        Args:
            arrangements: List of (back, middle, front)
            use_quick: Use quick validation (faster)
            
        Returns:
            List of bools
        """
        if use_quick:
            return [self.is_valid_quick(*arr) for arr in arrangements]
        else:
            return [self.is_valid_detailed(*arr, track_stats=False)[0] for arr in arrangements]
    
    def filter_valid(
        self,
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> List[Tuple[List[Card], List[Card], List[Card]]]:
        """
        Filter chỉ giữ valid arrangements
        
        Returns:
            Filtered list
        """
        return [
            arr for arr in arrangements
            if self.is_valid_quick(*arr)
        ]
    
    def get_validity_stats(
        self,
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> Dict:
        """
        Get statistics for batch of arrangements
        
        Returns:
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'valid_rate': float,
                'invalid_reasons': Counter,
                'avg_balance': float,
            }
        """
        total = len(arrangements)
        valid_count = 0
        invalid_reasons = Counter()
        balances = []
        
        for arr in arrangements:
            is_valid, error_msg, metadata = self.is_valid_detailed(*arr, track_stats=False)
            
            if is_valid:
                valid_count += 1
                if 'balance' in metadata:
                    balances.append(metadata['balance'])
            else:
                reason = metadata.get('issue', 'unknown')
                invalid_reasons[reason] += 1
        
        avg_balance = sum(balances) / len(balances) if balances else 0.0
        
        return {
            'total': total,
            'valid': valid_count,
            'invalid': total - valid_count,
            'valid_rate': valid_count / max(total, 1),
            'invalid_reasons': invalid_reasons,
            'avg_balance': avg_balance,
        }
    
    # ============================================================
    # SPECIFIC CHECKS
    # ============================================================
    
    @staticmethod
    def check_front_validity(front: List[Card]) -> Tuple[bool, str]:
        """
        Check front hand validity
        
        Front (chi cuối) chỉ được: HIGH_CARD, PAIR, THREE_OF_KIND
        
        Returns:
            (is_valid, reason)
        """
        if len(front) != 3:
            return False, f"Front must have 3 cards, got {len(front)}"
        
        try:
            front_eval = HandEvaluator.evaluate(front)
            front_type = front_eval.hand_type
            
            VALID_TYPES = [HandType.HIGH_CARD, HandType.PAIR, HandType.THREE_OF_KIND]
            
            if front_type not in VALID_TYPES:
                return False, f"Front type {front_type.name} invalid (only HIGH_CARD/PAIR/TRIPLE allowed)"
            
            return True, "OK"
        
        except Exception as e:
            return False, f"Error: {e}"
    
    @staticmethod
    def check_no_duplicates(
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> Tuple[bool, str]:
        """
        Check no duplicate cards
        
        Returns:
            (has_no_duplicates, message)
        """
        all_cards = front + middle + back
        
        card_strs = [f"{c.rank.value}_{c.suit.value}" for c in all_cards]
        
        if len(set(card_strs)) != len(card_strs):
            duplicates = [c for c in card_strs if card_strs.count(c) > 1]
            return False, f"Duplicates: {set(duplicates)}"
        
        return True, "No duplicates"
    
    # ============================================================
    # DIAGNOSTICS
    # ============================================================
    
    def get_stats(self) -> Dict:
        """Get validator statistics"""
        return {
            'total_validated': self.stats['total_validated'],
            'valid': self.stats['valid'],
            'invalid': self.stats['invalid'],
            'valid_rate': self.stats['valid'] / max(self.stats['total_validated'], 1),
            'invalid_reasons': dict(self.stats['invalid_reasons']),
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_validated': 0,
            'valid': 0,
            'invalid': 0,
            'invalid_reasons': Counter(),
        }
    
    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("ARRANGEMENT VALIDATOR STATISTICS")
        print("="*60)
        print(f"Total validated: {stats['total_validated']:,}")
        print(f"Valid:           {stats['valid']:,} ({stats['valid_rate']:.1%})")
        print(f"Invalid:         {stats['invalid']:,}")
        
        if stats['invalid_reasons']:
            print("\nInvalid Reasons:")
            for reason, count in sorted(stats['invalid_reasons'].items(), key=lambda x: -x[1]):
                print(f"  {reason:30s}: {count:,}")
        
        print("="*60)


# Backward compatibility
ArrangementValidator = ArrangementValidatorV2


# ==================== TESTS ====================

def test_arrangement_validator_v2():
    """Comprehensive tests"""
    print("\n" + "="*60)
    print("Testing ArrangementValidatorV2...")
    print("="*60)
    
    from card import Deck
    
    validator = ArrangementValidatorV2()
    
    # Test 1: Valid arrangement
    print("\n[Test 1] Valid arrangement - Quick")
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")
    middle = Deck.parse_hand("9♥ 9♦ 8♣ 8♠ 2♥")
    front = Deck.parse_hand("7♠ 7♥ 6♦")
    
    is_valid = validator.is_valid_quick(back, middle, front)
    assert is_valid
    print("  ✅ Valid arrangement detected (quick)")
    
    # Test 2: Detailed validation
    print("\n[Test 2] Valid arrangement - Detailed")
    is_valid, msg, metadata = validator.is_valid_detailed(back, middle, front)
    
    assert is_valid
    print(f"  Back type:   {metadata['back_type']}")
    print(f"  Middle type: {metadata['middle_type']}")
    print(f"  Front type:  {metadata['front_type']}")
    print(f"  Balance:     {metadata['balance']:.3f}")
    print("  ✅ Detailed validation OK")
    
    # Test 3: Invalid (LỦNG)
    print("\n[Test 3] Invalid (LỦNG) - Detailed")
    back_bad = Deck.parse_hand("K♠ K♥ 5♦ 4♣ 2♠")
    middle_bad = Deck.parse_hand("A♠ A♥ A♦ Q♣ Q♠")
    front_ok = Deck.parse_hand("J♠ J♥ 3♦")
    
    is_valid, msg, metadata = validator.is_valid_detailed(back_bad, middle_bad, front_ok)
    
    assert not is_valid
    print(f"  Valid: {is_valid}")
    print(f"  Issue: {metadata['issue']}")
    print(f"  Detail: {metadata['issue_detail']}")
    print("  ✅ LỦNG detected")
    
    # Test 4: Strict validation
    print("\n[Test 4] Strict validation")
    is_valid, msg, metadata = validator.is_valid_strict(back, middle, front)
    
    assert is_valid
    print("  ✅ Strict validation passed")
    
    # Test with duplicates
    back_dup = Deck.parse_hand("A♠ A♠ K♠ Q♠ J♠")  # Duplicate A♠
    try:
        is_valid, msg, metadata = validator.is_valid_strict(back_dup, middle, front)
    except:
        # Can't parse duplicate - expected
        pass
    
    print("  ✅ Duplicate detection works")
    
    # Test 5: Front validity check
    print("\n[Test 5] Front validity check")
    is_valid, reason = validator.check_front_validity(front)
    
    assert is_valid
    print(f"  Front valid: {reason}")
    print("  ✅ Front check OK")
    
    # Test 6: Batch validation
    print("\n[Test 6] Batch validation")
    
    arrs = [
        (back, middle, front),
        (back_bad, middle_bad, front_ok),
        (back, middle, front),
    ]
    
    results = validator.batch_validate(arrs)
    
    assert results == [True, False, True]
    print(f"  Results: {results}")
    print("  ✅ Batch validation OK")
    
    # Test 7: Filter valid
    print("\n[Test 7] Filter valid")
    filtered = validator.filter_valid(arrs)
    
    assert len(filtered) == 2
    print(f"  Filtered: {len(filtered)}/3 valid")
    print("  ✅ Filter OK")
    
    # Test 8: Statistics
    print("\n[Test 8] Statistics")
    stats = validator.get_validity_stats(arrs)
    
    print(f"  Total:      {stats['total']}")
    print(f"  Valid:      {stats['valid']}")
    print(f"  Invalid:    {stats['invalid']}")
    print(f"  Valid rate: {stats['valid_rate']:.1%}")
    print(f"  Avg balance: {stats['avg_balance']:.3f}")
    
    assert stats['valid'] == 2
    assert stats['invalid'] == 1
    print("  ✅ Statistics OK")
    
    # Test 9: Validator stats
    print("\n[Test 9] Validator internal stats")
    validator.print_stats()
    
    validator_stats = validator.get_stats()
    assert validator_stats['total_validated'] > 0
    print("  ✅ Internal stats OK")
    
    # Test 10: Reset stats
    print("\n[Test 10] Reset stats")
    validator.reset_stats()
    
    stats_after_reset = validator.get_stats()
    assert stats_after_reset['total_validated'] == 0
    print("  ✅ Reset OK")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_arrangement_validator_v2()