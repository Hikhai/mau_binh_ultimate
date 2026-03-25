"""
Arrangement Validator - Wrapper tiện lợi
"""
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card
from evaluator import HandEvaluator
from hand_types import HandType


class ArrangementValidator:
    """
    Validate arrangements với nhiều level khác nhau
    """
    
    @staticmethod
    def is_valid_basic(
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> bool:
        """
        Kiểm tra cơ bản (nhanh)
        
        Returns:
            True nếu valid, False nếu không
        """
        is_valid, _ = HandEvaluator.is_valid_arrangement(back, middle, front)
        return is_valid
    
    @staticmethod
    def is_valid_detailed(
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> Tuple[bool, str, dict]:
        """
        Kiểm tra chi tiết với thông tin đầy đủ
        
        Returns:
            (is_valid, error_message, metadata)
        """
        is_valid, error_msg = HandEvaluator.is_valid_arrangement(back, middle, front)
        
        # Thu thập metadata
        metadata = {}
        
        try:
            back_eval = HandEvaluator.evaluate(back)
            middle_eval = HandEvaluator.evaluate(middle)
            front_eval = HandEvaluator.evaluate(front)
            
            metadata['back_type'] = back_eval.hand_type.name
            metadata['middle_type'] = middle_eval.hand_type.name
            metadata['front_type'] = front_eval.hand_type.name
            
            metadata['back_rank'] = back_eval.primary_value
            metadata['middle_rank'] = middle_eval.primary_value
            metadata['front_rank'] = front_eval.primary_value
            
            # Check specific issues
            if not is_valid:
                if "Back" in error_msg and "Middle" in error_msg:
                    metadata['issue'] = 'back_weaker_than_middle'
                elif "Middle" in error_msg and "Front" in error_msg:
                    metadata['issue'] = 'middle_weaker_than_front'
                else:
                    metadata['issue'] = 'unknown'
        
        except Exception as e:
            metadata['error'] = str(e)
        
        return is_valid, error_msg, metadata
    
    @staticmethod
    def check_front_validity(front: List[Card]) -> Tuple[bool, str]:
        """
        Kiểm tra CHI CUỐI có hợp lệ không
        
        Chi cuối CHỈ được: HIGH_CARD, PAIR, THREE_OF_KIND
        
        Returns:
            (is_valid, reason)
        """
        if len(front) != 3:
            return False, f"Front must have 3 cards, got {len(front)}"
        
        try:
            front_eval = HandEvaluator.evaluate(front)
            front_type = front_eval.hand_type.value
            
            # Chi cuối chỉ có: 0 (HIGH_CARD), 1 (PAIR), 3 (THREE_OF_KIND)
            VALID_FRONT_TYPES = [0, 1, 3]
            
            if front_type not in VALID_FRONT_TYPES:
                return False, f"Front has invalid type: {front_eval.hand_type.name}"
            
            return True, "OK"
        
        except Exception as e:
            return False, f"Error evaluating front: {e}"
    
    @staticmethod
    def batch_validate(
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> List[bool]:
        """
        Validate batch of arrangements (nhanh)
        
        Returns:
            List of bools (True = valid, False = invalid)
        """
        return [
            ArrangementValidator.is_valid_basic(back, middle, front)
            for back, middle, front in arrangements
        ]
    
    @staticmethod
    def filter_valid(
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> List[Tuple[List[Card], List[Card], List[Card]]]:
        """
        Lọc chỉ giữ lại arrangements hợp lệ
        
        Returns:
            Filtered list
        """
        return [
            arr for arr in arrangements
            if ArrangementValidator.is_valid_basic(*arr)
        ]
    
    @staticmethod
    def get_validity_stats(
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> dict:
        """
        Thống kê validity
        
        Returns:
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'valid_rate': float,
                'invalid_reasons': Counter
            }
        """
        from collections import Counter
        
        total = len(arrangements)
        valid_count = 0
        invalid_reasons = []
        
        for arr in arrangements:
            is_valid, error_msg, _ = ArrangementValidator.is_valid_detailed(*arr)
            
            if is_valid:
                valid_count += 1
            else:
                # Extract reason
                if "LỦNG" in error_msg:
                    if "Back" in error_msg and "Middle" in error_msg:
                        invalid_reasons.append("back<middle")
                    elif "Middle" in error_msg and "Front" in error_msg:
                        invalid_reasons.append("middle<front")
                    else:
                        invalid_reasons.append("other_lung")
                else:
                    invalid_reasons.append("other")
        
        return {
            'total': total,
            'valid': valid_count,
            'invalid': total - valid_count,
            'valid_rate': valid_count / max(total, 1),
            'invalid_reasons': Counter(invalid_reasons)
        }


# ==================== TESTS ====================

def test_arrangement_validator():
    """Test ArrangementValidator"""
    print("Testing ArrangementValidator...")
    from card import Deck
    
    # Valid arrangement
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")
    middle = Deck.parse_hand("9♥ 9♦ 8♣ 8♠ 2♥")
    front = Deck.parse_hand("7♠ 7♥ 6♦")
    
    is_valid = ArrangementValidator.is_valid_basic(back, middle, front)
    assert is_valid
    print("  ✅ Valid arrangement detected")
    
    # Detailed validation
    is_valid, msg, metadata = ArrangementValidator.is_valid_detailed(back, middle, front)
    assert is_valid
    assert 'back_type' in metadata
    print(f"  ✅ Detailed validation: {metadata['back_type']}")
    
    # Invalid (LỦNG)
    back_bad = Deck.parse_hand("K♠ K♥ 5♦ 4♣ 2♠")
    middle_bad = Deck.parse_hand("A♠ A♥ A♦ Q♣ Q♠")
    front_ok = Deck.parse_hand("J♠ J♥ 3♦")
    
    is_valid = ArrangementValidator.is_valid_basic(back_bad, middle_bad, front_ok)
    assert not is_valid
    print("  ✅ Invalid (LỦNG) detected")
    
    is_valid, msg, metadata = ArrangementValidator.is_valid_detailed(back_bad, middle_bad, front_ok)
    assert not is_valid
    assert metadata.get('issue') == 'back_weaker_than_middle'
    print(f"  ✅ Issue identified: {metadata['issue']}")
    
    # Check front validity
    is_valid, reason = ArrangementValidator.check_front_validity(front)
    assert is_valid
    print(f"  ✅ Front valid: {reason}")
    
    # Batch validate
    arrs = [(back, middle, front), (back_bad, middle_bad, front_ok)]
    results = ArrangementValidator.batch_validate(arrs)
    assert results == [True, False]
    print(f"  ✅ Batch validate: {results}")
    
    # Filter valid
    filtered = ArrangementValidator.filter_valid(arrs)
    assert len(filtered) == 1
    print(f"  ✅ Filtered: {len(filtered)} valid")
    
    # Stats
    stats = ArrangementValidator.get_validity_stats(arrs)
    assert stats['valid'] == 1
    assert stats['invalid'] == 1
    assert stats['valid_rate'] == 0.5
    print(f"  ✅ Stats: {stats['valid']}/{stats['total']} valid")
    
    print("✅ ArrangementValidator tests passed!")


if __name__ == "__main__":
    test_arrangement_validator()