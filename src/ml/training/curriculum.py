"""
Curriculum Learning - Học từ dễ → khó
"""
import numpy as np
from typing import List, Dict, Callable


class CurriculumScheduler:
    """
    Quản lý curriculum learning
    
    Ý tưởng:
    - Epoch đầu: Học từ bài dễ (ít đôi, ít bonus, dễ xếp)
    - Epoch sau: Tăng dần độ khó
    - Cuối: Học từ mọi bài
    """
    
    def __init__(
        self,
        total_epochs: int,
        difficulty_levels: int = 3
    ):
        self.total_epochs = total_epochs
        self.difficulty_levels = difficulty_levels
        
        # Define difficulty thresholds
        self.thresholds = self._compute_thresholds()
    
    def _compute_thresholds(self) -> List[int]:
        """Compute epoch thresholds for each difficulty level"""
        step = self.total_epochs // self.difficulty_levels
        return [step * (i + 1) for i in range(self.difficulty_levels)]
    
    def get_difficulty_level(self, epoch: int) -> int:
        """
        Get current difficulty level (0 = easiest, higher = harder)
        """
        for level, threshold in enumerate(self.thresholds):
            if epoch < threshold:
                return level
        return self.difficulty_levels - 1
    
    @staticmethod
    def classify_sample_difficulty(sample: Dict) -> int:
        """
        Classify sample difficulty (0=easy, 1=medium, 2=hard)
        
        Criteria:
        - Easy: reward < 10 (simple hands)
        - Medium: 10 <= reward < 50 (some pairs)
        - Hard: reward >= 50 (bonus hands)
        """
        reward = sample['reward']
        
        if reward < 10:
            return 0  # Easy
        elif reward < 50:
            return 1  # Medium
        else:
            return 2  # Hard
    
    def filter_dataset_by_difficulty(
        self,
        dataset: List[Dict],
        max_difficulty: int
    ) -> List[Dict]:
        """
        Filter dataset to include only samples up to max_difficulty
        
        Args:
            dataset: Full dataset
            max_difficulty: Maximum difficulty to include (0, 1, 2)
            
        Returns:
            Filtered dataset
        """
        filtered = [
            s for s in dataset
            if self.classify_sample_difficulty(s) <= max_difficulty
        ]
        
        return filtered
    
    def get_epoch_dataset(
        self,
        full_dataset: List[Dict],
        epoch: int
    ) -> List[Dict]:
        """
        Get dataset for current epoch based on curriculum
        
        Returns:
            Dataset filtered by difficulty
        """
        difficulty_level = self.get_difficulty_level(epoch)
        return self.filter_dataset_by_difficulty(full_dataset, difficulty_level)
    
    def print_curriculum_plan(self):
        """Print curriculum learning plan"""
        print("📚 Curriculum Learning Plan:")
        print("="*50)
        
        for level in range(self.difficulty_levels):
            start_epoch = self.thresholds[level-1] if level > 0 else 0
            end_epoch = self.thresholds[level]
            
            difficulty_name = ['Easy', 'Medium', 'Hard'][level]
            
            print(f"  Level {level} ({difficulty_name}):")
            print(f"    Epochs: {start_epoch+1} - {end_epoch}")
            
            if level == 0:
                print(f"    Samples: reward < 10 (simple hands)")
            elif level == 1:
                print(f"    Samples: reward < 50 (pairs, trips)")
            else:
                print(f"    Samples: all (including bonus)")
        
        print("="*50)


# ==================== TESTS ====================

def test_curriculum_scheduler():
    """Test CurriculumScheduler"""
    print("Testing CurriculumScheduler...")
    
    scheduler = CurriculumScheduler(total_epochs=30, difficulty_levels=3)
    
   # Test difficulty level by epoch
    assert scheduler.get_difficulty_level(0) == 0  # Easy
    assert scheduler.get_difficulty_level(5) == 0
    assert scheduler.get_difficulty_level(9) == 0  # Epoch 1-10
    assert scheduler.get_difficulty_level(10) == 1  # Epoch 11-20 (Medium)
    assert scheduler.get_difficulty_level(15) == 1
    assert scheduler.get_difficulty_level(20) == 2  # Epoch 21-30 (Hard)
    assert scheduler.get_difficulty_level(25) == 2
    assert scheduler.get_difficulty_level(29) == 2
    
    print("  ✅ Difficulty levels OK")
    
    # Test sample classification
    easy_sample = {'reward': 5.0}
    medium_sample = {'reward': 25.0}
    hard_sample = {'reward': 80.0}
    
    assert scheduler.classify_sample_difficulty(easy_sample) == 0
    assert scheduler.classify_sample_difficulty(medium_sample) == 1
    assert scheduler.classify_sample_difficulty(hard_sample) == 2
    
    print("  ✅ Sample classification OK")
    
    # Test filtering
    dataset = [
        {'reward': 5.0},
        {'reward': 25.0},
        {'reward': 80.0},
        {'reward': 3.0},
        {'reward': 45.0},
    ]
    
    easy_only = scheduler.filter_dataset_by_difficulty(dataset, max_difficulty=0)
    assert len(easy_only) == 2  # rewards 5, 3
    
    medium_and_below = scheduler.filter_dataset_by_difficulty(dataset, max_difficulty=1)
    assert len(medium_and_below) == 4  # all except 80
    
    all_data = scheduler.filter_dataset_by_difficulty(dataset, max_difficulty=2)
    assert len(all_data) == 5  # all
    
    print("  ✅ Dataset filtering OK")
    
    # Print plan
    scheduler.print_curriculum_plan()
    
    print("✅ CurriculumScheduler tests passed!")


if __name__ == "__main__":
    test_curriculum_scheduler()