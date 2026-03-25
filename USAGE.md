# 📖 MAU BINH ML - COMPLETE USAGE GUIDE

## 🎯 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Complete Pipeline](#complete-pipeline)
4. [Advanced Usage](#advanced-usage)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [FAQ](#faq)

---

## 🚀 Quick Start

### Minimum Requirements

- Python 3.8+
- 8GB RAM (16GB recommended)
- CPU: 4 cores+ (GPU optional but faster)
- Disk: 5GB free space

### Installation

```bash
# 1. Clone repository
cd mau_binh_ultimate

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
cd src
python -c "from ml import MauBinhAgent; print('✅ Installation successful!')"
```

---

## 📦 Complete Pipeline

### Step 1: Generate Training Data

#### Option A: Small dataset (for testing - 10 minutes)

```bash
cd src
python -m ml.data.expert_generator --samples 10000 --workers 4
```

**Output:** `data/training/expert_v3_10000.pkl`

#### Option B: Medium dataset (recommended - 1 hour)

```bash
python -m ml.data.expert_generator --samples 100000 --workers 4
```

**Output:** `data/training/expert_v3_100000.pkl`

#### Option C: Large dataset (production - 3-4 hours)

```bash
python -m ml.data.expert_generator --samples 200000 --workers 8
```

**Output:** `data/training/expert_v3_200000.pkl`

**Expected statistics:**
```
📊 Dataset statistics:
   Valid: 200000/200000 (100.0%)
   With bonus: 12000/200000 (6.0%)
   Reward range: 1.52 - 149.15
   Reward mean: 8.47
```

---

### Step 2: Train Model

#### Quick Training (testing - 10 minutes)

```bash
python -m ml.training.trainer \
  --data data/training/expert_v3_10000.pkl \
  --network dqn \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --patience 15 \
  --name test_model
```

#### Standard Training (recommended - 1 hour)

```bash
python -m ml.training.trainer \
  --data data/training/expert_v3_100000.pkl \
  --network ensemble \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-4 \
  --patience 20 \
  --name standard_model_v1
```

#### Production Training (best quality - 2-3 hours)

```bash
python -m ml.training.trainer \
  --data data/training/expert_v3_200000.pkl \
  --network ensemble \
  --epochs 150 \
  --batch-size 128 \
  --lr 1e-4 \
  --patience 25 \
  --name production_v1
```

**Expected output:**
```
🚀 TRAINING: production_v1
============================================================
Epochs: 150
Batch size: 128
Learning rate: 0.0001

Epoch   1/150 | Train: 195.52 | Val: 191.05 | LR: 0.000040 💾 BEST!
Epoch   2/150 | Train: 191.32 | Val: 189.56 | LR: 0.000060 💾 BEST!
...
Epoch  47/150 | Train: 42.53 | Val: 38.14 | LR: 0.000065 💾 BEST!
...
⏹️  Early stopping at epoch 62

============================================================
✅ TRAINING COMPLETED
============================================================
Best val loss: 38.14
Models saved to: data\models\production_v1
```

**Models saved:**
- `data/models/production_v1/best_model.pth` ← **Use this for production**
- `data/models/production_v1/final_model.pth`
- `data/models/production_v1/checkpoint_epoch*.pth`
- `data/models/production_v1/training_history.pkl`

---

### Step 3: Validate Model

```bash
python -m ml.evaluation.validator \
  --model data/models/production_v1/best_model.pth \
  --tests 1000
```

**Expected output:**
```
🧪 Validating model with 1000 tests...
   Progress: 100/1000
   Progress: 200/1000
   ...

============================================================
📊 VALIDATION RESULTS
============================================================
Valid Rate:        98.50%
Avg Reward:        12.35
Bonus Rate:         7.20%
Perfect Rate:       1.80%

Statistics:
  Min reward:        1.52
  Max reward:      149.15
  Median reward:    10.25
  Std reward:       15.38
============================================================
✅ EXCELLENT - Model produces valid arrangements
✅ EXCELLENT - High average reward
```

**Quality benchmarks:**
- ✅ **Valid Rate ≥ 95%** → Production ready
- ⚠️ **Valid Rate < 95%** → Need more training
- ✅ **Avg Reward ≥ 10** → Good performance
- ⚠️ **Avg Reward < 8** → Retrain with more data
- ✅ **Bonus Rate ≥ 5%** → Finding bonuses well

---

### Step 4: Benchmark Against Baselines

```bash
python -m ml.evaluation.benchmark \
  --model data/models/production_v1/best_model.pth \
  --hands 500
```

**Expected output:**
```
🏁 Running benchmark with 500 hands...

======================================================================
🏆 BENCHMARK COMPARISON
======================================================================
Method            Valid%   Avg Reward   Bonus%   Time(ms)
----------------------------------------------------------------------
Random             62.0%         1.45     0.0%       0.00
Greedy             48.0%         2.49     0.0%       0.02
Ml                 98.5%        12.35     7.2%      45.23
======================================================================
✅ ML Agent wins! +396.4% better than greedy
```

**Performance targets:**
| Metric | Target | Excellent | Good | Poor |
|--------|--------|-----------|------|------|
| Valid Rate | >95% | >98% | 95-98% | <95% |
| Avg Reward | >10 | >15 | 10-15 | <10 |
| Bonus Rate | >5% | >8% | 5-8% | <5% |
| Inference Time | <50ms | <30ms | 30-50ms | >50ms |

---

## 🎯 Advanced Usage

### A. Data Augmentation (Tăng gấp đôi dataset)

```python
from ml.data import DataAugmentation
import pickle

# Load dataset
with open('data/training/expert_v3_100000.pkl', 'rb') as f:
    dataset = pickle.load(f)

print(f"Original dataset: {len(dataset)} samples")

# Augment (suit permutation + noise)
aug = DataAugmentation()
augmented_dataset = aug.augment_dataset(dataset, augmentation_factor=2)

print(f"Augmented dataset: {len(augmented_dataset)} samples")

# Save
with open('data/training/expert_v3_200000_aug.pkl', 'wb') as f:
    pickle.dump(augmented_dataset, f)

print("✅ Augmented dataset saved!")
```

**When to use:**
- Khi có ít data (<50k samples)
- Khi model overfitting
- Để tăng diversity

---

### B. Curriculum Learning (Học từ dễ → khó)

```python
from ml.training import TrainerV3, CurriculumScheduler

# Setup curriculum
scheduler = CurriculumScheduler(total_epochs=90, difficulty_levels=3)
scheduler.print_curriculum_plan()

# Output:
# 📚 Curriculum Learning Plan:
# ==================================================
#   Level 0 (Easy):
#     Epochs: 1 - 30
#     Samples: reward < 10 (simple hands)
#   Level 1 (Medium):
#     Epochs: 31 - 60
#     Samples: reward < 50 (pairs, trips)
#   Level 2 (Hard):
#     Epochs: 61 - 90
#     Samples: all (including bonus)
# ==================================================

# Train with curriculum
# (Note: Trainer needs modification to use curriculum - advanced feature)
```

**Benefits:**
- Faster convergence
- Better generalization
- Less overfitting

---

### C. Self-Play Fine-Tuning

```python
from ml.data import SelfPlayGenerator

# Generate self-play data from trained model
gen = SelfPlayGenerator(
    model_path='data/models/production_v1/best_model.pth'
)

# Generate 10k samples with exploration
dataset_path = gen.generate_with_model(
    num_samples=10000,
    epsilon=0.1,  # 10% exploration
    output_name='self_play_10k.pkl'
)

# Train on self-play data (fine-tuning)
# python -m ml.training.trainer --data data/training/self_play_10k.pkl ...
```

**Use cases:**
- Fine-tuning after initial training
- Exploring new strategies
- Adapting to specific opponents

---

### D. Beam Search for Better Solutions

```python
from ml.agent import BeamSearch
from card import Deck

# Create beam search solver
beam = BeamSearch(beam_width=10)

# Solve hand
cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
back, middle, front = beam.search(cards, depth=2)

print(f"Back:   {Deck.cards_to_string(back)}")
print(f"Middle: {Deck.cards_to_string(middle)}")
print(f"Front:  {Deck.cards_to_string(front)}")
```

**Comparison:**
- **Greedy**: Fast, ~80% optimal
- **Beam (width=5)**: Medium, ~95% optimal
- **Beam (width=10)**: Slow, ~98% optimal
- **Exhaustive**: Very slow, 100% optimal (not practical)

---

### E. Visualize Training Progress

```python
from ml.evaluation import MetricsVisualizer

# Print summary
MetricsVisualizer.print_summary('data/models/production_v1/training_history.pkl')

# Plot training curves
MetricsVisualizer.plot_training_history(
    'data/models/production_v1/training_history.pkl',
    save_path='training_plot.png'
)
```

**Output:**
```
============================================================
📊 TRAINING SUMMARY
============================================================
Total epochs:    62
Best val loss:   38.1432
Final train loss: 42.5384
Final val loss:   39.6669
============================================================
💾 Plot saved to training_plot.png
```

---

## 🏭 Production Deployment

### Method 1: Python Script

```python
# production_solver.py

from ml.agent import MauBinhAgent
from card import Deck

# Initialize agent (load once at startup)
agent = MauBinhAgent(
    model_path='data/models/production_v1/best_model.pth',
    device='cpu'  # or 'cuda' if GPU available
)

def solve_hand(cards_str: str):
    """
    Solve a Mau Binh hand
    
    Args:
        cards_str: "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    
    Returns:
        {
            'back': [...],
            'middle': [...],
            'front': [...],
            'reward': float,
            'is_valid': bool
        }
    """
    # Parse cards
    cards = Deck.parse_hand(cards_str)
    
    # Solve (fast: <50ms)
    back, middle, front = agent.solve(cards, mode='best')
    
    # Evaluate
    eval_result = agent.evaluate_arrangement((back, middle, front))
    
    return {
        'back': Deck.cards_to_string(back),
        'middle': Deck.cards_to_string(middle),
        'front': Deck.cards_to_string(front),
        **eval_result
    }

# Usage
if __name__ == "__main__":
    result = solve_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    print(result)
```

---

### Method 2: FastAPI Server

```python
# api_server.py

from fastapi import FastAPI
from pydantic import BaseModel
from ml.agent import MauBinhAgent
from card import Deck

app = FastAPI(title="Mau Binh Solver API")

# Load model once at startup
agent = MauBinhAgent(model_path='data/models/production_v1/best_model.pth')

class SolveRequest(BaseModel):
    cards: str  # "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"

@app.post("/solve")
def solve(request: SolveRequest):
    cards = Deck.parse_hand(request.cards)
    back, middle, front = agent.solve(cards, mode='best')
    eval_result = agent.evaluate_arrangement((back, middle, front))
    
    return {
        'back': Deck.cards_to_string(back),
        'middle': Deck.cards_to_string(middle),
        'front': Deck.cards_to_string(front),
        **eval_result
    }

# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Test API:**
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"cards": "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"}'
```

---

### Method 3: Batch Processing

```python
from ml.agent import MauBinhAgent
from card import Deck
import csv

agent = MauBinhAgent(model_path='data/models/production_v1/best_model.pth')

# Read hands from CSV
hands = []
with open('hands.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        hands.append(Deck.parse_hand(row[0]))

# Batch solve (faster than one-by-one)
results = agent.batch_solve(hands, mode='best')

# Write results
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Back', 'Middle', 'Front', 'Reward'])
    
    for (back, middle, front) in results:
        eval_result = agent.evaluate_arrangement((back, middle, front))
        writer.writerow([
            Deck.cards_to_string(back),
            Deck.cards_to_string(middle),
            Deck.cards_to_string(front),
            eval_result['reward']
        ])

print(f"✅ Processed {len(results)} hands")
```

---

## 🔧 Troubleshooting

### Problem 1: Low Valid Rate (<90%)

**Symptoms:**
```
Valid Rate: 78.5%  ❌
Avg Reward: 5.23
```

**Causes:**
- Not enough training data
- Model underfitting
- Wrong reward function

**Solutions:**

```bash
# Solution A: Generate more data
python -m ml.data.expert_generator --samples 200000 --workers 8

# Solution B: Train longer
python -m ml.training.trainer \
  --data data/training/expert_v3_200000.pkl \
  --epochs 200 \
  --patience 30

# Solution C: Use data augmentation
# (see Advanced Usage section)
```

---

### Problem 2: Overfitting (Train loss << Val loss)

**Symptoms:**
```
Epoch 50/100 | Train: 15.23 | Val: 85.67
```

**Solutions:**

```python
# Solution A: Increase weight decay
python -m ml.training.trainer \
  --data ... \
  --epochs 100 \
  --weight-decay 1e-4  # Increased from 1e-5

# Solution B: Use data augmentation
# (see Advanced Usage - Data Augmentation)

# Solution C: Early stopping with smaller patience
python -m ml.training.trainer \
  --data ... \
  --patience 10  # Reduced from 20
```

---

### Problem 3: Slow Training

**Symptoms:**
- Training takes >5 hours for 100k samples
- Each epoch takes >10 seconds

**Solutions:**

```bash
# Solution A: Use GPU (if available)
# Automatic - trainer detects GPU

# Solution B: Reduce batch size (if memory issue)
python -m ml.training.trainer \
  --data ... \
  --batch-size 32  # Reduced from 64

# Solution C: Use DQN instead of Ensemble
python -m ml.training.trainer \
  --data ... \
  --network dqn  # Faster than ensemble
```

---

### Problem 4: Low Bonus Rate (<3%)

**Symptoms:**
```
Bonus Rate: 2.1%  ⚠️
```

**Solutions:**

```python
# Solution A: Generate bonus-focused data
from ml.data import ExpertDataGeneratorV3

gen = ExpertDataGeneratorV3()

# Custom generation with bonus preference
# (requires code modification to prioritize bonus hands)

# Solution B: Adjust reward weights
# Edit src/ml/core/reward_calculator.py
# Change: BONUS_WEIGHT = 10.0 → 15.0

# Solution C: Train longer
python -m ml.training.trainer --epochs 200
```

---

### Problem 5: Slow Inference (>100ms per hand)

**Symptoms:**
```
Inference time: 150ms  ⚠️
```

**Solutions:**

```python
# Solution A: Use DQN only (faster than ensemble)
from ml.agent import MauBinhAgent

agent = MauBinhAgent(model_path='...')
# When solving:
arrangement = agent.solve(cards, mode='best')  # Uses ensemble

# Faster alternative - use DQN directly
from ml.networks import DQNNetwork
# Load DQN-only model

# Solution B: Batch processing
# Process multiple hands at once (see Production Deployment)

# Solution C: Use greedy decoder instead of beam search
```

---

## 🎓 Performance Tuning

### Network Architecture Selection

| Network | Speed | Quality | Memory | Use Case |
|---------|-------|---------|--------|----------|
| **DQN** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 💾 Low | Production (speed critical) |
| **Transformer** | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | 💾💾 Medium | Research, offline |
| **Ensemble** | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | 💾💾💾 High | **Recommended** |

**Recommendation:**
- **Development/Testing**: DQN (fast iteration)
- **Production (quality)**: Ensemble (best results)
- **Production (speed)**: DQN (acceptable quality, 3x faster)

---

### Hyperparameter Tuning

#### Learning Rate

```bash
# Too high (loss oscillates)
--lr 1e-3  ❌

# Good (default)
--lr 1e-4  ✅

# Too low (slow convergence)
--lr 1e-5  ⚠️
```

#### Batch Size

```bash
# Small batch (noisy gradients, slow)
--batch-size 16  ⚠️

# Medium batch (good balance)
--batch-size 64  ✅

# Large batch (smooth gradients, needs more memory)
--batch-size 256  ✅ (if RAM allows)
```

#### Training Epochs

```bash
# Quick test
--epochs 20  (testing only)

# Standard
--epochs 100  ✅

# Production
--epochs 200  ✅ (with early stopping)
```

---

## ❓ FAQ

### Q1: Tao cần bao nhiêu data để train?

**A:**
- **Minimum**: 10k samples (testing only)
- **Good**: 100k samples (acceptable quality)
- **Production**: 200k+ samples (best quality)
- **Overkill**: 500k+ samples (diminishing returns)

**Rule of thumb:** 100k samples = 1 hour generation, 1 hour training, good results.

---

### Q2: GPU có cần thiết không?

**A:**
- **Training**: GPU giúp nhanh hơn 5-10x (highly recommended)
- **Inference**: CPU đủ (<50ms per hand)
- **Data generation**: CPU-only (multiprocessing)

**Recommendation:**
- Training with GPU: ✅ Highly recommended
- Training with CPU: ✅ OK but slower
- Inference: ✅ CPU is fine

---

### Q3: Model size bao nhiêu?

**A:**
- **DQN**: ~1MB (241k parameters)
- **Transformer**: ~3MB (711k parameters)
- **Ensemble**: ~4MB (952k parameters)

All models load fast (<1 second) and fit in memory easily.

---

### Q4: Làm sao biết model đã đủ tốt?

**A:** Check validation metrics:

```
✅ PRODUCTION READY:
- Valid Rate ≥ 98%
- Avg Reward ≥ 12
- Bonus Rate ≥ 7%

⚠️  ACCEPTABLE:
- Valid Rate ≥ 95%
- Avg Reward ≥ 10
- Bonus Rate ≥ 5%

❌ NEEDS IMPROVEMENT:
- Valid Rate < 95%
- Avg Reward < 10
- Bonus Rate < 5%
```

---

### Q5: Có thể train thêm model đã train rồi không?

**A:** Có! (Fine-tuning)

```bash
# Continue training from checkpoint
python -m ml.training.trainer \
  --data data/training/expert_v3_200000.pkl \
  --network ensemble \
  --resume data/models/production_v1/best_model.pth \
  --epochs 50

# Or generate new data with existing model (self-play)
# See Advanced Usage - Self-Play Fine-Tuning
```

---

### Q6: Làm sao optimize inference speed?

**A:**

```python
# Method 1: Use DQN only (3x faster)
agent = MauBinhAgent(model_path='dqn_model.pth')

# Method 2: Batch processing
results = agent.batch_solve(batch_of_hands)  # Faster than loop

# Method 3: Disable ensemble (if using ensemble network)
agent.network.use_ensemble = False  # Use DQN only

# Method 4: PyTorch optimization
import torch
agent.network = torch.jit.script(agent.network)  # JIT compilation
```

---

### Q7: Có thể deploy lên cloud không?

**A:** Có! Multiple options:

**Option A: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0"]
```

**Option B: AWS Lambda** (serverless)
- Package model + code → ZIP
- Upload to Lambda
- Trigger via API Gateway

**Option C: Heroku / Railway**
```bash
# Add Procfile
web: uvicorn api_server:app --host 0.0.0.0 --port $PORT

# Deploy
git push heroku main
```

---

## 📞 Support & Contact

### Debug Mode

```python
# Enable verbose logging
from ml.agent import MauBinhAgent

agent = MauBinhAgent(model_path='...', verbose=True)
agent.solve(cards)  # Prints detailed info
```

### Check Module Versions

```bash
cd src
python -c "
from ml import __version__
from ml.core import __version__ as core_version
from ml.networks import __version__ as net_version
from ml.training import __version__ as train_version

print(f'ML Module: {__version__}')
print(f'Core: {core_version}')
print(f'Networks: {net_version}')
print(f'Training: {train_version}')
"
```

### Report Issues

Include:
1. Command run
2. Error message
3. Dataset size
4. System info (OS, Python version)

---

## 🎯 Next Steps

1. ✅ **Generate data** (100k samples minimum)
2. ✅ **Train model** (ensemble, 100 epochs)
3. ✅ **Validate** (target >95% valid rate)
4. ✅ **Benchmark** (compare vs baselines)
5. ✅ **Deploy** (integrate into your app)

**Good luck bro! 🚀💰**

---

**Version:** 2.0.0  
**Last Updated:** 2025-01-25  
**Author:** Ultimate Team