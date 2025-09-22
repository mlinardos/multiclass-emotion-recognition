# multiclass-emotion-recognition
An implementation of multi-label and multi-class emotion classification using transformer models on the Go_Emotions dataset. This project compares RoBERTa, XLNet, and ELECTRA architectures for recognizing 28 emotion categories in Reddit comments.

<div align="center">

| Model | Weighted F1-Score | Precision | Recall | ROC-AUC |
|:-----:|:-----------------:|:---------:|:------:|:-------:|
| **RoBERTa** | **0.6043** | 0.5542 | 0.6818 | 0.83 |
| ELECTRA | 0.6032 | 0.5464 | 0.6870 | 0.83 |
| XLNet | 0.5709 | 0.5062 | 0.6889 | 0.83 |


### Findings
- **Best Performance**: RoBERTa achieved the highest weighted F1-score (0.6043)
- **Strong Emotions**: All models excelled at positive emotions like gratitude (F1: 0.89-0.90), amusement (F1: 0.81-0.83), and love (F1: 0.77-0.81)
- **Challenging Emotions**: Models struggled with complex emotions like realization (F1: 0.21-0.26) and disappointment (F1: 0.28-0.34)
- **Class Imbalance**: Custom oversampling + Focal Loss improved rare emotion detection


<details>
<summary>Top Performing Emotions by Model</summary>

**RoBERTa**
- Gratitude: 0.90 F1-score
- Amusement: 0.83 F1-score  
- Love: 0.80 F1-score

**ELECTRA**
- Gratitude: 0.89 F1-score
- Amusement: 0.83 F1-score
- Love: 0.81 F1-score

**XLNet**
- Gratitude: 0.84 F1-score
- Amusement: 0.81 F1-score
- Love: 0.77 F1-score

</details>


### Training Strategy

#### Optimization Parameters
- **Optimizer**: AdamW
- **Weight Decay**: 0.1
- **Dropout**: 0.5
- **Mixed Precision**: PyTorch AMP with GradScaler
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Classification Threshold**: 0.4

#### Hyperparameter Search
- **Learning Rates**: [5e-5, 1e-5, 7e-6]
- **Focal Loss Gamma**: [2, 3]
- **Training Epochs**: 3 (hyperparameter tuning) + up to 5 (extended training)
- **Early Stopping**: Patience of 2 epochs based on validation F1-score

#### Best Configurations

| Model | Learning Rate | Focal Loss Gamma | Best Epoch | Validation F1 |
|-------|---------------|------------------|------------|---------------|
| **RoBERTa** | 7e-6 | 2 | 3 | 0.6046 |
| **XLNet** | 5e-5 | 3 | 2 | 0.5872 |
| **ELECTRA** | 1e-5 | 2 | 3 | 0.5997 |

### Class Imbalance Solutions

#### 1. Multi-Label Random Oversampling (ML-ROS)
- **Algorithm**: Custom implementation preserving emotion co-occurrence patterns
- **Target Strategy**: Median count × sampling ratio (default: 0.5)
- **Max Ratio**: 3× original count to prevent excessive duplication
- **Results**: Dataset size increased from 43,410 to 44,987 samples (+1,577)
- **Preservation**: Maintained 711 unique label combinations

#### 2. Focal Loss Implementation
- **Formula**: `FL(pt) = -α(1-pt)^γ log(pt)`
- **Purpose**: Down-weight easy examples, focus on hard-to-classify instances
- **Adaptation**: Modified for multi-label classification with logits
- **Parameters**: α (balancing factor), γ (focusing parameter)

#### 3. Evaluation Strategy
- **Weighted Metrics**: Account for class distribution in final scores
- **Macro Averaging**: Equal weight to all emotions regardless of frequency
- **Per-class Analysis**: Individual precision, recall, F1 for each emotion

#### Controlled Oversampling Algorithm
```python
# Pseudocode for ML-ROS implementation
for emotion in underrepresented_emotions:
    if current_count < median_count:
        target_count = min(median * 0.5, current_count * 3)
        oversample_instances(emotion, target_count)
