# multiclass-emotion-recognition
An implementation of multi-label and multi-class emotion classification using transformer models on the Go_Emotions dataset. This project compares RoBERTa, XLNet, and ELECTRA architectures for recognizing 28 emotion categories in Reddit comments.

### Training Strategy

#### Optimization Parameters
The models used AdamW optimizer with a weight decay of 0.1 and dropout rate of 0.5. Training employed PyTorch AMP with GradScaler for mixed precision, using a batch size of 16 and maximum sequence length of 128 tokens. All models used a classification threshold of 0.4 for final predictions.

#### Hyperparameter Search
The search space included learning rates of 5e-5, 1e-5, and 7e-6, combined with Focal Loss gamma values of 2 and 3. Models underwent 3 epochs during hyperparameter tuning followed by up to 5 additional epochs in extended training. Early stopping was implemented with a patience of 2 epochs based on validation F1-score improvements.

#### Best Configurations

| Model | Learning Rate | Focal Loss Gamma | Best Epoch | Validation F1 |
|-------|---------------|------------------|------------|---------------|
| **RoBERTa** | 7e-6 | 2 | 3 | 0.6046 |
| **XLNet** | 5e-5 | 3 | 2 | 0.5872 |
| **ELECTRA** | 1e-5 | 2 | 3 | 0.5997 |

### Class Imbalance Solutions

The project addressed class imbalance through three complementary approaches. First, a custom Multi-Label Random Oversampling (ML-ROS) algorithm was implemented to preserve emotion co-occurrence patterns while increasing representation of minority classes. The algorithm targeted a median count multiplied by a sampling ratio of 0.5, with a maximum ratio of 3× the original count to prevent excessive duplication. This approach increased the dataset size from 43,410 to 44,987 samples, adding 1,577 instances while maintaining all 711 unique label combinations.

Second, a custom Focal Loss implementation was developed specifically for multi-label classification. The loss function follows the formula `FL(pt) = -α(1-pt)^γ log(pt)`, where α provides class balancing and γ focuses attention on hard-to-classify instances. The implementation was adapted for multi-label scenarios using logits for numerical stability.

Third, the evaluation strategy employed weighted metrics to account for class distribution, macro averaging to give equal weight to all emotions regardless of frequency, and detailed per-class analysis providing individual precision, recall, and F1-scores for each emotion category.

### Key Decisions

The controlled oversampling algorithm represents a novel approach to multi-label imbalance, maintaining natural emotion co-occurrence patterns while boosting minority class representation. Unlike synthetic oversampling techniques, this method preserves authentic label relationships from the original dataset.

The custom Focal Loss adaptation for multi-label classification addresses both class imbalance and hard example focus simultaneously. The implementation uses numerically stable computations with logits and epsilon smoothing to prevent numerical instabilities during training.

The decision to avoid preprocessing was based on recent research by Siino et al. (2024) showing that minimal preprocessing often performs comparably to extensive text cleaning for transformer models. This approach preserved emotional nuances in raw text while adding special tokens `[NAME]` and `[RELIGION]` to handle anonymized content in the dataset.

Architecture modifications ensured fair comparison across all three models by implementing unified pooling using [CLS] token representation, consistent dropout of 0.5 before the final classification layer, and identical 28-unit output layers with independent sigmoid units for multi-label prediction.
