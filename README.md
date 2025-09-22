### Training Strategy

#### Optimization Parameters
&#8203; **Optimizer**: AdamW  
&#8203; **Weight Decay**: 0.1  
&#8203; **Dropout**: 0.5  
&#8203; **Mixed Precision**: PyTorch AMP with GradScaler  
&#8203; **Batch Size**: 16  
&#8203; **Max Sequence Length**: 128 tokens  
&#8203; **Classification Threshold**: 0.4  

#### Hyperparameter Search
&#8203; **Learning Rates**: [5e-5, 1e-5, 7e-6]  
&#8203; **Focal Loss Gamma**: [2, 3]  
&#8203; **Training Epochs**: 3 (hyperparameter tuning) + up to 5 (extended training)  
&#8203; **Early Stopping**: Patience of 2 epochs based on validation F1-score  

#### Best Configurations

| Model | Learning Rate | Focal Loss Gamma | Best Epoch | Validation F1 |
|-------|---------------|------------------|------------|---------------|
| **RoBERTa** | 7e-6 | 2 | 3 | 0.6046 |
| **XLNet** | 5e-5 | 3 | 2 | 0.5872 |
| **ELECTRA** | 1e-5 | 2 | 3 | 0.5997 |

### Class Imbalance Solutions

#### Multi-Label Random Oversampling (ML-ROS)
&#8203; **Algorithm**: Custom implementation preserving emotion co-occurrence patterns  
&#8203; **Target Strategy**: Median count × sampling ratio (default: 0.5)  
&#8203; **Max Ratio**: 3× original count to prevent excessive duplication  
&#8203; **Results**: Dataset size increased from 43,410 to 44,987 samples (+1,577)  
&#8203; **Preservation**: Maintained 711 unique label combinations  

#### Focal Loss Implementation
&#8203; **Formula**: `FL(pt) = -α(1-pt)^γ log(pt)`  
&#8203; **Purpose**: Down-weight easy examples, focus on hard-to-classify instances  
&#8203; **Adaptation**: Modified for multi-label classification with logits  
&#8203; **Parameters**: α (balancing factor), γ (focusing parameter)  

#### Evaluation Strategy
&#8203; **Weighted Metrics**: Account for class distribution in final scores  
&#8203; **Macro Averaging**: Equal weight to all emotions regardless of frequency  
&#8203; **Per-class Analysis**: Individual precision, recall, F1 for each emotion  

### Key Innovations

#### Controlled Oversampling Algorithm
&#8203; Maintains natural emotion co-occurrence patterns while boosting minority classes  
&#8203; Preserves authentic label relationships from original dataset  
&#8203; Avoids synthetic data generation that might introduce artifacts  

#### Custom Focal Loss for Multi-Label
&#8203; Adapted binary cross-entropy focal loss for 28-class multi-label scenario  
&#8203; Addresses both class imbalance and hard example focus simultaneously  
&#8203; Numerically stable implementation with logits and epsilon smoothing  

#### No Preprocessing Philosophy
&#8203; Preserves emotional nuances in raw text based on recent research findings  
&#8203; Special handling with `[NAME]` and `[RELIGION]` tokens added to vocabularies  
&#8203; Maintains authenticity of Reddit comment language patterns  

#### Architecture Modifications
&#8203; Unified [CLS] token pooling strategy across all three models  
&#8203; Consistent 0.5 dropout rate before final classification layer  
&#8203; 28 independent sigmoid units for multi-label prediction output