import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from training import evaluate


def calculate_metrics(probs, labels, emotion_names, threshold=0.4):
    preds = (probs > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    report = classification_report(labels, preds, target_names=emotion_names, zero_division=0)
    macro = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    micro = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)

    return {
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'classification_report': report,
        'macro_precision': macro[0],
        'macro_recall': macro[1],
        'macro_f1': macro[2],
        'micro_precision': micro[0],
        'micro_recall': micro[1],
        'micro_f1': micro[2],
    }


def evaluate_and_print(model, dataloader, criterion, device, emotion_names, model_name, threshold=0.4):
    loss, probs, labels = evaluate(model, dataloader, criterion, device)
    preds = (probs > threshold).astype(int)
    metrics = calculate_metrics(probs, labels, emotion_names, threshold=threshold)

    print(f"\n{model_name} Results (Threshold: {threshold}):")
    print(f"Loss: {loss:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")

    print("\nMacro-averaged metrics:")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1-score: {metrics['macro_f1']:.4f}")

    print("\nMicro-averaged metrics:")
    print(f"Precision: {metrics['micro_precision']:.4f}")
    print(f"Recall: {metrics['micro_recall']:.4f}")
    print(f"F1-score: {metrics['micro_f1']:.4f}")

    print("\nClassification Report:")
    print(metrics['classification_report'])

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    print("\nDetailed per-class metrics:")
    for i, emotion in enumerate(emotion_names):
        print(f"{emotion}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")

    metrics['loss'] = loss
    metrics['probs'] = probs
    metrics['labels'] = labels
    metrics['preds'] = preds
    metrics['threshold'] = threshold
    metrics['per_class_precision'] = precision
    metrics['per_class_recall'] = recall
    metrics['per_class_f1'] = f1
    metrics['per_class_support'] = support

    return metrics
