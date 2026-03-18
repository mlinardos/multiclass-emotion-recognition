import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc


def plot_distribution(df, title):
    plt.figure(figsize=(20, 10))
    sns.barplot(x='emotion', y='count', data=df.sort_values('count', ascending=False))
    plt.title(f'Distribution of Emotions in {title} Dataset', fontsize=20)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'distribution_{title.lower()}.png')
    plt.show()
    plt.close()


def plot_performance_curves(y_true, y_pred, model_name):
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    avg_precision = average_precision_score(y_true, y_pred, average="micro")

    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Micro-average Precision-Recall curve\nAverage Precision = {avg_precision:.2f}')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='b', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Micro-average ROC curve\nAUC = {roc_auc:.2f}')

    plt.tight_layout()
    plt.savefig(f'{model_name}_performance_curves.png')
    plt.show()
    plt.close()


def hyperparameter_tuning_plots(results, model_name, num_epochs):
    plt.figure(figsize=(12, 6))
    for result in results:
        label = f"gamma={result['gamma']}, lr={result['lr']}"
        plt.plot(range(1, num_epochs + 1), result['losses'], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'{model_name} Training Loss During Hyperparameter Tuning')
    plt.legend()
    plt.savefig(f'{model_name}_hyperparameter_tuning_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    for result in results:
        label = f"gamma={result['gamma']}, lr={result['lr']}"
        plt.plot(range(1, num_epochs + 1), result['f1_scores'], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} F1 Score During Hyperparameter Tuning')
    plt.legend()
    plt.savefig(f'{model_name}_hyperparameter_tuning_f1.png')
    plt.show()
    plt.close()


def final_model_plots(total_epochs, best_epoch, train_losses, val_losses, val_f1_scores, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, total_epochs + 1), val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch During Hyperparameter Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{model_name}_training_validation_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_epochs + 1), val_f1_scores, label='F1 Score')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch During Hyperparameter Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} Validation F1 Score')
    plt.legend()
    plt.savefig(f'{model_name}_validation_f1_score.png')
    plt.show()
    plt.close()
