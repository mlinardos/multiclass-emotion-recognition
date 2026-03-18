import warnings
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, XLNetTokenizer, ElectraTokenizer

import config
from data_loading import load_go_emotions, EmotionDataset
from data_analysis import run_analysis
from preprocessing import ml_ros, print_detailed_distribution
from models import (
    RoBERTaForMultiLabelClassification,
    XLNetForMultiLabelClassification,
    ELECTRAForMultiLabelClassification,
)
from training import hyperparameter_tuning, train_final_model
from metrics import evaluate_and_print
from visualization import final_model_plots, plot_performance_curves

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def setup_tokenizer(tokenizer_class, pretrained_name):
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)
    special_tokens_dict = {'additional_special_tokens': config.SPECIAL_TOKENS}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to {pretrained_name} tokenizer")
    return tokenizer


def build_dataloaders(train_texts, train_labels, ds, tokenizer, num_labels):
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, config.MAX_LEN, num_labels)
    val_dataset = EmotionDataset(list(ds['validation']['text']), list(ds['validation']['labels']), tokenizer, config.MAX_LEN, num_labels)
    test_dataset = EmotionDataset(list(ds['test']['text']), list(ds['test']['labels']), tokenizer, config.MAX_LEN, num_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def run_pipeline(model_class, tokenizer, model_name, train_loader, val_loader, test_loader, num_labels, device):
    best_params, best_model_state, best_history, best_epoch = hyperparameter_tuning(
        model_class=model_class,
        tokenizer=tokenizer,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_labels=num_labels,
        device=device,
        threshold=config.THRESHOLD,
        gammas=config.GAMMAS,
        learning_rates=config.LEARNING_RATES,
        num_epochs=config.TUNE_EPOCHS,
        dropout_rate=config.DROPOUT_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    model, criterion, train_losses, val_losses, val_f1_scores, total_epochs = train_final_model(
        model_class=model_class,
        tokenizer=tokenizer,
        model_name=model_name,
        best_model_state=best_model_state,
        best_params=best_params,
        train_loader=train_loader,
        val_loader=val_loader,
        num_labels=num_labels,
        device=device,
        threshold=config.THRESHOLD,
        best_history=best_history,
        best_epoch=best_epoch,
        num_epochs=config.FINAL_EPOCHS,
        patience=config.PATIENCE,
        dropout_rate=config.DROPOUT_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    model.load_state_dict(torch.load(f'{model_name.lower()}_best_model.pth'))
    final_model_plots(total_epochs, best_epoch, train_losses, val_losses, val_f1_scores, model_name)

    metrics = evaluate_and_print(model, test_loader, criterion, device, emotion_names, model_name, threshold=config.THRESHOLD)
    plot_performance_curves(metrics['labels'], metrics['preds'], model_name)

    return metrics


if __name__ == '__main__':
    set_seed(config.SEED)

    print("Is CUDA available:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ds, emotion_names, num_labels = load_go_emotions()
    print(f"Loaded {num_labels} emotion classes: {emotion_names}")

    # --- Analysis ---
    run_analysis(ds, emotion_names)

    # --- Oversampling ---
    oversampled_texts, oversampled_labels = ml_ros(
        list(ds['train']['text']), list(ds['train']['labels']), emotion_names,
        sampling_strategy=config.SAMPLING_STRATEGY,
        max_ratio=config.MAX_RATIO,
    )
    print("Original dataset distribution:")
    print_detailed_distribution(list(ds['train']['labels']), emotion_names)
    print("\nOversampled dataset distribution:")
    print_detailed_distribution(oversampled_labels, emotion_names)

    # --- RoBERTa ---
    print("\n" + "=" * 60)
    print("ROBERTA")
    print("=" * 60)
    roberta_tokenizer = setup_tokenizer(RobertaTokenizer, 'roberta-base')
    roberta_train_loader, roberta_val_loader, roberta_test_loader = build_dataloaders(
        oversampled_texts, oversampled_labels, ds, roberta_tokenizer, num_labels
    )
    roberta_metrics = run_pipeline(
        RoBERTaForMultiLabelClassification, roberta_tokenizer, 'RoBERTa',
        roberta_train_loader, roberta_val_loader, roberta_test_loader,
        num_labels, device,
    )

    # --- XLNet ---
    print("\n" + "=" * 60)
    print("XLNET")
    print("=" * 60)
    xlnet_tokenizer = setup_tokenizer(XLNetTokenizer, 'xlnet-base-cased')
    xlnet_train_loader, xlnet_val_loader, xlnet_test_loader = build_dataloaders(
        oversampled_texts, oversampled_labels, ds, xlnet_tokenizer, num_labels
    )
    xlnet_metrics = run_pipeline(
        XLNetForMultiLabelClassification, xlnet_tokenizer, 'XLNet',
        xlnet_train_loader, xlnet_val_loader, xlnet_test_loader,
        num_labels, device,
    )

    # --- ELECTRA ---
    print("\n" + "=" * 60)
    print("ELECTRA")
    print("=" * 60)
    electra_tokenizer = setup_tokenizer(ElectraTokenizer, 'google/electra-base-discriminator')
    electra_train_loader, electra_val_loader, electra_test_loader = build_dataloaders(
        oversampled_texts, oversampled_labels, ds, electra_tokenizer, num_labels
    )
    electra_metrics = run_pipeline(
        ELECTRAForMultiLabelClassification, electra_tokenizer, 'ELECTRA',
        electra_train_loader, electra_val_loader, electra_test_loader,
        num_labels, device,
    )
