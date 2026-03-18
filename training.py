import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from losses import FocalLoss
from visualization import hyperparameter_tuning_plots


def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    return total_loss / len(dataloader), all_probs, all_labels


def hyperparameter_tuning(
    model_class, tokenizer, model_name,
    train_loader, val_loader,
    num_labels, device, threshold,
    gammas, learning_rates, num_epochs,
    dropout_rate=0.5, weight_decay=0.1,
):
    best_params = None
    best_val_f1 = 0
    best_model_state = None
    best_history = None
    best_epoch = 0
    results = []

    for gamma in gammas:
        for lr in learning_rates:
            print(f"Training {model_name} with gamma={gamma}, lr={lr}")
            model = model_class(num_labels, dropout_rate=dropout_rate).to(device)
            model.resize_token_embeddings(len(tokenizer))

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = FocalLoss(gamma=gamma)
            scaler = GradScaler()

            train_losses = []
            val_losses = []
            val_f1_scores = []

            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
                val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
                val_preds = (val_probs > threshold).astype(int)
                val_f1 = f1_score(val_labels, val_preds, average='weighted')

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_params = {'gamma': gamma, 'lr': lr}
                    best_model_state = model.state_dict()
                    best_history = {
                        'train_losses': train_losses.copy(),
                        'val_losses': val_losses.copy(),
                        'val_f1_scores': val_f1_scores.copy(),
                    }
                    best_epoch = epoch + 1

            results.append({
                'gamma': gamma,
                'lr': lr,
                'losses': train_losses,
                'val_losses': val_losses,
                'f1_scores': val_f1_scores,
            })

    print(f"Best parameters: {best_params}")
    print(f"Best F1 score: {best_val_f1}")
    print(f"Best epoch: {best_epoch}")

    hyperparameter_tuning_plots(results, model_name, num_epochs)
    torch.save(best_model_state, f'{model_name.lower()}_best_model.pth')

    return best_params, best_model_state, best_history, best_epoch


def train_final_model(
    model_class, tokenizer, model_name,
    best_model_state, best_params,
    train_loader, val_loader,
    num_labels, device, threshold,
    best_history, best_epoch,
    num_epochs=5, patience=2,
    dropout_rate=0.5, weight_decay=0.1,
):
    model = model_class(num_labels, dropout_rate=dropout_rate).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(best_model_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=weight_decay)
    criterion = FocalLoss(gamma=best_params['gamma'])
    scaler = GradScaler()

    print(f"Further training of the best {model_name} model found in hyperparameter tuning")

    best_val_f1 = max(best_history['val_f1_scores'])
    counter = 0
    train_losses = best_history['train_losses']
    val_losses = best_history['val_losses']
    val_f1_scores = best_history['val_f1_scores']

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_preds = (val_probs > threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {best_epoch + epoch + 1}/{best_epoch + num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'{model_name.lower()}_best_model.pth')
            print(f"New best model saved with F1 score: {best_val_f1:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience and epoch + 1 < num_epochs:
                print(f"Early stopping triggered at epoch {best_epoch + epoch + 1}")
                break

    total_epochs = best_epoch + epoch + 1
    return model, criterion, train_losses, val_losses, val_f1_scores, total_epochs
