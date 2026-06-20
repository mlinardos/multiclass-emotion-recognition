import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def load_go_emotions():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    emotion_names = ds['train'].features['labels'].feature.names
    num_labels = len(emotion_names)
    return ds, emotion_names, num_labels


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, num_labels):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = texts
        self.labels = labels
        self.num_labels = num_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = [int(label) for label in self.labels[idx]]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        label_vector = torch.zeros(self.num_labels)
        label_vector[labels] = 1

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }
