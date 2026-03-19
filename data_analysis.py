import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

os.makedirs('figures', exist_ok=True)


def count_emotions(split, emotion_names):
    emotion_counts = {emotion: 0 for emotion in emotion_names}
    for labels in split['labels']:
        for label in labels:
            emotion_counts[emotion_names[label]] += 1
    return emotion_counts


def calculate_jaccard_similarity(split, emotion_names):
    n = len(emotion_names)
    cooccur = np.zeros((n, n))
    occur = np.zeros(n)

    for labels in split['labels']:
        for i in labels:
            occur[i] += 1
            for j in labels:
                cooccur[i][j] += 1

    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = cooccur[i][j]
            union = occur[i] + occur[j] - intersection
            jaccard[i][j] = intersection / union if union > 0 else 0

    return pd.DataFrame(jaccard, index=emotion_names, columns=emotion_names)


def calculate_combined_jaccard_similarity(splits, emotion_names):
    n = len(emotion_names)
    cooccur = np.zeros((n, n))
    occur = np.zeros(n)

    for split in splits:
        for labels in split['labels']:
            for i in labels:
                occur[i] += 1
                for j in labels:
                    cooccur[i][j] += 1

    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = cooccur[i][j]
            union = occur[i] + occur[j] - intersection
            jaccard[i][j] = intersection / union if union > 0 else 0

    return pd.DataFrame(jaccard, index=emotion_names, columns=emotion_names)


def get_top_cooccurrences(jaccard_df, n=15):
    mask = np.triu(np.ones(jaccard_df.shape), k=1).astype(bool)
    upper_tri = jaccard_df.where(mask)
    stacked = upper_tri.stack().reset_index()
    stacked.columns = ['emotion1', 'emotion2', 'similarity']
    return stacked.sort_values('similarity', ascending=False).head(n)


def check_duplicates(df):
    exact_duplicates = df[df.duplicated(keep=False)]
    print(f"\nNumber of exact duplicate entries: {len(exact_duplicates)}")

    content_duplicates = df.groupby(['text', 'labels']).filter(
        lambda x: len(x) > 1 and len(x['id'].unique()) > 1
    )
    print(f"\nNumber of entries with same content but different IDs: {len(content_duplicates)}")

    return exact_duplicates, content_duplicates


def check_split(ds, split_name, emotion_names):
    print(f"\nChecking {split_name} split:")
    split = ds[split_name]

    df = pd.DataFrame({
        'id': split['id'],
        'text': split['text'],
        'labels': [tuple(labels) for labels in split['labels']],
    })

    print("Missing values:")
    print(df.isnull().sum())
    print(f"Empty text entries: {df['text'].str.strip().eq('').sum()}")
    print(f"Entries with no labels: {sum(len(labels) == 0 for labels in df['labels'])}")
    print(f"Entries with invalid label indices: {sum(any(label >= len(emotion_names) for label in labels) for labels in df['labels'])}")

    check_duplicates(df)

    all_labels = [label for labels in df['labels'] for label in labels]
    print("\nLabel distribution:")
    for emotion_id, count in Counter(all_labels).most_common():
        print(f"{emotion_names[emotion_id]}: {count}")

    text_lengths = df['text'].str.len()
    print(f"\nText length statistics:\n{text_lengths.describe()}")


def check_cross_split_overlap(ds):
    def get_id_text_label_tuples(split):
        return set(zip(split['id'], split['text'], map(tuple, split['labels'])))

    train_tuples = get_id_text_label_tuples(ds['train'])
    val_tuples = get_id_text_label_tuples(ds['validation'])
    test_tuples = get_id_text_label_tuples(ds['test'])

    print(f"Exact overlap (train/val): {len(train_tuples.intersection(val_tuples))}")
    print(f"Exact overlap (train/test): {len(train_tuples.intersection(test_tuples))}")
    print(f"Exact overlap (val/test): {len(val_tuples.intersection(test_tuples))}")

    def get_text_label_pairs(tuples):
        return set((text, labels) for _, text, labels in tuples)

    train_pairs = get_text_label_pairs(train_tuples)
    val_pairs = get_text_label_pairs(val_tuples)
    test_pairs = get_text_label_pairs(test_tuples)

    print(f"\nContent overlap (train/val): {len(train_pairs.intersection(val_pairs))}")
    print(f"Content overlap (train/test): {len(train_pairs.intersection(test_pairs))}")
    print(f"Content overlap (val/test): {len(val_pairs.intersection(test_pairs))}")

    def get_text_to_id_label_dict(tuples):
        text_dict = {}
        for id_, text, labels in tuples:
            if text not in text_dict:
                text_dict[text] = set()
            text_dict[text].add((id_, labels))
        return text_dict

    train_dict = get_text_to_id_label_dict(train_tuples)
    val_dict = get_text_to_id_label_dict(val_tuples)
    test_dict = get_text_to_id_label_dict(test_tuples)

    def count_text_only_overlap(dict1, dict2):
        return sum(1 for text in dict1.keys() & dict2.keys() if dict1[text] != dict2[text])

    print(f"\nText-only overlap (train/val): {count_text_only_overlap(train_dict, val_dict)}")
    print(f"Text-only overlap (train/test): {count_text_only_overlap(train_dict, test_dict)}")
    print(f"Text-only overlap (val/test): {count_text_only_overlap(val_dict, test_dict)}")


def find_different_labels(ds, emotion_names):
    def get_text_label_pairs(split):
        return {(text, tuple(sorted(labels))) for text, labels in zip(split['text'], split['labels'])}

    train_pairs = get_text_label_pairs(ds['train'])
    val_pairs = get_text_label_pairs(ds['validation'])
    test_pairs = get_text_label_pairs(ds['test'])

    def _find(split1_pairs, split2_pairs, split1_name, split2_name):
        split1_texts = {text for text, _ in split1_pairs}
        split2_texts = {text for text, _ in split2_pairs}
        common_texts = split1_texts.intersection(split2_texts)

        different_labels = []
        for text in common_texts:
            labels1 = [labels for t, labels in split1_pairs if t == text]
            labels2 = [labels for t, labels in split2_pairs if t == text]
            if labels1 and labels2 and set(labels1[0]) != set(labels2[0]):
                different_labels.append((text, labels1[0], labels2[0]))

        print(f"\nTexts with different labels between {split1_name} and {split2_name}:")
        for text, l1, l2 in different_labels:
            print(f"Text: {text}")
            print(f"{split1_name} labels: {[emotion_names[l] for l in l1]}")
            print(f"{split2_name} labels: {[emotion_names[l] for l in l2]}")
            print("---")
        return different_labels

    train_val_diff = _find(train_pairs, val_pairs, "Train", "Validation")
    train_test_diff = _find(train_pairs, test_pairs, "Train", "Test")
    val_test_diff = _find(val_pairs, test_pairs, "Validation", "Test")

    print(f"\nTotal cases of different labels:")
    print(f"Train-Validation: {len(train_val_diff)}")
    print(f"Train-Test: {len(train_test_diff)}")
    print(f"Validation-Test: {len(val_test_diff)}")


def count_special_tokens(texts):
    pattern = r'\[(NAME|RELIGION)\]'
    token_counter = Counter()
    for text in texts:
        matches = re.findall(pattern, text)
        token_counter.update(matches)
    return token_counter


def analyze_special_tokens(ds):
    for split_name in ['train', 'validation', 'test']:
        token_counts = count_special_tokens(ds[split_name]['text'])
        print(f"\nSpecial tokens in {split_name} split:")
        for token, count in token_counts.most_common():
            print(f"[{token}]: {count}")
        print(f"Total special tokens: {sum(token_counts.values())}")

    all_texts = list(ds['train']['text']) + list(ds['validation']['text']) + list(ds['test']['text'])
    all_token_counts = count_special_tokens(all_texts)
    print("\nSpecial tokens in all splits combined:")
    for token, count in all_token_counts.most_common():
        print(f"[{token}]: {count}")
    print(f"Total special tokens across all splits: {sum(all_token_counts.values())}")
    print(f"Unique special tokens across all splits: {len(all_token_counts)}")


def run_analysis(ds, emotion_names):
    train_counts = count_emotions(ds['train'], emotion_names)
    val_counts = count_emotions(ds['validation'], emotion_names)
    test_counts = count_emotions(ds['test'], emotion_names)

    total_counts = {e: train_counts[e] + val_counts[e] + test_counts[e] for e in emotion_names}
    df_counts = pd.DataFrame.from_dict(total_counts, orient='index', columns=['count'])
    df_counts = df_counts.sort_values('count', ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x=df_counts.index, y='count', data=df_counts)
    plt.title('Distribution of Emotions', fontsize=20)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/distribution_all.png')
    plt.show()
    plt.close()

    from visualization import plot_distribution
    for counts, title in [(train_counts, 'Train'), (val_counts, 'Validation'), (test_counts, 'Test')]:
        df = pd.DataFrame.from_dict(counts, orient='index', columns=['count']).reset_index().rename(columns={'index': 'emotion'})
        plot_distribution(df, title)

    train_jaccard = calculate_jaccard_similarity(ds['train'], emotion_names)
    val_jaccard = calculate_jaccard_similarity(ds['validation'], emotion_names)
    test_jaccard = calculate_jaccard_similarity(ds['test'], emotion_names)

    for jaccard, name in [(train_jaccard, 'Train'), (val_jaccard, 'Validation'), (test_jaccard, 'Test')]:
        print(f"\nTop 15 co-occurring emotions - {name} set:")
        print(get_top_cooccurrences(jaccard))

    combined_jaccard = calculate_combined_jaccard_similarity(
        [ds['train'], ds['validation'], ds['test']], emotion_names
    )
    top_combined = get_top_cooccurrences(combined_jaccard, n=15)
    formatted = top_combined.copy()
    formatted['similarity'] = formatted['similarity'].apply(lambda x: f"{x:.6f}")
    print("\nTop 15 co-occurring emotions (all sets combined):")
    print(formatted.to_string(index=False))

    for split in ['train', 'validation', 'test']:
        check_split(ds, split, emotion_names)

    check_cross_split_overlap(ds)
    find_different_labels(ds, emotion_names)
    analyze_special_tokens(ds)
