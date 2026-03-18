import random
from collections import Counter


def ml_ros(texts, labels, emotion_names, sampling_strategy=0.5, max_ratio=3, random_seed=42):
    random.seed(random_seed)

    label_counts = Counter()
    for label_list in labels:
        label_counts.update(label_list)

    median_count = sorted(label_counts.values())[len(label_counts) // 2]

    oversampled_texts = texts.copy()
    oversampled_labels = labels.copy()

    for label in range(len(emotion_names)):
        current_count = label_counts[label]
        if current_count < median_count:
            target_count = min(int(median_count * sampling_strategy), current_count * max_ratio)
            n_to_sample = max(0, int(target_count - current_count))

            if n_to_sample > 0:
                indices = [i for i, l in enumerate(labels) if label in l]
                sampled_indices = random.choices(indices, k=n_to_sample)

                for idx in sampled_indices:
                    oversampled_texts.append(texts[idx])
                    oversampled_labels.append(labels[idx].copy())

    return oversampled_texts, oversampled_labels


def print_detailed_distribution(labels, emotion_names):
    label_counts = Counter()
    combo_counts = Counter()
    label_per_sample = []

    for label_list in labels:
        label_counts.update(label_list)
        combo_counts[tuple(sorted(label_list))] += 1
        label_per_sample.append(len(label_list))

    total_samples = len(labels)
    total_labels = sum(label_counts.values())
    multi_label_samples = sum(1 for count in label_per_sample if count > 1)

    print(f"Dataset Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Total label occurrences: {total_labels}")
    print(f"Average labels per sample: {total_labels / total_samples:.2f}")
    print(f"Samples with multiple labels: {multi_label_samples} ({multi_label_samples / total_samples:.2%})")
    print(f"Unique label combinations: {len(combo_counts)}")

    print("\nLabel Distribution:")
    for emotion, count in label_counts.most_common():
        print(f"{emotion_names[emotion]:<15} {count:5d} ({count / total_samples:.2%})")

    print("\nLabel Count Distribution:")
    for count in range(1, max(label_per_sample) + 1):
        samples = sum(1 for c in label_per_sample if c == count)
        print(f"{count} label: {samples:5d} samples ({samples / total_samples:.2%})")
