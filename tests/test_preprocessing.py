from preprocessing import ml_ros
# Three emotions named a, b, c -> label ids 0, 1, 2.
EMOTION_NAMES = ["a", "b", "c"]


def make_imbalanced_dataset():
    """Build a dataset where label 2 is a clear minority.

    Label counts: label 0 -> 5, label 1 -> 5, label 2 -> 1.
    The median of [1, 5, 5] is 5, so only label 2 is below the median.
    """
    texts = [f"text_{i}" for i in range(11)]
    labels = (
        [[0], [0], [0], [0], [0]]   # 5 samples carrying label 0
        + [[1], [1], [1], [1], [1]]  # 5 samples carrying label 1
        + [[2]]                      # 1 sample carrying label 2 (the minority)
    )
    return texts, labels


def test_minority_label_gets_oversampled():
    texts, labels = make_imbalanced_dataset()

    new_texts, new_labels = ml_ros(texts, labels, EMOTION_NAMES)

    # Target for label 2 is min(int(5 * 0.5), 1 * 3) = min(2, 3) = 2.
    # It currently has 1 sample, so exactly one extra copy is added.
    assert len(new_texts) == len(texts) + 1
    assert len(new_labels) == len(labels) + 1

    # The single added sample is the one carrying the minority label.
    added_labels = new_labels[len(labels):]
    assert added_labels == [[2]]


def test_texts_and_labels_remain_the_same_length():
    texts, labels = make_imbalanced_dataset()

    new_texts, new_labels = ml_ros(texts, labels, EMOTION_NAMES)

    assert len(new_texts) == len(new_labels)


def test_same_seed_gives_the_same_result():
    texts, labels = make_imbalanced_dataset()

    first_run = ml_ros(texts, labels, EMOTION_NAMES, random_seed=123)
    second_run = ml_ros(texts, labels, EMOTION_NAMES, random_seed=123)

    assert first_run == second_run


def test_already_balanced_dataset_is_left_alone():
    # Both labels appear once, so neither is below the median -> no oversampling.
    texts = ["happy text", "sad text"]
    labels = [[0], [1]]

    new_texts, new_labels = ml_ros(texts, labels, ["a", "b"])

    assert new_texts == texts
    assert new_labels == labels


def test_original_inputs_are_not_modified():
    texts, labels = make_imbalanced_dataset()
    texts_before = list(texts)
    labels_before = [list(label) for label in labels]

    ml_ros(texts, labels, EMOTION_NAMES)

    assert texts == texts_before
    assert labels == labels_before


def test_original_samples_are_kept_at_the_front():
    texts, labels = make_imbalanced_dataset()

    new_texts, _ = ml_ros(texts, labels, EMOTION_NAMES)

    # Oversampling only appends, so the originals stay as the leading prefix.
    assert new_texts[: len(texts)] == texts
