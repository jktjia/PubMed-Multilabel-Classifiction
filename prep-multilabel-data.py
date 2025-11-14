import argparse
from typing import Tuple
import pandas as pd

# fix the randomness to ensure reproducibility
random_seed = 42


def _parse_args():
    """
    Command-line arguments to the system. Allows for optional specification of location of file containing full data set.

    Returns: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="trainer.py")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pubmed-multi-label-text-classification-dataset-processed.csv",
        help="path to full data set",
    )
    args = parser.parse_args()
    return args


def _split_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, dev, test datasets.

    Args:
        path (str): path to full dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: training data, dev data, test data
    """
    full_data = pd.read_csv(path)

    train_data_full = full_data.sample(frac=0.8, random_state=random_seed)
    test_data = full_data.drop(train_data_full.index)

    train_data = train_data_full.sample(frac=0.8, random_state=random_seed)
    dev_data = train_data_full.drop(train_data.index)

    return train_data, dev_data, test_data


def _summarize_dataset(data: pd.DataFrame, processed: bool = True):
    num_entries = len(data)
    print(num_entries, "entries")
    print()

    if processed:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]
    else:
        labels = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "V",
            "Z",
        ]

    labels_only = data[labels]
    print(len(labels), "labels")

    num_labels = labels_only.sum(axis=1)
    mean_label_count = num_labels.mean()
    print("Average labels per example:", mean_label_count)
    print()

    label_counts = labels_only.apply(pd.Series.value_counts).transpose()
    label_counts["percent"] = label_counts[1] / num_entries
    print(label_counts["percent"])
    print()


def _create_data_files(
    train_exs: pd.DataFrame, dev_exs: pd.DataFrame, test_exs: pd.DataFrame
):
    train_exs.to_csv("data/train_data.csv")
    dev_exs.to_csv("data/dev_data.csv")
    test_exs.to_csv("data/test_data.csv")


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    train_exs, dev_exs, test_exs = _split_dataset(args.data_path)

    print("Training Data")
    train_summary = _summarize_dataset(train_exs)

    print("Dev Data")
    dev_summary = _summarize_dataset(dev_exs)

    # print("Test Data")
    # tes_summary = _summarize_dataset(test_exs)

    _create_data_files(train_exs, dev_exs, test_exs)
