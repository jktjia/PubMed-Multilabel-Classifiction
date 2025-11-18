import argparse
from typing import Tuple
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from utils import default_labels, processed_labels

nltk.download("punkt")

# fix the randomness to ensure reproducibility
random_seed = 42


def _parse_args():
    """
    Command-line arguments to the system. Allows for optional specification of location of file containing full data set.

    Returns: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="prep_multilabel_data.py")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pubmed-multi-label-text-classification-dataset-processed.csv",
        help="path to full data set",
    )
    parser.add_argument("--create_files", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--summary",
        type=str,
        default="ALL",
        help="dataset to summarize (ALL, TRAIN, DEV, or TEST)",
    )
    args = parser.parse_args()
    return args


def _split_dataset(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, dev, test datasets.

    Args:
        data (pd.DataFrame): full dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: training data, dev data, test data
    """
    train_data_full = data.sample(frac=0.8, random_state=random_seed)
    test_data = data.drop(train_data_full.index)

    train_data = train_data_full.sample(frac=0.8, random_state=random_seed)
    dev_data = train_data_full.drop(train_data.index)

    return train_data, dev_data, test_data


def _summarize_dataset(data: pd.DataFrame, processed: bool = True):
    """
    Print out summary statistics for the dataset. Summary statistics include number of entries, number of labels, average number of labels per entry, percent of entries with each label.

    Args:
        data (pd.DataFrame): Dataset to summarize.
        processed (bool, optional): Whether or not the data is from the processed dataset. This affects what labels are available. Defaults to True.
    """
    num_entries = len(data)
    print(num_entries, "entries")
    print()

    data["abstract_len"] = data.apply(
        lambda row: len(word_tokenize(row["abstractText"])), axis=1
    )
    abstract_lens = data["abstract_len"]
    print("Example abstract lengths:")
    print(abstract_lens.describe())
    print()

    if processed:
        labels = processed_labels
    else:
        labels = default_labels

    labels_only = data[labels]
    print(len(labels), "labels")

    num_labels = labels_only.sum(axis=1)
    print("Number of labels per example:")
    print(num_labels.describe())
    print()

    label_counts = labels_only.apply(pd.Series.value_counts).transpose()
    label_counts["percent"] = label_counts[1] / num_entries
    print(label_counts["percent"])


def _create_data_files(
    train_exs: pd.DataFrame, dev_exs: pd.DataFrame, test_exs: pd.DataFrame
):
    """
    Create csv files in the data/ folder for the train/dev/test data

    Args:
        train_exs (pd.DataFrame): train dataset
        dev_exs (pd.DataFrame): dev dataset
        test_exs (pd.DataFrame): test dataset
    """
    train_exs.to_csv("data/train-data.csv")
    train_exs.sample(frac=0.025, random_state=random_seed).to_csv(
        "data/train-data-small.csv"
    )
    dev_exs.to_csv("data/dev-data.csv")
    dev_exs.sample(frac=0.025, random_state=random_seed).to_csv(
        "data/dev-data-small.csv"
    )
    test_exs.to_csv("data/test-data.csv")
    test_exs.sample(frac=0.025, random_state=random_seed).to_csv(
        "data/test-data-small.csv"
    )


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    if args.summary not in ["ALL", "TRAIN", "DEV", "TEST"]:
        raise Exception("Invalid summary type")

    full_data = pd.read_csv(args.data_path)

    train_exs, dev_exs, test_exs = _split_dataset(full_data)

    match args.summary:
        case "ALL":
            _summarize_dataset(full_data)
        case "TRAIN":
            print("Training Data")
            _summarize_dataset(train_exs)
        case "DEV":
            print("Dev Data")
            _summarize_dataset(dev_exs)
        case "TEST":
            print("Test Data")
            _summarize_dataset(test_exs)

    if args.create_files:
        _create_data_files(train_exs, dev_exs, test_exs)
