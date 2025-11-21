import argparse
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import processed_labels, label_names

sns.set_style("whitegrid")
sns.set_theme(font_scale=1.5, rc={"text.usetex": True})
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


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
    args = parser.parse_args()
    return args


def _plot_data_summary(data: pd.DataFrame):
    label_counts = data[processed_labels].sum().sort_values(ascending=False)

    data["label_count"] = data[processed_labels].sum(axis=1)
    data["abstract_len"] = data.apply(
        lambda row: len(word_tokenize(row["abstractText"])), axis=1
    )

    plt.figure(figsize=(10, 6))
    plot_data = label_counts.rename(index=label_names)
    sns.barplot(x=plot_data.values, y=plot_data.index, palette="viridis")
    # plt.title("Frequency of MeSH Root Labels")
    plt.xlabel("Number of Samples")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("plots/label_freq.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["label_count"], discrete=True, color="blue", binwidth=1)
    # plt.title("Number of Labels per Abstract")
    plt.xlabel("Number of Labels")
    plt.xticks(range(0, len(processed_labels) + 1))
    plt.tight_layout()
    plt.savefig("plots/labels_per_sample.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["abstract_len"], bins=50, color="blue")
    # plt.title("Distribution of Abstract Lengths")
    plt.xlabel("Abstract Length")
    plt.tight_layout()
    plt.savefig("plots/text_lengths.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    full_data = pd.read_csv(args.data_path)
    _plot_data_summary(full_data)
