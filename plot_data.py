import argparse
import json
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import processed_labels, label_names

sns.set_style("whitegrid")
sns.set_theme(font_scale=1.25, rc={"text.usetex": True})
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

DEST_DIR = "plots"


def _parse_args():
    """
    Command-line arguments to the system. Allows for optional specification of location of file containing full data set.

    Returns: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="plot_data.py")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/pubmed-multi-label-text-classification-dataset-processed.csv",
        help="path to full data set",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="ALL",
        help="which plots to generate",
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
    sns.barplot(
        x=plot_data.values, y=plot_data.index, hue=plot_data.index, palette="viridis"
    )
    plt.xlabel("Number of Samples")
    plt.ylabel("")
    plt.tight_layout()
    label_freq_out = f"{DEST_DIR}/label_freq.png"
    plt.savefig(label_freq_out, dpi=300)
    plt.close()
    print("Label freq graph saved to %s" % label_freq_out)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x="label_count", discrete=True, color="blue", binwidth=1)
    plt.xlabel("Number of Labels")
    plt.xticks(range(0, len(processed_labels) + 1))
    plt.tight_layout()
    label_sample_out = f"{DEST_DIR}/labels_per_sample.png"
    plt.savefig(label_sample_out, dpi=300)
    plt.close()
    print("Label count graph saved to %s" % label_sample_out)

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x="abstract_len", bins=50, color="blue")
    # plt.title("Distribution of Abstract Lengths")
    plt.xlabel("Abstract Length")
    plt.tight_layout()
    text_len_out = f"{DEST_DIR}/text_lengths.png"
    plt.savefig(text_len_out, dpi=300)
    plt.close()
    print("Abstract length graph saved to %s" % text_len_out)


def _read_performance(model_paths: dict[str, str]):
    perf_data = pd.DataFrame()
    for model, path in model_paths.items():
        with open(path, "r") as file:
            data = json.load(file)
            if model == "Trivial":
                data = data * 25
            df = pd.DataFrame(data)
            df["Model"] = model
            df["epoch"] = df.index
            df = df.reindex()
            perf_data = pd.concat([perf_data, df])
    return perf_data


def _plot_performance(data: pd.DataFrame):
    metrics = [
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "exact_match_ratio",
        "hamming_loss",
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.catplot(
            x="epoch",
            y=metric,
            hue="Model",
            data=data,
            kind="point",
            palette="viridis",
            aspect=1.5,
        )
        plt.xlabel("Epoch")
        plt.ylim(0, 1)
        plt.ylabel("")
        plt.xticks([4 * x for x in range(0, 7)])
        out = "%s/%s.png" % (DEST_DIR, metric)
        plt.savefig(out, dpi=300)
        plt.close()
        print("%s graph saved to %s" % (metric.replace("_", " ").capitalize(), out))


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    if args.plots == "ALL" or args.plots == "DATA":
        full_data = pd.read_csv(args.data_path)
        print("\nRead in %i entries of data" % len(full_data))
        _plot_data_summary(full_data)

    if args.plots == "ALL" or args.plots == "PERF":
        model_paths = {
            # "Trivial": "outputs/trivial_output.json",
            "LR": "outputs/lr_output.json",
            "CNN": "outputs/cnn_output.json",
            "RNN": "outputs/rnn_output.json",
        }
        perf = _read_performance(model_paths=model_paths)
        print("\nRead in performance data for %i models" % len(model_paths))
        _plot_performance(perf)
