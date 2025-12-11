import argparse
import time

from evaluate_model import print_eval
from utils import processed_labels, create_vocab, read_examples
from models import (
    TrivialMultilabelClassifier,
    train_LR,
    train_CNN,
    train_RNN,
    train_BERT,
)


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.

    Returns:
        the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="multilabel_classifier.py")
    parser.add_argument(
        "--model",
        type=str,
        default="TRIVIAL",
        help="model to run (TRIVIAL, LR, CNN, RNN, 0or BERT)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of epochs to train for"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dataset",
        type=str,
        default="FULL",
        help="dataset size for training (FULL, MED, or SMALL)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    # Train and evaluate
    start_time = time.time()

    if args.dataset == "SMALL":
        vocab = create_vocab("data/train-data-small.csv")
        train_exs = read_examples("data/train-data-small.csv")
        dev_exs = read_examples("data/dev-data-small.csv")
        test_exs = read_examples("data/test-data-small.csv")
    elif args.dataset == "MED":
        vocab = create_vocab("data/train-data-med.csv")
        train_exs = read_examples("data/train-data-med.csv")
        dev_exs = read_examples("data/dev-data-med.csv")
        test_exs = read_examples("data/test-data-med.csv")
    else:
        vocab = create_vocab("data/train-data.csv")
        train_exs = read_examples("data/train-data.csv")
        dev_exs = read_examples("data/dev-data.csv")
        test_exs = read_examples("data/test-data.csv")
    print(
        repr(len(train_exs[0]))
        + " / "
        + repr(len(dev_exs[0]))
        + " / "
        + repr(len(test_exs[0]))
        + " train/dev/test examples"
    )
    # print("%i items in vocabulary" % len(vocab))

    embedding_layer = None

    if args.model == "LR":
        model = train_LR(
            args,
            train_exs,
            dev_exs,
            num_labels=len(processed_labels),
            vocab=vocab,
            embedding_layer=embedding_layer,
            plot_loss=True,
            output_epoch_metrics=True,
        )
    elif args.model == "CNN":
        model = train_CNN(
            args,
            train_exs,
            dev_exs,
            num_labels=len(processed_labels),
            vocab=vocab,
            embedding_layer=embedding_layer,
            plot_loss=True,
            output_epoch_metrics=True,
        )
    elif args.model == "RNN":
        model = train_RNN(
            args,
            train_exs,
            dev_exs,
            num_labels=len(processed_labels),
            vocab=vocab,
            embedding_layer=embedding_layer,
            plot_loss=True,
            output_epoch_metrics=True,
        )
    elif args.model == "BERT":  # best learning with 0.0001
        print("\n=====READ=====\n")
        print(
            "Training BERT model in model-class format (not in notebook format) with 50 train and 20 dev examples for demonstration purposes."
        )
        print(
            "For actual training+testing, we ran the BERT notebook on Colab with full dataset (>2 hrs train time).\n\n"
        )
        train_exs = [e[:50] for e in train_exs]
        dev_exs = [e[:50] for e in dev_exs]
        model = train_BERT(args, train_exs, dev_exs, num_labels=len(processed_labels))
    else:
        model = TrivialMultilabelClassifier(num_labels=len(processed_labels))

    print("\n=====Train Accuracy=====\n")
    print_eval(model, [vals[:200] for vals in train_exs])

    print("\n=====Dev Accuracy=====\n")
    print_eval(model, dev_exs)

    # print("\n=====Test Accuracy=====\n")
    # print_eval(model, test_exs)

    train_eval_time = time.time() - start_time
    print("\nTime for training and evaluation: %.2f seconds" % train_eval_time)
