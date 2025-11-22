import argparse
import time

from evaluate_model import print_eval
from utils import processed_labels, read_examples
from models import TrivialMultilabelClassifier, train_LR, train_CNN


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
        help="model to run (TRIVIAL, LR, CNN, or RNN)",
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
        train_exs, vocab = read_examples("data/train-data-small.csv")
        dev_exs, _ = read_examples("data/dev-data-small.csv")
        test_exs, _ = read_examples("data/test-data-small.csv")
    elif args.dataset == "MED":
        train_exs, vocab = read_examples("data/train-data-med.csv")
        dev_exs, _ = read_examples("data/dev-data-med.csv")
        test_exs, _ = read_examples("data/test-data-med.csv")
    else:
        train_exs, vocab = read_examples("data/train-data.csv")
        dev_exs, _ = read_examples("data/dev-data.csv")
        test_exs, _ = read_examples("data/test-data.csv")
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " train/dev examples")
    print("%i items in vocabulary" % len(vocab))

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
    else:
        model = TrivialMultilabelClassifier(num_labels=len(processed_labels))

    print("\n=====Train Accuracy (200 examples)=====\n")
    print_eval(model, train_exs[:200])

    print("\n=====Dev Accuracy (whole dataset)=====\n")
    print_eval(model, dev_exs)

    train_eval_time = time.time() - start_time
    print("\nTime for training and evaluation: %.2f seconds" % train_eval_time)
