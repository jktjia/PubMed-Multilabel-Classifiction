import argparse
import time

from evaluate_model import evaluate
from multilabel_example import read_examples
from models import train_LR, TrivialMultilabelClassifier
from prep_multilabel_data import processed_labels


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="trainer.py")
    parser.add_argument(
        "--model", type=str, default="TRIVIAL", help="model to run (TRIVIAL or LR)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of epochs to train for"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    # Load train, dev, and test exs and index the words.
    train_exs = read_examples("data/train-data-small.csv")
    dev_exs = read_examples("data/dev-data-small.csv")
    print(repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " train/dev examples")

    # Train and evaluate
    start_time = time.time()
    if args.model == "LR":
        model = train_LR(args, train_exs, dev_exs, num_labels=len(processed_labels))
    else:
        model = TrivialMultilabelClassifier(num_labels=len(processed_labels))

    print("=====Train Accuracy=====")
    evaluate(model, train_exs)

    print("=====Dev Accuracy=====")
    evaluate(model, dev_exs)

    train_eval_time = time.time() - start_time
    print("Time for training and evaluation: %.2f seconds" % train_eval_time)
