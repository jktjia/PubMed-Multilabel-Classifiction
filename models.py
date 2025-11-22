import json
import math
import random
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from evaluate_model import evaluate
from utils import Indexer

default_embed_size = 256
default_hidden_size = 128


class MultilabelClassifier(object):
    """
    Multilabel classifier base type
    """

    def __init__(self, num_labels):
        self.num_labels = num_labels

    def predict(self, ex_words: List[str]) -> List[int]:
        """
        Makes a prediction on the given sentence

        Args:
            ex_words (List[str]): words to predict on

        Returns:
            List[int]: 0 or 1 for each label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[List[int]]:
        """
        Makes predictions for each sentence in a given list of sentences

        Args:
            all_ex_words (List[List[str]]): list of sentences to predict

        Returns:
            List[List[int]]: 0 or 1 for each label for each sentence
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialMultilabelClassifier(MultilabelClassifier):
    """
    Trivial multilabel classifier that always returns 0 for all labels
    """

    def predict(self, ex_words: List[str]) -> List[int]:
        return [0] * self.num_labels


class LR(nn.Module):
    def __init__(
        self,
        num_labels: int,
        vocab: Indexer,
        embedding_layer: nn.Embedding | None = None,
    ):
        super(LR, self).__init__()
        self.embed = (
            embedding_layer
            if embedding_layer
            else nn.Embedding(
                len(vocab), default_embed_size, padding_idx=vocab.index_of("<PAD>")
            )
        )
        self.l = nn.Linear(
            embedding_layer.embedding_dim if embedding_layer else default_embed_size,
            num_labels,
        )

    def forward(self, indices):
        embed_avg = torch.mean(self.embed(indices), dim=-2)
        return self.l(embed_avg)


class CNN(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        num_labels: int,
        hidden_d: int,
        vocab: Indexer,
        embedding_layer: nn.Embedding | None = None,
    ):
        super(CNN, self).__init__()
        self.embedding_layer = (
            embedding_layer
            if embedding_layer
            else nn.Embedding(
                len(vocab), default_embed_size, padding_idx=vocab.index_of("<PAD>")
            )
        )
        self.conv = nn.Conv1d(
            embedding_layer.embedding_dim if embedding_layer else default_embed_size,
            hidden_d,
            kernel_size,
            stride,
        )
        self.l = nn.Linear(hidden_d, num_labels)

    def forward(self, x):
        embed = self.embedding_layer(x)
        if embed.dim() == 3:
            embed = torch.permute(embed, dims=(0, -1, -2))
        elif embed.dim() == 2:
            embed = torch.permute(embed, dims=(-1, -2))
        conv = self.conv(embed)
        max = torch.max(conv, -1).values
        return self.l(max)


class NNMultilabelClassifier(MultilabelClassifier):
    """
    Logistic regression multilabel classifier
    """

    def __init__(self, num_labels: int, vocab: Indexer, module: nn.Module):
        super().__init__(num_labels)
        self.vocab = vocab
        self.module = module
        self.sigmoid = nn.Sigmoid()

    def predict(self, ex_words: List[str]) -> int:
        x = torch.tensor([self.vocab.index_of(word) for word in ex_words]).int()
        probs = self.sigmoid(self.module.forward(x))
        prediction = (probs > 0.5).int()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        indices = [
            torch.tensor([self.vocab.index_of(word) for word in ex])
            for ex in all_ex_words
        ]
        max_len = max(len(t) for t in indices)
        x = torch.stack(
            [
                torch.nn.functional.pad(
                    t,
                    (0, max_len - len(t)),
                    mode="constant",
                    value=self.vocab.index_of("<PAD>"),
                )
                for t in indices
            ]
        ).int()
        probs = self.sigmoid(self.module.forward(x))
        prediction = (probs > 0.5).int()
        return prediction


def train_NNClassifier(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
    vocab: Indexer,
    model: nn.Module,
    loss_plot: str | None = None,
    epoch_metrics: str | None = None,
    min_length: int | None = None,
) -> NNMultilabelClassifier:
    """
    Trains a multilabel classifier based on a given model on the given training examples

    Args:
        args (_type_): command-line args
        train_exs (_type_): train examples
        dev_exs (_type_): dev examples
        num_labels (int): number of labels
        vocab (Indexer): an indexer of the vocabulary in the examples
        model (nn.Module): internal model.
        plot_loss (bool | None, optional): whether the loss per epoch should be plotted. Defaults to None.
        output_epoch_metrics (bool | None, optional): whether the performance metrics per epoch should be outputted. Defaults to None.

    Returns:
        NNMultilabelClassifier: trained multilabel classifier
    """
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate
    batch_size = args.batch_size

    classifier = NNMultilabelClassifier(num_labels, vocab, module=model)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_exs_indices = [
        (torch.tensor([vocab.index_of(word) for word in ex[0]]), ex[1])
        for ex in train_exs
    ]
    dev_exs_indices = [
        (
            torch.tensor([vocab.index_of(word) for word in ex[0]]).int(),
            torch.tensor(ex[1]).float(),
        )
        for ex in dev_exs
    ]
    ex_idxs = [i for i in range(0, len(train_exs))]
    train_loss = []
    dev_loss = []
    dev_metrics = []
    for epoch in range(0, num_epochs):
        model.train()
        random.shuffle(ex_idxs)
        total_loss = 0.0
        for batch in range(math.ceil(len(ex_idxs) / batch_size)):
            exs = [
                train_exs_indices[idx]
                for idx in ex_idxs[batch_size * batch : batch_size * (batch + 1)]
            ]
            max_len = max(len(t[0]) for t in exs)
            if min_length:
                max_len = max(max_len, min_length)
            x = torch.stack(
                [
                    torch.nn.functional.pad(
                        ex[0],
                        (0, max_len - len(ex[0])),
                        mode="constant",
                        value=vocab.index_of("<PAD>"),
                    )
                    for ex in exs
                ]
            ).int()
            y = torch.tensor([ex[1] for ex in exs]).float()

            probs = model.forward(x)

            model.zero_grad()

            loss = loss_fn(probs, y)

            total_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()
            # print("Epoch %i, batch %i loss: %f" % (epoch, batch, loss))
        print("Epoch %i loss: %f" % (epoch, total_loss / len(train_exs)))
        train_loss += [total_loss / len(train_exs)]

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dev_exs_indices:
                if min_length and len(x) < min_length:
                    x = torch.nn.functional.pad(
                        x,
                        (0, min_length - len(x)),
                        mode="constant",
                        value=vocab.index_of("<PAD>"),
                    )
                output = model.forward(x)
                loss = loss_fn(output, y)
                total_loss += loss.item()
            dev_loss += [total_loss / len(dev_exs)]
        dev_metrics += [evaluate(classifier=classifier, exs=dev_exs)]

    if loss_plot:
        plt.plot(train_loss, label="Train Loss")
        plt.plot(dev_loss, label="Dev Loss")
        plt.title("Loss by Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_plot)

    if epoch_metrics:
        with open(epoch_metrics, "w") as outfile:
            json.dump(dev_metrics, outfile)

    return classifier


def train_LR(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
    vocab: Indexer,
    embedding_layer: nn.Embedding | None = None,
    plot_loss: bool | None = None,
    output_epoch_metrics: bool | None = None,
) -> NNMultilabelClassifier:
    """
    Trains a logistic regression multilabel classifier on the given training examples

    Args:
        args (_type_): command-line args
        train_exs (_type_): train examples
        dev_exs (_type_): dev examples
        num_labels (int): number of labels
        vocab (Indexer): an indexer of the vocabulary in the examples
        embedding_layer (nn.Embedding | None, optional): optional pretrained embedding layer. Defaults to None.
        plot_loss (bool | None, optional): whether the loss per epoch should be plotted. Defaults to None.
        output_epoch_metrics (bool | None, optional): whether the performance metrics per epoch should be outputted. Defaults to None.

    Returns:
        NNMultilabelClassifier: trained logistic regression multilabel classifier
    """

    model = LR(
        num_labels,
        vocab,
        embedding_layer,
    )

    return train_NNClassifier(
        args,
        train_exs,
        dev_exs,
        num_labels,
        vocab,
        model,
        "plots/lr_loss.png" if plot_loss else None,
        "outputs/lr_output.json" if output_epoch_metrics else None,
    )


def train_CNN(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
    vocab: Indexer,
    embedding_layer: nn.Embedding | None = None,
    plot_loss: bool | None = None,
    output_epoch_metrics: bool | None = None,
) -> NNMultilabelClassifier:
    """
    Trains a convolutional neural network regression multilabel classifier on the given training examples

    Args:
        args (_type_): command-line args
        train_exs (_type_): train examples
        dev_exs (_type_): dev examples
        num_labels (int): number of labels
        vocab (Indexer): an indexer of the vocabulary in the examples
        embedding_layer (nn.Embedding | None, optional): optional pretrained embedding layer. Defaults to None.
        plot_loss (bool | None, optional): whether the loss per epoch should be plotted. Defaults to None.
        output_epoch_metrics (bool | None, optional): whether the performance metrics per epoch should be outputted. Defaults to None.

    Returns:
        NNMultilabelClassifier: trained CNN multilabel classifier
    """

    kernel_size = 50
    stride = 20

    model = CNN(
        num_labels=num_labels,
        embedding_layer=embedding_layer,
        kernel_size=kernel_size,
        hidden_d=default_hidden_size,
        stride=stride,
        vocab=vocab,
    )

    return train_NNClassifier(
        args,
        train_exs,
        dev_exs,
        num_labels,
        vocab,
        model,
        "plots/cnn_loss.png" if plot_loss else None,
        "outputs/cnn_output.json" if output_epoch_metrics else None,
        min_length=kernel_size,
    )
