from collections import Counter
import random
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.lm import Vocabulary
import nltk

from utils import create_vocab, MultilabelExample


nltk.download("punkt")


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
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialMultilabelClassifier(MultilabelClassifier):
    def predict(self, ex_words: List[str]) -> List[int]:
        return [0] * self.num_labels


class LRMultilabelModule(nn.Module):
    def __init__(self, num_labels: int, vocab_size: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(vocab_size, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, bow_vector):
        return self.seq(bow_vector)


class LRMultilabelClassifier(MultilabelClassifier):
    def __init__(self, num_labels: int, vocab: Vocabulary):
        super().__init__(num_labels)
        self.vocab = vocab
        self.module = LRMultilabelModule(
            num_labels=num_labels,
            vocab_size=len(vocab),
        )

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: list of 0 or 1 for the labels
        """
        word_count = Counter(ex_words)
        bow_vector = [word_count[word] for word in self.vocab]
        x = torch.tensor(bow_vector, dtype=torch.float32)
        probs = self.module.forward(x)
        prediction = (probs > 0.5).int()
        return prediction

    # def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
    #     length = max([len(ex) for ex in all_ex_words])
    #     x = torch.stack([self.words_to_tensor(ex, length) for ex in all_ex_words])
    #     return self.nn_module.forward(x).argmax(dim=1)


def train_LR(
    args,
    train_exs: List[MultilabelExample],
    dev_exs: List[MultilabelExample],
    num_labels: int,
    # vocab: Vocabulary,
) -> LRMultilabelClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate

    vocab = create_vocab(train_exs)
    print("%i words in vocabulary" % len(vocab))

    model = LRMultilabelClassifier(
        num_labels,
        vocab,
    )
    optimizer = optim.Adam(model.module.parameters(), lr=initial_learning_rate)
    loss_fn = nn.BCELoss()
    ex_idxs = [i for i in range(0, len(train_exs))]
    for epoch in range(0, num_epochs):
        random.shuffle(ex_idxs)
        total_loss = 0.0
        for ex_idx in ex_idxs:
            ex = train_exs[ex_idx]
            word_count = Counter(ex.words)
            bow_vector = [word_count[word] for word in vocab]
            x = torch.tensor(bow_vector, dtype=torch.float32)
            y = torch.tensor(ex.labels).float()

            probs = model.module.forward(x)

            model.module.zero_grad()

            loss = loss_fn(probs, y)
            total_loss += loss

            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return model
