from collections import Counter
import random
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.lm import Vocabulary
import nltk

from multilabel_example import MultilabelExample
from utils import create_vocab


nltk.download("punkt")


class MultilabelClassifier(object):
    """
    Sentiment classifier base type
    """

    def __init__(self, num_labels):
        self.num_labels = num_labels

    def predict(self, ex_words: List[str]) -> List[int]:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[List[int]]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialMultilabelClassifier(MultilabelClassifier):
    def predict(self, ex_words: List[str]) -> List[int]:
        """
        :param ex:
        :return: 0, always predicts negative labels
        """
        return [0] * self.num_labels
