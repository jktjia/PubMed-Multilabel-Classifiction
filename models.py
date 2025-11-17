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
