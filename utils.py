from typing import List
import pandas as pd
from nltk.lm import Vocabulary
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")


class MultilabelExample:
    """
    Data wrapper for a single example for multilabel classification.

    Attributes:
        words (List[str]): words in the example
        labels (List[int]): list of 0 or 1 for each label (0 = negative, 1 = positive)
    """

    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __repr__(self):
        return repr(self.words) + "; labels=" + repr(self.labels)

    def __str__(self):
        return self.__repr__()


def read_examples(path):
    """
    Reads in a csv of data and parses it into MultilabelExamples

    Args:
        path (str): path to csv of examples

    Returns:
        List[MultilabelExample]: list of examples parsed as MultilabelExamples
        Vocabulary: vocabulary of abstracts in the examples
    """
    df = pd.read_csv(path)
    df = df[processed_labels + ["abstractText"]]
    dicts = df.to_dict("records")
    exs = []
    for d in dicts:
        labels = [0] * len(processed_labels)
        for i, y in enumerate(processed_labels):
            labels[i] = d[y]
        words = word_tokenize(text=d["abstractText"], language="english")
        exs += [MultilabelExample(words=words, labels=labels)]
    return exs


def create_vocab(
    exs: List[MultilabelExample], include_stopwords: bool | None = None
) -> Vocabulary:
    """
    creates a vocabulary from a list of examples

    Args:
        exs (List[MultilabelExample]): list of examples
        include_stopwords (bool | None): should stopwords be included in the vocabulary

    Returns:
        Vocabulary: nltk vocabulary
    """
    vocab = Vocabulary(unk_cutoff=2)

    words = [i for ex in exs for i in ex.words]
    if not include_stopwords:
        words = [w for w in words if w not in nltk.corpus.stopwords.words("english")]

    vocab.update(words)

    return vocab


processed_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "L",
    "M",
    "N",
    "Z",
]

default_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "V",
    "Z",
]
