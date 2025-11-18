import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.lm import Vocabulary
import nltk

from prep_multilabel_data import processed_labels

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
    # vocab = Vocabulary(unk_cutoff=1)
    for d in dicts:
        labels = [0] * len(processed_labels)
        for i, y in enumerate(processed_labels):
            labels[i] = d[y]
        words = word_tokenize(text=d["abstractText"], language="english")
        # vocab.update(words)
        exs += [MultilabelExample(words=words, labels=labels)]
    return exs
