import pandas as pd
from nltk.lm import Vocabulary
from nltk.tokenize import word_tokenize
import nltk
import torch
from torch.utils.data import TensorDataset

nltk.download("punkt")


class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """

    def __init__(self):
        self.objs_to_ints = {"<UNK>": 0}
        self.ints_to_objs = {0: "<UNK>"}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != 0

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if object not in self.objs_to_ints:
            return 0
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def create_vocab(path, ignore_stopwords: bool | None = None):
    """
    Reads in a csv of data and creates a vocabulary from the examples

    Args:
        path (str): path to csv of examples
        ignore_stopwords (bool | None): whether or not to exclude stopwords from the vocabulary

    Returns:
        Indexer: indexer of vocabulary of abstracts in the examples
    """
    df = pd.read_csv(path)
    abstracts = [a for a in df["abstractText"]]
    vocabulary = Vocabulary()
    for d in abstracts:
        words = word_tokenize(text=d, language="english")
        vocab_words = [w for w in words]
        if ignore_stopwords:
            vocab_words = [
                w
                for w in vocab_words
                if w not in nltk.corpus.stopwords.words("english")
            ]
        vocabulary.update(vocab_words)
    vocab = Indexer()
    vocab.add_and_get_index("<PAD>")
    for word in vocabulary:
        vocab.add_and_get_index(word)
    return vocab


def read_examples(path, vocab: Indexer):
    """
    Reads in a csv of data and parses it into a tensor dataset

    Args:
        path (str): path to csv of examples
        vocab (Indexer): indexer of available vocabulary

    Returns:
        TensorDataset: dataset of the root labels and tokenized abstract text
    """
    df = pd.read_csv(path)
    df = df[processed_labels + ["abstractText"]]
    dicts = df.to_dict("records")
    xs = []
    ys = []
    for d in dicts:
        labels = [0] * len(processed_labels)
        for i, y in enumerate(processed_labels):
            labels[i] = d[y]

        words = word_tokenize(text=d["abstractText"], language="english")
        # indices = [vocab.index_of(word) for word in words]
        xs += [words]
        ys += [labels]

    # max_len = max(len(x) for x in xs)
    # xs = [x + [vocab.index_of("<PAD>")] * (max_len - len(x)) for x in xs]

    # return TensorDataset(
    #     torch.tensor(xs, dtype=torch.int),
    #     torch.tensor(ys, dtype=torch.float),
    # )
    return xs, ys


def create_dataset(tokens, labels, vocab: Indexer):
    xs = [[vocab.index_of(word) for word in words] for words in tokens]

    max_len = max(len(x) for x in xs)
    xs = [x + [vocab.index_of("<PAD>")] * (max_len - len(x)) for x in xs]

    return TensorDataset(
        torch.tensor(xs, dtype=torch.int),
        torch.tensor(labels, dtype=torch.float),
    )


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

label_names = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals",
    "E": "Techniques",
    "F": "Psychiatry",
    "G": "Phenomena",
    "H": "Disciplines",
    "I": "Anthropology",
    "J": "Technology",
    "L": "Information",
    "M": "Named Groups",
    "N": "Health Care",
    "Z": "Geographical",
}
