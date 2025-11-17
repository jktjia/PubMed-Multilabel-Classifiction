from typing import List
from nltk.lm import Vocabulary
import nltk

nltk.download("punkt")


def create_vocab(exs: List[List[str]]) -> Vocabulary:
    vocab = Vocabulary(unk_cutoff=1)

    words = [i for s in exs for i in s]

    vocab.update(words)

    return vocab
