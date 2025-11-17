from typing import List
from nltk.lm import Vocabulary
import nltk

from multilabel_example import MultilabelExample

nltk.download("punkt")


def create_vocab(exs: List[MultilabelExample]) -> Vocabulary:
    """
    creates a vocabulary from a list of examples

    Args:
        exs (List[MultilabelExample]): list of examples

    Returns:
        Vocabulary: nltk vocabulary
    """
    vocab = Vocabulary(unk_cutoff=1)

    words = [i for ex in exs for i in ex.words]

    vocab.update(words)

    return vocab
