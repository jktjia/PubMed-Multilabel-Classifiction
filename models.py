import json
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from evaluate_model import evaluate

from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import BCEWithLogitsLoss
from tqdm import trange
from nltk.tokenize import word_tokenize
from utils import Indexer, create_dataset

default_embed_size = 256
default_hidden_size = 128

PLOT_DIR = "plots"
OUTPUT_DIR = "outputs"


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


class RNN(nn.Module):
    def __init__(
        self,
        num_labels: int,
        hidden_d: int,
        vocab: Indexer,
        embedding_layer: nn.Embedding | None = None,
    ):
        super().__init__()
        self.emb = (
            embedding_layer
            if embedding_layer
            else nn.Embedding(
                len(vocab), default_embed_size, padding_idx=vocab.index_of("<PAD>")
            )
        )
        self.lstm = nn.LSTM(
            embedding_layer.embedding_dim if embedding_layer else default_embed_size,
            hidden_d,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_d * 2, num_labels)

    def forward(self, x):
        embedded = self.emb(x)
        _, (h, _) = self.lstm(embedded)
        hidden = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(hidden)


class BaselineMultilabelClassifier(MultilabelClassifier):
    """
    Baseline multilabel classifier
    """

    def __init__(self, num_labels: int, vocab: Indexer, module: nn.Module):
        super().__init__(num_labels)
        self.vocab = vocab
        self.module = module
        self.sigmoid = nn.Sigmoid()

    def predict(self, ex: str) -> int:
        ex_words = word_tokenize(ex)
        x = torch.tensor([self.vocab.index_of(word) for word in ex_words]).int()
        probs = self.sigmoid(self.module.forward(x))
        prediction = (probs > 0.5).int()
        return prediction

    def predict_all(self, all_ex: List[str]) -> List[int]:
        all_ex_words = [word_tokenize(ex) for ex in all_ex]
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


def train_BaselineClassifier(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
    vocab: Indexer,
    model: nn.Module,
    loss_plot: str | None = None,
    epoch_metrics: str | None = None,
    min_length: int | None = None,
) -> BaselineMultilabelClassifier:
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
        BaselineMultilabelClassifier: trained multilabel classifier
    """
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate
    batch_size = args.batch_size

    classifier = BaselineMultilabelClassifier(num_labels, vocab, module=model)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_words = [word_tokenize(ex) for ex in train_exs[0]]
    train_ds = create_dataset(train_words, train_exs[1], vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_words = [word_tokenize(ex) for ex in dev_exs[0]]
    dev_ds = create_dataset(dev_words, dev_exs[1], vocab)
    dev_loader = DataLoader(dev_ds)

    train_loss = []
    dev_loss = []
    dev_metrics = []
    for epoch in trange(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            probs = model.forward(x)

            model.zero_grad()

            loss = loss_fn(probs, y)

            total_loss += loss.item() * batch_size

            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_exs[0])
        train_loss += [avg_train_loss]

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dev_loader:
                output = model.forward(x)
                loss = loss_fn(output, y)
                total_loss += loss.item()
            dev_loss += [total_loss / len(dev_exs[0])]
        metrics = evaluate(classifier=classifier, exs=dev_exs)
        dev_metrics += [metrics]

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Micro F1: {metrics['micro_f1']:.4f} | Exact Match: {metrics['exact_match_ratio']:.4f}"
        )

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
) -> BaselineMultilabelClassifier:
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

    return train_BaselineClassifier(
        args,
        train_exs,
        dev_exs,
        num_labels,
        vocab,
        model,
        "%s/%s/lr_loss.png" % (PLOT_DIR, args.dataset.lower()) if plot_loss else None,
        "%s/%s/lr_output.json" % (OUTPUT_DIR, args.dataset.lower()),
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
) -> BaselineMultilabelClassifier:
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

    kernel_size = 64
    stride = 16

    model = CNN(
        num_labels=num_labels,
        embedding_layer=embedding_layer,
        kernel_size=kernel_size,
        hidden_d=default_hidden_size,
        stride=stride,
        vocab=vocab,
    )

    return train_BaselineClassifier(
        args,
        train_exs,
        dev_exs,
        num_labels,
        vocab,
        model,
        "%s/%s/cnn_loss.png" % (PLOT_DIR, args.dataset.lower()) if plot_loss else None,
        "%s/%s/cnn_output.json" % (OUTPUT_DIR, args.dataset.lower()),
        min_length=kernel_size,
    )


def train_RNN(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
    vocab: Indexer,
    embedding_layer: nn.Embedding | None = None,
    plot_loss: bool | None = None,
    output_epoch_metrics: bool | None = None,
) -> BaselineMultilabelClassifier:
    """
    Trains a ==r== neural network regression multilabel classifier on the given training examples

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
        NNMultilabelClassifier: trained RNN multilabel classifier
    """

    model = RNN(
        num_labels=num_labels,
        embedding_layer=embedding_layer,
        hidden_d=default_hidden_size,
        vocab=vocab,
    )

    return train_BaselineClassifier(
        args,
        train_exs,
        dev_exs,
        num_labels,
        vocab,
        model,
        "%s/%s/rnn_loss.png" % (PLOT_DIR, args.dataset.lower()) if plot_loss else None,
        "%s/%s/rnn_output.json" % (OUTPUT_DIR, args.dataset.lower())
        if output_epoch_metrics
        else None,
    )


class BERTMultilabelClassifier(MultilabelClassifier):
    """
    BERT-based multilabel classifier using BioBERT
    """

    def __init__(self, model, tokenizer, num_labels: int, device=None, max_length=128):
        super().__init__(num_labels)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    # def _words_to_text(self, ex_words: List[str]) -> str:
    #     """Convert tokenized words back to text string for BERT"""
    #     # Join words, handling punctuation spacing
    #     text = " ".join(ex_words)
    #     # Fix common spacing issues
    #     import re

    #     text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    #     text = re.sub(r'(["\'])\s+', r"\1", text)
    #     return text

    def predict(self, ex: str) -> List[int]:
        """
        Makes a prediction from raw text string

        Args:
            ex (str): raw text to predict on

        Returns:
            List[int]: 0 or 1 for each label
        """
        encodings = self.tokenizer(
            ex,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int().cpu().numpy()[0].tolist()

        return predictions

    def predict_all(self, all_exs: List[str]) -> List[List[int]]:
        """
        Makes predictions from a list of raw text strings

        Args:
            all_exs (List[str]): list of raw texts to predict on

        Returns:
            List[List[int]]: 0 or 1 for each label for each text
        """
        predictions = []
        batch_size = 16  # Process in batches

        for i in range(0, len(all_exs), batch_size):
            batch_texts = all_exs[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probs = torch.sigmoid(logits)
                batch_predictions = (probs > 0.5).int().cpu().numpy().tolist()
                predictions.extend(batch_predictions)

        return predictions


def train_BERT(
    args,
    train_exs,
    dev_exs,
    num_labels: int,
) -> BERTMultilabelClassifier:
    """
    Trains a BERT multilabel classifier on the given training examples

    Args:
        args: command line args
        train_exs: training examples
        dev_exs: development set
        num_labels (int): number of labels
        train_csv_path (str, optional): path to training CSV to read texts directly
        dev_csv_path (str, optional): path to dev CSV to read texts directly

    Returns:
        BERTMultilabelClassifier: A trained BERTMultilabelClassifier model
    """
    # BERT parameters
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    max_length = 128
    batch_size = args.batch_size if args.batch_size > 1 else 16
    num_epochs = args.num_epochs if args.num_epochs > 0 else 25
    learning_rate = args.learning_rate if args.learning_rate > 0 else 2e-5

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading BioBERT model: {model_name}")
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, use_safetensors=True
    )
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    model.to(device)

    # Extract texts and labels from examples
    # Try to read directly from CSV for better accuracy, otherwise reconstruct from words
    train_labels = train_exs[1]
    train_texts = train_exs[0]
    dev_labels = dev_exs[1]
    dev_texts = dev_exs[0]

    # Validate lengths match
    if len(train_texts) != len(train_labels):
        raise ValueError(
            f"Train texts ({len(train_texts)}) and labels ({len(train_labels)}) length mismatch!"
        )
    if len(dev_texts) != len(dev_labels):
        raise ValueError(
            f"Dev texts ({len(dev_texts)}) and labels ({len(dev_labels)}) length mismatch!"
        )

    # Tokenize
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    print("Tokenizing dev data...")
    dev_encodings = tokenizer(
        dev_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create tensors
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
    dev_labels_tensor = torch.tensor(dev_labels, dtype=torch.float)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        train_labels_tensor,
    )
    dev_dataset = TensorDataset(
        dev_encodings["input_ids"], dev_encodings["attention_mask"], dev_labels_tensor
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    dev_dataloader = DataLoader(
        dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size
    )

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = BCEWithLogitsLoss()

    classifier = BERTMultilabelClassifier(
        model, tokenizer, num_labels, device, max_length
    )

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    dev_metrics = []
    for epoch in trange(num_epochs, desc="Epoch"):
        # Training
        model.train()
        running_loss = 0.0

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]

            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)

        # Validation
        model.eval()
        all_preds = []
        all_true_labels = []

        with torch.no_grad():
            for batch in dev_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]
                preds = torch.sigmoid(logits)

                all_preds.append(preds.cpu())
                all_true_labels.append(b_labels.cpu())
        dev_metrics += [evaluate(classifier=classifier, exs=dev_exs)]

        # Calculate metrics
        all_preds = torch.cat(all_preds).numpy()
        all_true_labels = torch.cat(all_true_labels).numpy()

        threshold = 0.5
        pred_bools = all_preds > threshold
        true_bools = all_true_labels == 1

        # Simple accuracy calculation
        correct = (pred_bools == true_bools).sum()
        total = pred_bools.size
        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Dev Accuracy: {accuracy:.4f}"
        )

    with open(
        "%s/%s/bert_output.json" % (OUTPUT_DIR, args.dataset.lower()), "w"
    ) as outfile:
        json.dump(dev_metrics, outfile)

    print("\nTraining completed!")

    # Return classifier wrapper
    return classifier
