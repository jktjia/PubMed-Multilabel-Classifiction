from collections import Counter
import math
import random
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.lm import Vocabulary
import nltk
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import BCEWithLogitsLoss
from tqdm import trange
from utils import create_vocab, MultilabelExample, processed_labels, read_abstract_texts


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


class LRMultilabelModule(nn.Module):
    def __init__(self, num_labels: int, in_size: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_size, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, bow_vector):
        return self.seq(bow_vector)


class LRMultilabelClassifier(MultilabelClassifier):
    """
    Logistic regression multilabel classifier
    """

    def __init__(self, num_labels: int, vocab: Vocabulary):
        super().__init__(num_labels)
        self.vocab = vocab
        self.module = LRMultilabelModule(
            num_labels=num_labels,
            in_size=len(vocab),
        )

    def predict(self, ex_words: List[str]) -> int:
        word_count = Counter(ex_words)
        bow_vector = [word_count[word] for word in self.vocab]
        x = torch.tensor(bow_vector, dtype=torch.float32)
        probs = self.module.forward(x)
        prediction = (probs > 0.5).int()
        return prediction

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        x = torch.stack(
            [
                torch.tensor([Counter(ex)[word] for word in self.vocab])
                for ex in all_ex_words
            ]
        ).float()
        probs = self.module.forward(x)
        prediction = (probs > 0.5).int()
        return prediction


def train_LR(
    args,
    train_exs: List[MultilabelExample],
    dev_exs: List[MultilabelExample],
    num_labels: int,
) -> LRMultilabelClassifier:
    """
    Trains a logistic regression multilabel classifier on the given training examples

    Args:
        args (_type_): command line args
        train_exs (List[MultilabelExample]): training examples
        dev_exs (List[MultilabelExample]): development set
        num_labels (int): number of labels

    Returns:
        LRMultilabelClassifier: A trained LRMultilabelClassifier model
    """
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate
    batch_size = args.batch_size

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
        for n in range(math.ceil(len(ex_idxs) / batch_size)):
            exs = [
                train_exs[idx] for idx in ex_idxs[batch_size * n : batch_size * (n + 1)]
            ]
            x = torch.stack(
                [
                    torch.tensor([Counter(ex.words)[word] for word in vocab])
                    for ex in exs
                ]
            ).float()
            y = torch.tensor([ex.labels for ex in exs]).float()

            probs = model.module.forward(x)

            model.module.zero_grad()

            loss = loss_fn(probs, y)
            total_loss += loss

            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return model

class BERTMultilabelClassifier(MultilabelClassifier):
    """
    BERT-based multilabel classifier using BioBERT
    """
    
    def __init__(self, model, tokenizer, num_labels: int, device=None, max_length=128):
        super().__init__(num_labels)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def _words_to_text(self, ex_words: List[str]) -> str:
        """Convert tokenized words back to text string for BERT"""
        # Join words, handling punctuation spacing
        text = " ".join(ex_words)
        # Fix common spacing issues
        import re
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'(["\'])\s+', r'\1', text)
        return text
    
    def predict(self, ex_words: List[str]) -> List[int]:
        """
        Makes a prediction on the given sentence
        
        Args:
            ex_words (List[str]): words to predict on
            
        Returns:
            List[int]: 0 or 1 for each label
        """
        text = self._words_to_text(ex_words)
        return self.predict_from_text(text)
    
    def predict_from_text(self, text: str) -> List[int]:
        """
        Makes a prediction from raw text string
        
        Args:
            text (str): raw text to predict on
            
        Returns:
            List[int]: 0 or 1 for each label
        """
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int().cpu().numpy()[0].tolist()
        
        return predictions
    
    def predict_all(self, all_ex_words: List[List[str]]) -> List[List[int]]:
        """
        Makes predictions for each sentence in a given list of sentences
        
        Args:
            all_ex_words (List[List[str]]): list of sentences to predict
            
        Returns:
            List[List[int]]: 0 or 1 for each label for each sentence
        """
        texts = [self._words_to_text(words) for words in all_ex_words]
        return self.predict_all_from_texts(texts)
    
    def predict_all_from_texts(self, texts: List[str]) -> List[List[int]]:
        """
        Makes predictions from a list of raw text strings
        
        Args:
            texts (List[str]): list of raw texts to predict on
            
        Returns:
            List[List[int]]: 0 or 1 for each label for each text
        """
        predictions = []
        batch_size = 16  # Process in batches
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probs = torch.sigmoid(logits)
                batch_predictions = (probs > 0.5).int().cpu().numpy().tolist()
                predictions.extend(batch_predictions)
        
        return predictions


def train_BERT(
    args,
    train_exs: List[MultilabelExample],
    dev_exs: List[MultilabelExample],
    num_labels: int,
    train_csv_path: str = None,
    dev_csv_path: str = None,
) -> BERTMultilabelClassifier:
    """
    Trains a BERT multilabel classifier on the given training examples
    
    Args:
        args: command line args
        train_exs (List[MultilabelExample]): training examples
        dev_exs (List[MultilabelExample]): development set
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading BioBERT model: {model_name}")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    
    model.to(device)
    
    # Extract texts and labels from examples
    # Try to read directly from CSV for better accuracy, otherwise reconstruct from words
    train_labels = [ex.labels for ex in train_exs]
    dev_labels = [ex.labels for ex in dev_exs]
    
    if train_csv_path:
        try:
            train_texts = read_abstract_texts(train_csv_path)
            # Slice to match number of examples if needed
            if len(train_texts) > len(train_exs):
                train_texts = train_texts[:len(train_exs)]
            print(f"Read {len(train_texts)} abstract texts from {train_csv_path}")
        except Exception as e:
            print(f"Warning: Could not read from {train_csv_path}: {e}. Reconstructing from words.")
            train_texts = [" ".join(ex.words) for ex in train_exs]
    else:
        train_texts = [" ".join(ex.words) for ex in train_exs]
    
    if dev_csv_path:
        try:
            dev_texts = read_abstract_texts(dev_csv_path)
            # Slice to match number of examples if needed
            if len(dev_texts) > len(dev_exs):
                dev_texts = dev_texts[:len(dev_exs)]
            print(f"Read {len(dev_texts)} abstract texts from {dev_csv_path}")
        except Exception as e:
            print(f"Warning: Could not read from {dev_csv_path}: {e}. Reconstructing from words.")
            dev_texts = [" ".join(ex.words) for ex in dev_exs]
    else:
        dev_texts = [" ".join(ex.words) for ex in dev_exs]
    
    # Ensure all texts are strings and not empty
    train_texts = [str(text) if text is not None else "" for text in train_texts]
    dev_texts = [str(text) if text is not None else "" for text in dev_texts]
    
    # Validate lengths match
    if len(train_texts) != len(train_labels):
        raise ValueError(f"Train texts ({len(train_texts)}) and labels ({len(train_labels)}) length mismatch!")
    if len(dev_texts) != len(dev_labels):
        raise ValueError(f"Dev texts ({len(dev_texts)}) and labels ({len(dev_labels)}) length mismatch!")

    # Tokenize
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    print("Tokenizing dev data...")
    dev_encodings = tokenizer(
        dev_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create tensors
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
    dev_labels_tensor = torch.tensor(dev_labels, dtype=torch.float)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        train_labels_tensor
    )
    dev_dataset = TensorDataset(
        dev_encodings['input_ids'],
        dev_encodings['attention_mask'],
        dev_labels_tensor
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=batch_size
    )
    
    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = BCEWithLogitsLoss()
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Dev Accuracy: {accuracy:.4f}")
    
    print("\nTraining completed!")
    
    # Return classifier wrapper
    return BERTMultilabelClassifier(model, tokenizer, num_labels, device, max_length)