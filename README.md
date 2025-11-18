# PubMed Multilabel Classification

Developing and comparing models for multilabel classification of PubMed MeSH root labels.

The data used in this project comes from [PubMed MultiLabel Text Classification Dataset MeSH](https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification).

## Running multilabel classification models

To train and test the multilabel classifiers locally on small versions of the datasets, run the following command:

```
python multilabel_classifier.py [--model MODEL] [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
```

Currently, the available models are TRIVIAL (trivial classifier that labels every label as true) and LR (logistic regression).
