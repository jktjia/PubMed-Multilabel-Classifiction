# PubMed Multilabel Classification

Developing and comparing models for multilabel classification of PubMed MeSH root labels.

The data used in this project comes from [PubMed MultiLabel Text Classification Dataset MeSH](https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification).

## Running multilabel classification models

To train and test the multilabel classifiers locally on small versions of the datasets, run the following command:

```
python multilabel_classifier.py --dataset SMALL [--model MODEL] [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
```

Currently, the available models are TRIVIAL (trivial classifier that assigns every label as false), LR (logistic regression), CNN, RNN and BERT (BioBert). BERT runner is for demo-ing in model-class format - for actual run use BERT_Colab/BERT_Colab_run.ipynb (>2 hrs train time on A100)
