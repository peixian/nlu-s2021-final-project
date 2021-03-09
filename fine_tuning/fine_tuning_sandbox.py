from fine_tune_boilerplate import TokenizedDataset, get_tokenized_train_val_split, get_fine_tuned_model
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
import torch


train_dataset = load_dataset('glue', 'cola', split='train')
model_type = 'bert-base-cased'
train_dataset, val_dataset = get_tokenized_train_val_split(model_type, train_dataset, 'sentence', 'label')

ft_model = get_fine_tuned_model(model_type, train_dataset, val_dataset, model_class=BertForSequenceClassification)
