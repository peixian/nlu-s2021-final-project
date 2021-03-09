# following the tutorial here: https://huggingface.co/transformers/custom_datasets.html
# and here: https://huggingface.co/docs/datasets/quicktour.html
import transformers
import datasets
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

        

def get_tokenized_train_val_split(model_type, train_dataset, text_col_name, label_col_name):

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    train_dataset_temp =  train_dataset.train_test_split()
    train_texts, train_labels = train_dataset_temp['train'][text_col_name], train_dataset_temp['train'][label_col_name]
    val_texts, val_labels = train_dataset_temp['test'][text_col_name], train_dataset_temp['test'][label_col_name]

    train_encodings = tokenizer(train_texts, return_tensors='pt', truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, return_tensors='pt', truncation=True, padding=True)
    

    train_dataset = TokenizedDataset(train_encodings, train_labels)
    val_dataset = TokenizedDataset(val_encodings, val_labels)

    return train_dataset, val_dataset

def get_fine_tuned_model(model_type, train_dataset, val_dataset, model_class=None):
    

    if model_class is None: 
        model = AutoModel.from_pretrained(model_type)
    else: 
        model = model_class.from_pretrained(model_type)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

    return model

