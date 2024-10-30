import os
import pandas as pd
import numpy as np
import torch
import re

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    AdamW,
)


import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# define a class for the AMP data that will correctly format the sequence information
# for fine-tuning with huggingface API
# the input dataframe columns must be formatted the same way as the given example


class amp_data:
    def __init__(self, df, tokenizer_name="Rostlab/prot_bert_bfd", max_len=200):

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=False
        )
        self.max_len = max_len

        self.seqs, self.labels = self.get_seqs_labels()

    def get_seqs_labels(self):
        # isolate the amino acid sequences and their respective AMP labels
        seqs = list(df["aa_seq"])
        labels = list(df["AMP"].astype(int))

        #         assert len(seqs) == len(labels)
        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(
            seq, truncation=True, padding="max_length", max_length=self.max_len
        )

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample["labels"] = torch.tensor(self.labels[idx])

        return sample


# read in the train dataset
# create an amp_data class of the dataset

data_url = "https://raw.githubusercontent.com/GIST-CSBL/AMP-BERT/main/all_veltri.csv"
df = pd.read_csv(data_url, index_col=0)
df = df.sample(frac=1, random_state=0)
print(df.head(7))

train_dataset = amp_data(df)


# define the necessary metrics for performance evaluation


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    #     conf = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        #         'confusion matrix': conf
    }


# define the model initializing function for Trainer in huggingface


def model_init():
    return BertForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd")


# training on entire data
# no evaluation/validation

# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=15,
#     learning_rate=5e-5,
#     per_device_train_batch_size=1,
#     warmup_steps=0,
#     weight_decay=0.1,
#     logging_dir="./logs",
#     logging_steps=100,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy="no",
#     save_strategy="no",
#     gradient_accumulation_steps=64,
#     fp16=True,
#     fp16_opt_level="O2",
#     run_name="AMP-BERT",
#     seed=0,
#     load_best_model_at_end=True,
# )


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    fp16=False,  # Disable mixed precision
    # If you are on Apple M1/M2, enable this
    bf16=(
        True if torch.backends.mps.is_available() else False
    ),  # Enable bfloat16 precision on mps if available
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


# performance metrics on the training data itself

predictions, label_ids, metrics = trainer.predict(train_dataset)
print(metrics)
