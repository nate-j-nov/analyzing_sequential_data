# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to extract the vector of audio sequence 

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
from sklearn.decomposition import PCA
import torch

CHECKPOINT = "distilbert-base-uncased"
rt_df = pd.DataFrame()

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels = 5)
metric = load_metric("accuracy")

def tokenize_func(examples): 
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)

def main(): 

    # load and reshape dataset
    print("Loading and shaping the dataset...")
    rt = load_dataset('csv', delimiter='\t', data_files='../data/train.tsv', split='train')
    rt_df = pd.DataFrame(rt)
    new_col = rt_df[['SentenceId','PhraseId']].groupby(['SentenceId']).min('PhraseId')
    new  = rt_df.merge(new_col, on=['SentenceId'], how='left', suffixes=(None,'_r'))
    rt_df = new.drop(new[new.PhraseId != new.PhraseId_r].index)
    rt_df = rt_df.reset_index(drop=True)

    rt_ds = Dataset.from_pandas(rt_df)

    rt_ds = rt_ds.train_test_split(test_size = 0.2, shuffle = True)
    rt_ds = rt_ds.remove_columns(["PhraseId", "SentenceId", "PhraseId_r"])
    rt_ds = rt_ds.rename_column("Phrase", "text")
    rt_ds = rt_ds.rename_column("Sentiment", "label")

    print("Tokenizing dataset...")
    tokenized_datasets = rt_ds.map(tokenize_func, batched = True)
    print(tokenized_datasets)

    small_train_ds = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_ds = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

    #    inputs = tokenizer(rt_small, padding=True, truncation=True, return_tensors="pt")

    training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = small_train_ds,
        eval_dataset = small_eval_ds,
        compute_metrics = compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("/home/nate/cs_7180_proj3/out/")
    
if __name__ == "__main__": 
    main()

