# See: https://huggingface.co/docs/transformers/v4.23.1/en/training

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import os

train = "train"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Trainer does not automatically evaluate model performance during traing. 
# Pass trainer a function to compute report metrics. The HF evaluate 
# lib provides simple accuracy func you can load. 
# Also imported numpy at this step
metric = evaluate.load("accuracy")

def main(): 
    dataset = load_dataset("yelp_review_full"); 
    # print(f"dataset[train][100]: {dataset[train][100]}")

    # Some of this will be review
    tokenized_datasets = dataset.map(tokenize_func, batched = True) 

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
    
    # Transformers provides a trainer class optimized for training HF
    # Start by loading the model and specify the number of expected labels. 
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 5)
    


    # Specify where to save checkpoints 
    # Can also change the hyper parameters in here. 
    training_args = TrainingArguments(output_dir = "test_trainer", evaluation_strategy="epoch")

    # Create Trainer object with your model, training args,
    # training and test datasets, and eval function
    trainer = Trainer(
                model = model, 
                args = training_args, 
                train_dataset = small_train_dataset,
                eval_dataset = small_eval_dataset,
                compute_metrics = compute_metrics
                )

    trainer.train()
    trainer.save_model("/home/nate/cs_7180_proj3/out/")

    return 

# This helps dataset.map tokenize the data
def tokenize_func(examples): 
    return tokenizer(examples["text"], padding = "max_length", truncation = True)

# call computer on metric to calculate the accuracy of predictions.
# Before passing predictions to compute, you need to convert predictions to logits
# (All HF models return logits)
def compute_metrics(eval_pred): 
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)

if __name__ == "__main__": 
    main()
