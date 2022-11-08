# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to extract the vector of audio sequence

from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
import torch

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

    chunks = list(range(0, 7000, 1000))

    checkpoint = "../out/"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained(checkpoint)
    for chunk in chunks: 
        print(f"Running chunk {chunk}...")
        if chunk == 6000: 
            end = len(rt_ds['train'])
            rt_small = list(rt_ds['train']['text'])[chunk:]
        else: 
            end = chunk + 1000
            rt_small = list(rt_ds['train']['text'])[chunk:chunk+1000]
        inputs = tokenizer(rt_small, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): 
            outputs = model(**inputs)
            cls_out_np = outputs.last_hidden_state.detach().numpy()[:,0,:]
            encoded_df = pd.concat([rt_df[['PhraseId','SentenceId','Phrase','Sentiment']][chunk:end].reset_index(drop=True), pd.DataFrame(cls_out_np)], axis=1)
            encoded_df.to_csv(f"../data/e_{chunk}_rt_fine_tuned.csv")

    return

if __name__ == "__main__": 
    main()

