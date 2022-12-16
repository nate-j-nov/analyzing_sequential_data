# Brenden Collins // Nate Novak
# CS 7180: Advanced Perception
# Fall 2022
# Program to compare accuracy of the various models. 

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

#####################
# POTENTIALL REMOVE #
#####################

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

    test = rt_ds['test']

    #print(f"rt_labels: {rt_labels}\nrt_text: {rt_text}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(test['text'][:10], padding = True, truncation = True, return_tensors= "pt")
    not_ft = AutoModel.from_pretrained("distilbert-base-uncased")
    ft = AutoModel.from_pretrained("../out/")
    print("Running not fine-tuned model...")
    not_ft_out = not_ft(**inputs)
    print(f"not_ft_out: {not_ft_out[0]}")
    
    print("Running fine-tuned model...")
    ft_out = ft(**inputs).logits
    print(f"ft_out: {ft_out[0]}")

    return 

if __name__ == "__main__": 
    main()
