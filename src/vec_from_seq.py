# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to retrieve the encoded vector encoding for a given text sequence

from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd


###########################################################
# NOTE: due to the size of the transformer and data,      #
#   we ran this in chunks of roughly 1000 rows at a time. #
#   This kept the program from crashing. Hence the use    #
#   of concat_csv.py                                      #
###########################################################
def main(): 
    # load dataset
    print("\nLoading and cleaning data...")
    rt = load_dataset('csv', delimiter='\t', data_files='../data/train.tsv', split='train')

    # Reshape data so we only have full sentences from the rotton tomatoes dataset
    rt_df = pd.DataFrame(rt)
    new_col = rt_df[['SentenceId','PhraseId']].groupby(['SentenceId']).min('PhraseId')
    new  = rt_df.merge(new_col, on=['SentenceId'], how='left', suffixes=(None,'_r'))
    rt_df = new.drop(new[new.PhraseId != new.PhraseId_r].index)
    rt_df = rt_df.reset_index(drop=True)

    rt_small = list(rt_df['Phrase'])[7000:]
    
    print("\nLoading model and checkpoint...")
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(rt_small, padding=True, truncation=True, return_tensors="pt")
    
    print("\nRunning data through pretrained model...")
    model = AutoModel.from_pretrained(checkpoint)
    outputs = model(**inputs)

    cls_out_np = outputs.last_hidden_state.detach().numpy()[:,0,:]

    # write hidden state parameters to file
    print("\nWriting hidden state parameters to file...")
    encoded_df = pd.concat([rt_df[['PhraseId','SentenceId','Phrase','Sentiment']][7000:].reset_index(drop=True), pd.DataFrame(cls_out_np)], axis=1)
    print(encoded_df)
    encoded_df.to_csv("../data/encoded_rt.csv")
     
    return

if __name__ == "__main__": 
    main()

