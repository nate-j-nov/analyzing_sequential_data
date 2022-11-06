# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Perform dimension reduction and clustering analyses on encoded sentence data 


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import math
import sys

def main():

    print("Reading encoded data from csv...")
    cls_out = pd.read_csv("../data/encoded_nograd.csv")
    #print(f"cls_out.head: {cls_out.head()}")
    #print(f"cls_out.columns: {cls_out.columns}")

    cls_np = np.array(cls_out.drop(["Unnamed: 0", "PhraseId", "SentenceId", "Phrase"], axis = 1))
   # print(f"cls_np shape: {cls_np.shape}")
   # print(f"cls_np head: {cls_np[:10]}")

    print("Creating train and evaluation data...")

    train_split=math.floor(0.8*cls_np.shape[0])
    
    # Create train split
    train_data = cls_np[:train_split, 1:]
    train_labels = cls_np[:train_split, 0] 
    counter = 0
    for l in train_labels: 
        if str(l) == "nan":
            print(l)
            print(str(counter))
            sys.exit()
        counter += 1
        print(l)

    #print(f"train_data.shape {train_data.shape}")
    #print(f"train_labels.shape {train_labels.shape}")
    
    # Create eval split
    eval_data = cls_np[train_split:, 1:]
    eval_labels = cls_np[train_split:, 0] 

    #print(f"eval_data.shape {eval_data.shape}")
    #print(f"eval_labels.shape {eval_labels.shape}")
    
    # Verify all data is being used
    print(f"train + eval = {cls_np.shape[0]}: {train_data.shape[0] + eval_data.shape[0] == cls_np.shape[0]}")
    
    print("Running random forest...")
    randfor = RandomForestClassifier()
    randfor.fit(train_data, train_labels)
    rf_score = randfor.score(eval_data, eval_labels)

    print(f"Random Forest Score: {rf_score}")

    print("Running on dummy classifier...")
    dummy = DummyClassifier(strategy = "uniform")
    dummy.fit(train_data, train_labels)
    d_score = dummy.score(eval_data, eval_labels)

    print(f"Dummy Score: {d_score}")
    
    print("Done")
    
    return

if __name__ == "__main__": 
    main()
