# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Performing a random forest classifier on the encodings


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import math
import sys

def main():

    print("\nReading encoded data from csv...")
    cls_out = pd.read_csv("../data/encoded.csv")

    cls_np = np.array(cls_out.drop(["Unnamed: 0", "PhraseId", "SentenceId", "Phrase"], axis = 1))

    print("\nCreating train and evaluation data..")

    train_split=math.floor(0.8*cls_np.shape[0])
    
    # Create train split
    train_data = cls_np[:train_split, 1:]
    train_labels = cls_np[:train_split, 0] 
    
    # Create eval split
    eval_data = cls_np[train_split:, 1:]
    eval_labels = cls_np[train_split:, 0] 
    
    # Verify all data is being used
    print(f"train + eval = {cls_np.shape[0]}: {train_data.shape[0] + eval_data.shape[0] == cls_np.shape[0]}")
    
    print("\nRunning random forest...")
    randfor = RandomForestClassifier()
    randfor.fit(train_data, train_labels)
    rf_score = randfor.score(eval_data, eval_labels)

    print(f"Random Forest Score: {rf_score}")

    print("\nRunning on dummy classifier...")
    dummy = DummyClassifier(strategy = "uniform")
    dummy.fit(train_data, train_labels)
    d_score = dummy.score(eval_data, eval_labels)
    
    print(f"Dummy Score: {d_score}")

    print("\nRunning Random Forest on first 100 dimensions...")
    train_data = cls_np[:train_split, 1:101]
    train_labels = cls_np[:train_split, 0] 

    eval_data = cls_np[train_split:, 1:101]
    eval_labels = cls_np[train_split:, 0] 

    randfor.fit(train_data, train_labels)
    rf_score_100 = randfor.score(eval_data, eval_labels)
    print(f"Random forest on 100 dims: {rf_score_100}")

    print("Done")
    
    return

if __name__ == "__main__": 
    main()
