# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Perform dimension reduction and clustering analyses on encoded sentence data 


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def main():

    # load data
    cls_out = pd.read_csv("../data/encoded.csv")
    print(cls_out)
    print(cls_out.columns)

    cls_np = np.array(cls_out.drop(["Unnamed: 0", "PhraseId", "SentenceId", "Phrase"], axis=1))
    print(f"cls_np shape: {cls_np.shape}")

    sentiment = cls_np[:,0]
    vector = cls_np[:,1:]
    print(sentiment)
    print(vector.shape)

    # PCA on all sentences - CLS vector only
    pca = PCA()
    pca.fit(vector)
    
    print(f"Num components: {pca.n_components_}")
    print(f"First 100 axes variance: {pca.explained_variance_ratio_[:100]}")
    test = np.array(pca.explained_variance_ratio_)[:10].sum()
    print(f"Percent of variance in the first 10 axes: {test}")
#    print(pca.singular_values_)

    return

if __name__ == "__main__": 
    main()
