Brenden Collins // Nate Novak
CS7180: Advanced Computer Perception
Project 3: Sequential Data

Exploration and Analysis of Transformer Encoder Models

OS / IDE:
    MacOS / Vim
    OR
    Ubuntu 22.04 / Vim

Instructions on Running: 
    from the 'src' folder: 
    python3 vec_from_seq.py -> Runs the data through the selected transformer encoder, creating a csv file of the encodings. Ran multiple times to put the data into chuncks
    python3 concat_csv -> utility script that concats several csv files created by vec_from_seq.py
    python3 randomforest.py -> script to cunduct random forest classifer on the encoder embeddings
    python3 dr_sc.py -> script to conduct dimensionality reduction and further exploration of the data. 
