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
        To create data:
            python3 vec_from_seq.py -> Runs the data through the selected transformer encoder, creating a csv file of the encodings. Ran multiple times to put the data into chuncks
              - NOTE: each output 'encoded_rt.csv' file had a suffix manually appended
            python3 concat_csv -> utility script that concats several csv files created by vec_from_seq.py
              - NOTE: concatenates all csv files in project/data/ folder
        Once 'encoded.csv' is obtained:
            python3 randomforest.py -> script to conduct random forest classifer on the encoder embeddings
            python3 dr_sc.py -> script to conduct dimensionality reduction and further exploration of the data. 
