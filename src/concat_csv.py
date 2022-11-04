# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to concatenate CSV files because our machines could not
#   run DistelBERT on all 8600 sentences at once


# https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
import os
import glob
import pandas as pd


os.chdir("../data")
ext = 'csv'

files = [i for i in glob.glob(f'*.{ext}')]

combined_csv = pd.concat([pd.read_csv(f) for f in files])

# export to csv
combined_csv.to_csv("../data/encoded.csv", index=False, encoding='utf-8-sig')


