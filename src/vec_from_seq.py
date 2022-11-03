# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to extract the vector of audio sequence
from transformers import pipeline

import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def main(): 

    rot_tom = load_dataset("rotten_tomatoes", split="train")
    print(f"rot_tom: {rot_tom}")

    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    
    embeddings = model.get_input_embeddings()
    print(f"embeddings.shape: {embeddings}")

    # create ids of encoded input vectors
    for i in range(len(rot_tom["text"])): 
        sentence = rot_tom["text"][i]
        label = rot_tom["label"][i]
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        print(f"input_ids {input_ids}")
        last_hidden_state = model.base_model.encoder(input_ids, return_dict=True).last_hidden_state
        
        break


    ###################################
    # Below this is old stuff. Ignore #
    ###################################
   # pd_data = pd.DataFrame(rot_tom)
   # print(f"pd_data.head {pd_data.head()}")
   # #print(f"np_label.shape: {np_label.shape}")

   # pd_data[["model_out", "model_score"]] = pd.DataFrame([x for x in model(list(pd_data["text"]))])

   # d = {'POSITIVE':1, 'NEGATIVE': 0}
   # pd_data['model_out'] = pd_data['model_out'].map(d)

   # grouped = pd_data.groupby(["label", "model_out"]).count()

   # print(f"{grouped}")

   # # See: https://huggingface.co/blog/fine-tune-wav2vec2-english

    return

sentences = [ 
    "I quite happy, but mourning my mother's death", 
    "I'm so happy my brother died", 
    "My brother died after a long hard battle with cancer, and I'm kind of thrilled he suffered until the bitter end",
    "hjksadf klj asdkl; weoi[rj hfd",
    "aaaaaaaaaa ooooo uuuuuiiiiiii oooooo iiiiiiiiiiiI",
    "AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH", 
    "IIIIIIIIIIIIIIIIIIIIIIIIIII", 
    "yyeooourya hdgagej alagejja gakuuuq scbnns ooiuthw ahdeg",
    "loiiaiuivhea ha keoah lower haheaee libuea ahehch?",
    "hahahahahahahahaha"
    ]

if __name__ == "__main__": 
    main()
