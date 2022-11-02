# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to extract the vector of audio sequence
from transformers import pipeline

import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd

def main(): 


    rot_tom = load_dataset("rotten_tomatoes", split="train")
    print(f"rot_tom: {rot_tom}")

    model = pipeline('sentiment-analysis')

    pd_data = pd.DataFrame(rot_tom)
    print(f"pd_data.head {pd_data.head()}")
    #print(f"np_label.shape: {np_label.shape}")

    pd_data[["model_out", "model_score"]] = pd.DataFrame([x for x in model(list(pd_data["text"]))])

    d = {'POSITIVE':1, 'NEGATIVE': 0}
    pd_data['model_out'] = pd_data['model_out'].map(d)

    grouped = pd_data.groupby(["label", "model_out"]).count()

    print(f"{grouped}")

    
    #pd_data["model_out"] = trans(pd_data["model_out"])
    #pd_data[["model_score", "model_out"]] = pd_data["model"][["label", "score"]]
    #print(f"pd_data.head {pd_data.head(30)}")

   # for i in range(len(text)): 
   #     sentence = text[i]
   #     rt_label= label[i]
   #     output = model(sentence)
   #     print(f"{sentence} | rot_tom label: {rt_label}\n{output}")

    
   # print(f"output.shape: {np.array(output).shape}"); 
   # print(f"last hidden layer mask: {output[0][-1]}")

    # See: https://huggingface.co/blog/fine-tune-wav2vec2-english


     

    #wave2vec = 

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
