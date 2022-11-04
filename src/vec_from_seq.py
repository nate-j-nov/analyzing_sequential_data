# Brenden Collins // Nate Novak
# CS 7180: Advanced Computer Perception
# Fall 2022
# Program to extract the vector of audio sequence

from transformers import pipeline
from transformers import AutoTokenizer, AutoModel

import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd
from sklearn.decomposition import PCA

def main(): 

    # load dataset
    rt = load_dataset('csv', data_files='../data/train.csv', split='train')
    print(rt)
    print(f"rt[0]: {rt[0]}")

#### Toy example of tokenizer/model split on handmade sentences
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I feel like using HuggingFace is maybe too easy.",
        "I also don't know what to use instead that will be feasible",
        "NLP models are far too large to get good results without something pre-trained",
        "Pizza is delicious!"
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    model = AutoModel.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)


#    outputs_np = outputs.last_hidden_state.detach().numpy()
#    print(outputs_np)
#    out_shape = outputs_np.shape
#    print(out_shape)
#    outputs_np = outputs_np.reshape(out_shape[0], out_shape[1]*out_shape[2])
#    print(outputs_np.shape)




##### Use pandas to checkout model output against ground truth
#    pd_data = pd.DataFrame(rot_tom)
#    print(f"pd_data.head {pd_data.head()}")
#
#    pd_data[["model_out", "model_score"]] = pd.DataFrame([x for x in model(list(pd_data["text"]))])
#    d = {'POSITIVE':1, 'NEGATIVE': 0}
#    pd_data['model_out'] = pd_data['model_out'].map(d)
#
#    grouped = pd_data.groupby(["label", "model_out"]).count()
#    print(f"{grouped}")

##########
### Test np reshape because I always forget how it works
##########
#    test = np.zeros((2,3,4))
#    test[0,1,:] = 1
#    test[0,2,:] = 2
#    test[1,0,:] = 3
#    test[1,1,:] = 4
#    test[1,2,:] = 5
#    print(test)
#    print(test.shape)
#
#    test2 = test.reshape((2,12))
#    print(test2)

#########
# Play with a single sentence encoding
#########
#    a = outputs_np[0,:,:]
#    print(a.shape)
#    print(f"a: {a}")
#
#    pca = PCA()
#    pca.fit(a.T)
#    print(f"Num components: {pca.n_components_}")
#    print(pca.explained_variance_ratio_)
#    print(pca.singular_values_)

    
    # See: https://huggingface.co/blog/fine-tune-wav2vec2-english
    # See also: https://huggingface.co/course/chapter2/2?fw=pt
     
    return

if __name__ == "__main__": 
    main()


#sentences = [ 
#    "I quite happy, but mourning my mother's death", 
#    "I'm so happy my brother died", 
#    "My brother died after a long hard battle with cancer, and I'm kind of thrilled he suffered until the bitter end",
#    "hjksadf klj asdkl; weoi[rj hfd",
#    "aaaaaaaaaa ooooo uuuuuiiiiiii oooooo iiiiiiiiiiiI",
#    "AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH", 
#    "IIIIIIIIIIIIIIIIIIIIIIIIIII", 
#    "yyeooourya hdgagej alagejja gakuuuq scbnns ooiuthw ahdeg",
#    "loiiaiuivhea ha keoah lower haheaee libuea ahehch?",
#    "hahahahahahahahaha"
#    ]



#    rot_tom = load_dataset("rotten_tomatoes", split="train")
#    print(rot_tom.info)
#    print(f"rot_tom: {rot_tom}")
#    print(f"rot_tom[0]: {rot_tom[0]}")
#    model = pipeline('sentiment-analysis')
#    outputs = model(rot_tom["text"][:30])
#    print(outputs)
