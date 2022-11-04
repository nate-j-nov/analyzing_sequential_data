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
    rt = load_dataset('csv', delimiter='\t', data_files='../data/train.tsv', split='train')
    print(rt)
    print(f"rt[0]: {rt[0]}")

    rt_df = pd.DataFrame(rt)
    new_col = rt_df[['SentenceId','PhraseId']].groupby(['SentenceId']).min('PhraseId')
    new  = rt_df.merge(new_col, on=['SentenceId'], how='left', suffixes=(None,'_r'))
    rt_df = new.drop(new[new.PhraseId != new.PhraseId_r].index)
    rt_df = rt_df.reset_index(drop=True)
    print(rt_df)

    rt_small = list(rt_df['Phrase'])[7000:]
    print(rt_small[:5])
#    rt_input = list(rt_df['Phrase'])

#### Toy example of tokenizer/model split on handmade sentences
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(rt_small, padding=True, truncation=True, return_tensors="pt")
    print(inputs)
    
    model = AutoModel.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)


#  https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
# https://datasciencetoday.net/index.php/en-us/nlp/211-paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained
    cls_out_np = outputs.last_hidden_state.detach().numpy()[:,0,:]
    print(f"cls_out_np: {cls_out_np}")
    print(f"cls_out_np shape: {cls_out_np.shape}")
#    outputs_np = outputs_np.reshape(out_shape[0], out_shape[1]*out_shape[2])
#    print(outputs_np.shape)

    # write hidden state parameters to file
    print(f"rt_df: {rt_df}")
    encoded_df = pd.concat([rt_df[['PhraseId','SentenceId','Phrase','Sentiment']][7000:].reset_index(drop=True), pd.DataFrame(cls_out_np)], axis=1)
    print(encoded_df)
    encoded_df.to_csv("../data/encoded_rt.csv")

#    # PCA on all sentences - CLS vector only
#    pca = PCA()
#    pca.fit(cls_out_np)
#    
#    print(f"Num components: {pca.n_components_}")
#    print(pca.explained_variance_ratio_)
#    test = np.array(pca.explained_variance_ratio_)[:30].sum()
#    print(test)
#    print(pca.singular_values_)


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
