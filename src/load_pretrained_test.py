# See: https://huggingface.co/docs/transformers/v4.23.1/en/autoclass_tutorial

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForSequenceClassification

def main(): 
    # Load a tokenizer with AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # Need to use a feature extractor.
    # processes the audio signal or image into correct input format

    feature_extractor = AutoFeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.save_pretrained("/home/nate/cs_7180_proj3/out/bertuncased/")
    return


if __name__ == "__main__": 
    main()
