# See: https://huggingface.co/blog/encoder-decoder

from transformers import MarianMTModel, MarianTokenizer
import torch

def main(): 
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    print(f"Tokenizer: {tokenizer}")

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    #print(f"model: {model}")

    # create ids of encoded input vectors
    input_ids = tokenizer("I want to buy a car", return_tensors="pt").input_ids
    print(f"input_ids {input_ids}")

    # Create beginning-of-sentence token
    decoder_input_ids = tokenizer("<pad>", add_special_tokens = False, return_tensors="pt").input_ids
    print(f"decoder_input_ids {decoder_input_ids}")
    assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"

    # Pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
    outputs = model(input_ids, decoder_input_ids = decoder_input_ids, return_dict = True)
    #print(f"outputs: {outputs}")
    #print(f"outputs.shape: {outputs.shape}")

    # get the encoded sequence 
    encoded_sequence = outputs.encoder_last_hidden_state
    print(f"encoded_sequence.shape: {encoded_sequence.shape}")
    print(f"encoded_sequence: {encoded_sequence}")

    return 

if __name__ == "__main__": 
    main()
