# See: https://huggingface.co/docs/transformers/v4.23.1/en/preprocessing
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

audio = "audio" # dealing with formatted text
array = "array"
def main(): 

    dataset = load_dataset("PolyAI/minds14", name="en-US", split="train") 

    # It's important our audio's data sampling rate 
    # matches the sampling rate of the dataset used to pretrain the model. 
    # You can upsample by doing the following: 
    dataset = dataset.cast_column(audio, Audio(sampling_rate=16_000))

    print(f"Dataset sample: {dataset[0][audio]}")
    # Next, load a feature extractor the normalize and pad inputs. 
    # When padding text data, a 0 is added for shorter sequences
    # The same idea is applied to audio data. 
    
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # Pass the audio array to the feature extractor.
    # They also recommend adding the sampling_rate arg in the 
    # feat extractor to be better debug any silent errors that may appear. 
    
    audio_input = [dataset[0][audio][array]]
    
    print(f"feature Extractor: {feature_extractor(audio_input, sampling_rate = 16000)}")
    print(str(feature_extractor)) 

    print("Shapes to show different sizes in input data:"); 
    print(str(dataset[0]["audio"]["array"].shape))
    print(str(dataset[1]["audio"]["array"].shape))

    # Apply the preprocess_function tot he first few samples in the dataset
    preprocessed_data = preprocess(dataset[:5], feature_extractor)
    input_values = "input_values"

    print(f"Preprocessed dataset FIRST: {preprocessed_data[input_values][0].shape}")
    print(f"Preprocessed dataset SECOND: {preprocessed_data[input_values][1].shape}")
    return

# create function to preprocess dataset so audio samples are same length
def preprocess(data, feature_extractor): 
    audio_arrays = [x[array] for x in data[audio]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )

    return inputs

if __name__ == "__main__": 
    main();
