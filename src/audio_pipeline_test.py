from datasets import load_dataset 
import torch
from transformers import pipeline

def main(): 
	torch.manual_seed(42)
	ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

	audio_file=ds[0]["audio"]["path"]

	audio_classifier = pipeline(
		task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
	)

	preds = audio_classifier(audio_file)
	
	preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

	print(str(preds))

if __name__ == "__main__": 
	main()

