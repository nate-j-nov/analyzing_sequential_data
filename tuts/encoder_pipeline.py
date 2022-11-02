# See: https://huggingface.co/docs/transformers/v4.21.3/en/tasks/audio_classification

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

TRAIN = "train"
def main(): 
    minds = load_dataset("PolyAI/minds14", name = "en-US", split = "train")
    minds = minds.train_test_split(test_size = 0.2)

    # Check out the dataset:
    print(f"minds dataset: {minds}")

    minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
    print(f"minds[0] trimmed: {minds[TRAIN][0]}")

    labels = minds["train"].features["intent_class"].names

    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels): 
        label2id[label] = str(i)
        id2label[str(i)] = label

    print(f"id2label[2] = {id2label[str(2)]}")

    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_minds = encoded_minds.rename_column("intent_class", "label")
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        tokenizer=feature_extractor,
    )

    trainer.train()
    return

    

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )

    return inputs

if __name__ == "__main__": 
    main()
