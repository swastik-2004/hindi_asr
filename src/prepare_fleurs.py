from datasets import load_dataset, Audio
import re

LANG = "hi_in"

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)  # keep Hindi chars + space
    text = re.sub(r"\s+", " ", text)
    return text

def prepare_dataset():
    dataset = load_dataset("google/fleurs", LANG)

    # resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess(batch):
        batch["text"] = normalize_text(batch["transcription"])
        batch["input_length"] = len(batch["audio"]["array"])
        return batch

    dataset = dataset.map(preprocess, remove_columns=["transcription"])

    # filter very long clips (Whisper likes <30s)
    dataset = dataset.filter(lambda x: x["input_length"] < 30 * 16000)

    train_size = min(2000, len(dataset["train"]))
    val_size = min(300, len(dataset["validation"]))

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))
    dataset.save_to_disk("data/fleurs_hi_processed")


    return dataset

if __name__ == "__main__":
    ds = prepare_dataset()
    print(ds)
