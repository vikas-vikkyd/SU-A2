import os
import pandas as pd
import librosa
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from peft import LoraConfig, get_peft_model, TaskType
import warnings

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

data_path = "/content/aac"
test_json_path = "q1_test_json.json"
train_json_path = "q1_train_json.json"
output_dir = "peft-wav2vec2-base-960h"

model_id = "facebook/wav2vec2-xls-r-300m"
base_model = AutoModelForCTC.from_pretrained(model_id)
processor = Wav2Vec2Processor.from_pretrained(model_id)


def map_to_array(x):
    speech, _ = librosa.load(x["audio_file_path"], sr=16000, mono=True)
    return {"speech": speech}


def prepare_label_file(data_list, json_path):
    data = []
    for label in data_list:
        for audio_folder in os.listdir(os.path.join(data_path, label)):
            if audio_folder != ".DS_Store":
                for audio_file in os.listdir(
                    os.path.join(data_path, label, audio_folder)
                ):
                    if audio_file != ".DS_Store":
                        data.append(
                            {
                                "audio_file_path": os.path.join(
                                    data_path, label, audio_folder, audio_file
                                ),
                                "labels": label,
                            }
                        )
    # prepare dataframe
    df = pd.DataFrame.from_dict(data)
    df.head(100).to_json(json_path, orient="records", lines=True)


def generate_dataset():
    """Main module to generate training and testing dataset"""
    # read all folder list
    data_list = os.listdir(data_path)
    data_list.sort()
    # data_list.remove(".DS_Store")

    # define train and test list
    train_list = data_list[0:100]
    test_list = data_list[100:]

    # prepare label file
    prepare_label_file(test_list, test_json_path)
    prepare_label_file(train_list, train_json_path)

    # generate train and test
    # train
    train_dataset = load_dataset("json", data_files=train_json_path, split="train")
    train_dataset = train_dataset.map(map_to_array)
    # test
    test_dataset = load_dataset("json", data_files=test_json_path, split="train")
    test_dataset = test_dataset.map(map_to_array)
    return train_dataset, test_dataset


def preprocessing(example):
    example["input_ids"] = processor(
        example["speech"], sampling_rate=16000
    ).input_values[0]
    return example


def main():
    """Main module to finetune the model"""
    train_dataset, test_dataset = generate_dataset()
    train_dataset = train_dataset.map(preprocessing)
    train_dataset = train_dataset.remove_columns(["audio_file_path", "speech"])
    train_dataset = train_dataset.select_columns(["input_ids", "labels"])

    # Define model
    # lora cofig
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=["k_proj", "q_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # define peft model
    peft_model = get_peft_model(base_model, lora_config)
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1,
    )

    # define trainer
    peft_trainer = Trainer(
        model=peft_model, args=peft_training_args, train_dataset=train_dataset,
    )

    # train the model
    peft_trainer.train()


if __name__ == "__main__":
    main()
