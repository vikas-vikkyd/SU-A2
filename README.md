# Speech Understanding Programming Assignment - 2
This repository contains Implementation of programming assignment 2.
Repository contains notebooks also which were executed on Google colab.

## Author

- Vikas Kumar Singh - M23AIR545

## Steps to execute the code
- Execute all the commands from project directory
- Use `python -m venv envA2` to create virtual environment
- Use `source envSUA2/bin/activate` to activate the environment
- Use `pip install -r requirements.txt` to install the dependencies
- Data path need to be updated accordingly in all code
- Use `python q1_pretrained.py` to see accuracy for pretrained model
- Use `python q1_finetuning.py` to finetune the "facebook/wav2vec2-xls-r-300m" model using PEFT (LORA)
- Execute SU_A2_q2.ipynb to generate MLE features for provided dataset and Spectrogram for different languages.
- Use `python trainer_q2.py` to train the classification model to identify language

## References
+ https://huggingface.co/docs/transformers/en/model_doc/wav2vec2