This directory contains different versions of bert used as a text processor



# Available Bert Versions
- bert-base-uncased
- distilbert-base-uncased
- albert-base-v2
- roberta-base



# To Train 

1. Download [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data?select=train.csv.zip).

2. Place it into `dataset` directory.

3. Install transformers.
```bash
    pip install pytorch-transformers
```

4. Fine tune Bert model
```bash
    python main.py\ 
    -arch <bert_version>\
    -lr <learning_rate>\
    -batch <batch_size>\
    -validate_step <validation_step>\
    -epochs <number_of_epochs>\
```
- **bert_version**: any version of the 4 stated above, default is `distilbert-base-uncased`
- **learning_rate**: learning rate used to train bert model, default is `5e-5`
- **batch_size**: number of samples per batch, default is `16`
- **validation_step**: validate each how many epochs, default is `1`
- **number_of_epochs**: number of epochs, default is `50`

e.g. to train `bert-base-uncased` version
```bash
    python main.py -arch bert-base-uncased
```

5. To complete training from the last training time, add `--resume_from_last_trial` option to the command in (*4*).




