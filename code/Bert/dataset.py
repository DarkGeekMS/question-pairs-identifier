import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
import pickle


def load_data():

    with open('./dataset/train_encondings_q1.pkl', 'rb') as f:
        train_encodings_1 = pickle.load(f)

    with open('./dataset/train_encondings_q2.pkl', 'rb') as f:
        train_encodings_2 = pickle.load(f)


    with open('./dataset/val_encondings_q1.pkl', 'rb') as f:
        val_encodings_1 = pickle.load(f)

    with open('./dataset/val_encondings_q2.pkl', 'rb') as f:
        val_encodings_2 = pickle.load(f)


    with open('./dataset/train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    with open('./dataset/val_labels.pkl', 'rb') as f:
        val_labels = pickle.load(f)

    return train_encodings_1, train_encodings_2, train_labels, val_encodings_1, val_encodings_2, val_labels


def do_data(csv_file, model_type, train_split_ratio = 0.8):
    # read the data
    dataset = pd.read_csv(csv_file).dropna()

    # split dataset to train-val splits
    q1 = dataset['question1']
    q2 = dataset['question2']
    questions = list(zip(q1, q2))
    labels = dataset['is_duplicate']
    train_questions, val_questions, train_labels, val_labels = train_test_split(questions, labels, train_size = train_split_ratio)

    # encode the text
    if model_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if model_type == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    if model_type == 'albert-base-v2':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    if model_type == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 

    train_questions_1 = [item[0] for item in train_questions]
    train_questions_2 = [item[1] for item in train_questions]
    val_questions_1 = [item[0] for item in val_questions]
    val_questions_2 = [item[1] for item in val_questions]
            
    train_encodings_1 = tokenizer(train_questions_1, truncation=True, padding=True)
    train_encodings_2 = tokenizer(train_questions_2, truncation=True, padding=True)
    val_encodings_1 = tokenizer(val_questions_1, truncation=True, padding=True)
    val_encodings_2 = tokenizer(val_questions_2, truncation=True, padding=True)

    with open('./dataset/train_encondings_q1.pkl', 'wb') as f:
        pickle.dump(train_encodings_1, f)

    with open('./dataset/train_encondings_q2.pkl', 'wb') as f:
        pickle.dump(train_encodings_2, f)


    with open('./dataset/val_encondings_q1.pkl', 'wb') as f:
        pickle.dump(val_encodings_1, f)

    with open('./dataset/val_encondings_q2.pkl', 'wb') as f:
        pickle.dump(val_encodings_2, f)


    with open('./dataset/train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)

    with open('./dataset/val_labels.pkl', 'wb') as f:
        pickle.dump(val_labels, f)


    return train_encodings_1, train_encodings_2, train_labels, val_encodings_1, val_encodings_2, val_labels



class TextToAttrDataset(Dataset):
    def __init__(self, encodings_q1, encodings_q2, labels):
        self.encodings_q1 = encodings_q1
        self.encodings_q2 = encodings_q2
        self.labels = labels

    def __getitem__(self, idx):
        item_q1 = {key: torch.tensor(val[idx]) for key, val in self.encodings_q1.items()}
        item_q2 = {key: torch.tensor(val[idx]) for key, val in self.encodings_q2.items()}
        item_label = torch.tensor(self.labels.iloc[idx])
        item = {
            'q1': item_q1, 
            'q2': item_q2,
            'label': item_label
        }
        return item

    def __len__(self):
        return len(self.labels)
