import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from transformers import  DistilBertModel, AlbertModel, RobertaModel, BertModel, AdamW

class BertRegressor(nn.Module):    
    def __init__(self, model_type):
        super(BertRegressor,self).__init__()

        if model_type == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if model_type == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if model_type == 'albert-base-v2':
            self.bert = AlbertModel.from_pretrained('albert-base-v2')

        if model_type == 'roberta-base':
            self.bert = RobertaModel.from_pretrained('roberta-base') 

        self.linear = nn.Linear(768, 2, bias=True)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.1)
                
    def forward(self, input_ids_q1, input_ids_q2, attention_mask_q1, attention_mask_q2):
        outputs_q1 = self.bert(input_ids_q1, attention_mask=attention_mask_q1)
        outputs_q2 = self.bert(input_ids_q2, attention_mask=attention_mask_q2)
        pooled_output_q1,_ = torch.max(outputs_q1['last_hidden_state'], 1)
        pooled_output_q2,_ = torch.max(outputs_q2['last_hidden_state'], 1)
        pooled_output = torch.max(pooled_output_q1, pooled_output_q2)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        logits = self.softmax(logits)
        return logits



class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier,self).__init__()
        embedding_dim = 155
        hidden_dim = 128

        self.BiLSTM = nn.LSTM(input_size = embedding_dim,
                                hidden_size = hidden_dim,
                                num_layers = 1, 
                                bidirectional = True)
        
        self.BiLSTM2 = nn.LSTM(input_size = hidden_dim * 2,
                                hidden_size = hidden_dim * 2,
                                num_layers = 1, 
                                bidirectional = True)
    
    def forward(self, input_ids_q1, input_ids_q2, attention_mask_q1, attention_mask_q2):
        q1,_ = self.BiLSTM(input_ids_q1)
        return x




class LSTMClassifier(nn.Module):

    def __init__(self, dimension=128):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out
