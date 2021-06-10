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


