import numpy as np
import torch
import os
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from model import BertRegressor
from dataset import do_data, load_data, TextToAttrDataset
from loss import MSELoss
from torch.utils.data import DataLoader
from transformers import AdamW
from datetime import datetime 
from tqdm import tqdm
from statistics import mean 
from checkpoint_tracker import CheckpointTracker
import pickle

class Trainer():
    def __init__(self, configs):
        # configs
        self.do_data = configs['do_data']
        self.lr = configs['lr']
        self.lr_decay = configs['lr_decay']
        self.lr_decay_step_size = configs['lr_decay_step_size']
        self.batch_size = configs['batch_size']
        self.num_of_epochs = configs['num_of_epochs']
        self.architecture = configs['architecture']
        self.resume = configs['resume']

        #device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # checkpoint tracker to save model checkpoint
        self.checkpoint_tracker = CheckpointTracker(self.architecture, self.device)

        # load model weights
        self.model = self.checkpoint_tracker.load_checkpoint(self.resume)

        # dataset
        if self.do_data == True:
            train_encodings_1, train_encodings_2, train_labels, val_encodings_1, val_encodings_2, val_labels = do_data(csv_file = 'dataset/train.csv', model_type = self.architecture)
        else:
            train_encodings_1, train_encodings_2, train_labels, val_encodings_1, val_encodings_2, val_labels = load_data()



        train_dataset = TextToAttrDataset(train_encodings_1, train_encodings_2, train_labels)
        val_dataset = TextToAttrDataset(val_encodings_1, val_encodings_2, val_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step_size, gamma=self.lr_decay)
        
        # loss
        self.loss_criterion = nn.CrossEntropyLoss()

        
    def train(self, validate_step):
        '''
        Function defining the entire training loop
        '''
        for epoch in range(self.num_of_epochs):
            self.training_epoch()            

            if epoch % validate_step == (validate_step - 1):
                with torch.no_grad():
                    self.validation_epoch()
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                    f'Epoch: {epoch}\t'
                    f'Train loss: {self.training_epoch_loss:.4f}\t'
                    f'Valid loss: {self.validation_epoch_loss:.4f}\t'
                    f'Valid acc:  {self.validation_epoch_accuracy:.4f}\t'
                    )
                                    
            # save checkpoint
            self.checkpoint_tracker.save_checkpoint(self.model)

            self.scheduler.step()



    ###############
    # train iteration
    def training_epoch(self):
        '''
        single trainig epoch
        '''

        self.model.train()
        self.training_epoch_loss = 0

        print("start training epoch...")
        pbar = tqdm(self.train_loader)
        for batch in pbar:
            
            input_ids_q1 = batch['q1']['input_ids'].to(self.device)
            attention_mask_q1 = batch['q1']['attention_mask'].to(self.device)
            input_ids_q2 = batch['q2']['input_ids'].to(self.device)
            attention_mask_q2 = batch['q2']['attention_mask'].to(self.device)

            y_true = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()

            y_hat = self.model(input_ids_q1, input_ids_q2, attention_mask_q1=attention_mask_q1, attention_mask_q2=attention_mask_q2)

            if len(list(y_hat.size())) < 2:
                y_hat = torch.unsqueeze(y_hat, 0)
            
            loss = self.loss_criterion(y_hat, y_true)
            self.training_epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # accuracy
            pbar.set_description(f'Acc = : {self.get_accuracy(y_hat, y_true):.4f}')

        self.training_epoch_loss = self.training_epoch_loss / len(self.train_loader)

    ###############
    # validate 
    def validation_epoch(self):
        '''
        Function for the validation step of the training loop
        '''
        print("start validation epoch...")
        self.model.eval()
        self.validation_epoch_loss = 0

        batches_accuracies = []
        pbar = tqdm(self.validation_loader)
        for batch in pbar:
            input_ids_q1 = batch['q1']['input_ids'].to(self.device)
            attention_mask_q1 = batch['q1']['attention_mask'].to(self.device)
            input_ids_q2 = batch['q2']['input_ids'].to(self.device)
            attention_mask_q2 = batch['q2']['attention_mask'].to(self.device)
            y_true = batch['label'].to(self.device)
            
            y_hat = self.model(input_ids_q1, input_ids_q2, attention_mask_q1=attention_mask_q1, attention_mask_q2=attention_mask_q2)

            if len(list(y_hat.size())) < 2:
                y_hat = torch.unsqueeze(y_hat, 0)

            loss = self.loss_criterion(y_hat, y_true) 

            self.validation_epoch_loss += loss.item()

            # accuracy
            accuracy = self.get_accuracy(y_hat, y_true)
            pbar.set_description(f'Acc = : {accuracy:.4f}')
            batches_accuracies.append(accuracy)

        self.validation_epoch_loss = self.validation_epoch_loss / len(self.validation_loader)
        self.validation_epoch_accuracy = mean(batches_accuracies)


    ###############
    # batch accuracy
    def get_accuracy(self, y_hat, y_true):
        y_hat = torch.argmax(y_hat, 1)
        accuracy = (y_hat == y_true).sum()/y_true.size(0)
        return accuracy.item()
        



