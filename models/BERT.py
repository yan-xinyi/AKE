# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''
   This python file mainly contains code for training, testing and evaluation of Pre-trained model based AKE models for BERT Pre-trained model.
   The main function is BERT(), called from main().
      1* Through the TextDataSet (Dataset): build the dictionary for vovabulary and cognitive signals.
      2* Build BERT() model
      3* Start training and calculate the loss value.
      4* Conduct testing, again read the data first, build the model, in the prediction, evaluate its results.
'''

from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from config import *
import torch.nn as nn
import torch
from evaluate import *
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import f1_score

weight = 'bert-base-uncased'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_len = 35


# Load the train data
def load_traindata(train_path):
    train_file = json.load(open(train_path,'r',encoding='utf-8'))
    train_sens, train_features, train_tags = [],[],[]
    train_word_nums = []

    sens = ''
    nums = 0
    for key in train_file.keys():
        tags = []
        items = train_file[key]
        sens = ''
        nums = 0
        features = []
        for item in items:
            sens += item[0]
            sens += ' '
            tags.append(item[-1])
            features.append(item[1:-1])
            nums += 1
        train_sens.append(sens.strip())
        train_word_nums.append(nums)
        train_tags.append(tags)
        train_features.append(features)
    return train_sens, train_word_nums, train_tags, train_features


# Load the test data
def load_testdata(test_path):
    test_file = json.load(open(test_path, 'r', encoding='utf-8'))
    test_sens, test_features, test_tags = [],[],[]
    test_word_nums = []

    sens = ''
    nums = 0
    for key in test_file.keys():
        tags = []
        items = test_file[key]
        sens = ''
        nums = 0
        features = []
        for item in items:
            sens += item[0]
            sens += ' '
            tags.append(item[-1])
            features.append(item[1:-1])
            nums += 1
        test_sens.append(sens.strip())
        test_word_nums.append(nums)
        test_tags.append(tags)
        test_features.append(features)
    return test_sens, test_word_nums, test_tags, test_features


# Align the tags to words
def align_label(text,labels,features, tokenizer):
  input = tokenizer(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt')
  word_ids = input.word_ids()
  input_ids = input['input_ids'] 
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  
  previous_word_idx = None
  new_labels = []
  new_features = []
  no_features = [0 for i in range(1,26)]

  for word_idx in word_ids:
      if word_idx is None:
          new_labels.append('none')
          new_features.append(no_features)
        #   new_labels.append('O')

      elif word_idx != previous_word_idx:
          try:
              new_labels.append(labels[word_idx])
              new_features.append(features[word_idx])
          except:
              new_labels.append('none')
              new_features.append(no_features)
            #   new_labels.append('O')
      else:
          try:
              new_labels.append(labels[word_idx] if label_all_tokens else 'none')
              new_features.append(features[word_idx] if label_all_tokens else no_features)
            #   new_labels.append(labels[word_idx] if label_all_tokens else 'O')
          except:
              new_labels.append('none')
              new_features.append(no_features)
      previous_word_idx = word_idx

  label_ids = [tag2ids[label] for label in new_labels]
  return label_ids, tokens, new_features


# Biuld the data set
class MyDataset(Dataset):
    def __init__(self, texts, old_features, tags, tokenizer):
        self.texts = texts
        self.tags = tags
        self.old_features = old_features
        
        self.labels = []
        self.tokens = []
        self.features = []
        
        self.input_ids = None
        self.attention_masks = None
        self.tokenizer = tokenizer

    def encode(self):
        for i in tqdm(range(len(self.texts))):
          text = self.texts[i]
          tag = self.tags[i]
          feature = self.old_features[i]
          tags, tokens, features = align_label(text,tag,feature, self.tokenizer)
          self.labels.append(tags)
          self.tokens.append(tokens)
          self.features.append(features)
          
        self.features = np.array(self.features,float)
        self.inputs = self.tokenizer(self.texts, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt')
        self.input_ids = self.inputs['input_ids']
        self.attention_masks = self.inputs['attention_mask']

    def __getitem__(self, idx):
        return self.input_ids[idx,:], self.attention_masks[idx,:], self.tokens[idx], torch.tensor(self.features[idx],dtype=torch.float32), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.input_ids)


# Define the model architecture and load the weights
class BertNerModel(nn.Module):
    def __init__(self,num_labels):
        super(BertNerModel,self).__init__()

        self.bert = BertModel.from_pretrained(weight)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768+25,num_labels)

    def forward(self,input_ids,attention_mask,extra_features,token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        pooled_output = outputs[0]
        bert_outputs = self.dropout(pooled_output)
        
        outputs = torch.concat((bert_outputs,extra_features[:,:,:]),-1)
        # outputs = bert_outputs
        outputs = self.classifier(outputs)
        
        return outputs
    

# Convert the tag into keyphrases
def TagConvert(raw_tags, words_set, poss=None):
    true_tags = []
    for i in range(raw_tags.shape[0]):
      kw_list = []
      nkw_list = ""
      for j in range(len(raw_tags[i])):
          item = raw_tags[i][j]
          if item == 0:
              continue
          if poss !=None and j in poss[i]:
              continue
          # if item == 5:
          #     continue
          if item == 4:
              kw_list.append(str(words_set[j][i]))
          if item == 1:
              nkw_list += str(words_set[j][i])
          if item == 2:
              nkw_list += " "
              nkw_list += str(words_set[j][i])
          if item == 3:
              nkw_list += " "
              nkw_list += str(words_set[j][i])
              kw_list.append(nkw_list)
              nkw_list = ""

      true_tags.append(kw_list)
    return true_tags


# Main Function
def BERT():

    # Load the data
    train_sens, train_word_nums, train_tags, train_features = load_traindata(train_path)
    test_sens, test_word_nums, test_tags, test_features = load_traindata(test_path)

    # Define the model architecture and load the weights
    tokenizer = BertTokenizerFast.from_pretrained(weight)

    # Biuld the train data set
    train_dataset = MyDataset(train_sens, train_features, train_tags, tokenizer)
    train_dataset.encode()

    # Biuld the test data set
    test_dataset = MyDataset(test_sens, test_features, test_tags)
    test_dataset.encode()

    # Biuld the data loader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=128)

    # Start training
    model = BertNerModel(num_labels=6)
    model = model.to(device)

    # Define the optimizer
    optim = AdamW(model.parameters(),lr=5e-5,weight_decay=1e-2)
    loss_fn = CrossEntropyLoss(reduction='none', ignore_index=0)
    loss_fn = loss_fn.to(device)

    # Start training epoch
    best_f1 = 0.0
    for epoch in tqdm(range(epochs)):
        loss_value = 0.0
        model.train()
        label_true, label_pred = [], []
        for i,batch in enumerate(train_dataloader):
            optim.zero_grad()
            input_ids, attention_masks, _, features, tags = batch
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))

            loss = loss_fn(pred_tags.permute(0,2,1),tags.to(device))
            loss = loss.mean()
            loss.backward()
            optim.step()

            pred_tags = F.softmax(pred_tags,dim=-1)
            pred_tags = torch.argmax(pred_tags,dim=-1)

            y_pred, y_true = to_array(pred_tags, tags)
            label_true.extend(y_true)
            label_pred.extend(y_pred)
        
            loss_value += loss.item()

        label_train_f1 = f1_score(label_true, label_pred, average='macro')

        model.eval()
        kw_true, kw_pred = [], []
        label_true, label_pred = [],[]
        for i,batch in enumerate(test_dataloader):
            input_ids, attention_masks, tokens, features, tags = batch
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0
                        module.train(False)
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
            pred_tags = F.softmax(pred_tags,dim=-1)
            pred_tags = torch.argmax(pred_tags,dim=-1)

            y_pred, y_true = to_array(pred_tags, tags)
            label_true.extend(y_true)
            label_pred.extend(y_pred)

            # more balance evaluate
            poss = []
            for i in range(len(tags)):
                pos = []
                for j in range(len(tags[i])):
                    if tags[i][j] == 0:
                        pos.append(j)
                poss.append(pos)
                
            kw_true.extend(TagConvert(tags,tokens))
            kw_pred.extend(TagConvert(pred_tags,tokens,poss))

        label_f1 = f1_score(label_true, label_pred, average='macro')
        train_P, train_R, train_F1 = evaluate(kw_true, kw_pred)
        
        if train_F1 > best_f1:
            best_f1 = train_F1
            best_epoch = epoch
            torch.save(model.state_dict(),'./pretrain_pt/bert.pt')
            
        print("epoch{}:  loss:{:.2f}   train_f1_value:{:.2f}  test_f1_value:{:.2f}  kw_f1_value:{:.2f}".format(
            epoch+1, loss_value / len(train_dataloader), label_train_f1, label_f1, train_F1
        ))


    # Start testing
    model = BertNerModel(num_labels=6)
    model.load_state_dict(torch.load('./pretrain_pt/bert.pt'))
    model = model.to(device)

    model.eval()
    kw_true, kw_pred = [], []
    label_true, label_pred = [],[]
    for i,batch in enumerate(test_dataloader):
        input_ids, attention_masks, tokens, tags = batch
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0
                    module.train(False)
            pred_tags = model(input_ids.to(device), attention_masks.to(device))
            pred_tags = F.softmax(pred_tags,dim=-1)
            pred_tags = torch.argmax(pred_tags,dim=-1)

        y_pred, y_true = to_array(pred_tags, tags)
        label_true.extend(y_true)
        label_pred.extend(y_pred)

        # more balance evaluate
        poss = []
        for i in range(len(tags)):
            pos = []
            for j in range(len(tags[i])):
                if tags[i][j] == 0:
                    pos.append(j)
            poss.append(pos)
            
        kw_true.extend(TagConvert(tags,tokens))
        kw_pred.extend(TagConvert(pred_tags,tokens,poss))

    label_f1 = f1_score(label_true, label_pred, average='macro')
    test_P, test_R, test_F1 = evaluate(kw_true, kw_pred)

    return test_P, test_R, test_F1, best_epoch

