# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''This python file is mainly used for hyperparameter setting. As shown in the following parameter items, it mainly includes:
   1* Mapping Features
   2* Data Path
   3* Training Parameters
   4* Models Parameters
'''
import torch
# Mapping
tag2ids = {'[PAD]': 0,'B': 1, 'I': 2, 'E': 3,'S': 4,"O": 5}
num_tag=6
id2tags = {val: key for key, val in tag2ids.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data path
# train_path = './dataset/ET-train.json'
# test_path = './dataset/ET-test.json'
# vocab_path = './dataset/ET-vocab.json'
train_path = './dataset/GT-train.json'
test_path = './dataset/GT-test.json'
vocab_path = './dataset/GT-vocab.json'
save_path = 'result/General-Twitter/'
glove_path = './datas/glove.6B.100d.txt'
word2vecFile = './datas/word2vec.6B.100d.txt'

# Train
device = "cuda"
paddingkey = "<pad>"
run_times = 5
epochs = 20
fs_num = 2
fs_name =  "EEG56"
embed_dim = 128
hidden_dim = 256
batch_size = 64
max_length = 64
vocab_size = 85535                           # or 37347
dropout_value = 0.5

# Models
## Choos the Model
'''
    Modify the value of model_type to select a model
    1: BiLSTM,    2: BiLSTM+CRF,    3:att-BiLSTM,    4: satt-BiLSTM, 
    5: att-BiLSTM+CRF,     6: satt-BiLSTM+CRF,       7: satt-BiLSTM+CRF+GloVe
'''
model_type = 1

lr = 0.001
num_layers = 1
weight_decay = 1e-6
factor = 0.5
patience = 3


