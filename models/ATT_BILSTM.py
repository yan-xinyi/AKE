 # -*- coding: utf-8 -*-
# @Time    : 2022/3/8 17:02
# @Author  : leizhao150
import numpy as np

import logging
from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from config import *
from evaluate import *
from torch.utils.data import Dataset, DataLoader

torch.cuda.set_device(0)
param_path = "./result/tf_param.pkl"

class TextDataSet(Dataset):

    def __init__(self, data_path, vocab_path):
        # read the dictionary
        self.word2ids = {word: i for i, word in enumerate(json.load(open(vocab_path, 'r', encoding='utf-8')))}
        self.id2words = {val: key for key, val in self.word2ids.items()}

        # read the data
        self.datas = list(json.load(open(data_path, 'r', encoding='utf-8')).values())

    def __getitem__(self, item):
        text = self.datas[item]
        word_to_ids, tag_to_ids = [], []
        eeg_list, et_list = [], []
        i, text_len = 0, len(text)
        attention_mask = []
        while i < max_length:
            if i < text_len:
                word = text[i]
                word_id = self.word2ids['[UNK]']
                if word[0] in self.word2ids.keys():
                    word_id = self.word2ids[word[0]]
                tag_id = tag2ids[word[-1]]
                et = list(map(float, word[1: 18]))
                eeg = list(map(float, word[18: -1]))

                word_to_ids.append(word_id)
                tag_to_ids.append(tag_id)
                et_list.append(et)
                eeg_list.append(eeg)
                attention_mask.append(1)
            else:
                word_to_ids.append(self.word2ids['[PAD]'])
                tag_to_ids.append(tag2ids['[PAD]'])
                et_list.append([0.0] * 17)
                eeg_list.append([0.0] * 8)
                attention_mask.append(0)
            i += 1
        return {
            "input_ids": torch.tensor(word_to_ids).long().to(device),
            "tags": torch.tensor(tag_to_ids).long().to(device),
            "et": torch.tensor(et_list).float().to(device),
            "eeg": torch.tensor(eeg_list).float().to(device),
            "attention_mask": torch.tensor(attention_mask).byte().to(device),
            "text_len": text_len
        }

    def __len__(self):
        return len(self.datas)



class ATTBiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim,
                 hidden_dim, num_layers=1, num_tags=6):
        super(ATTBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=embed_dim + fs_num,
                              hidden_size=hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=2 * hidden_dim)
        self.tanh = nn.Tanh()
        self.lstm_dropout = nn.Dropout(p=dropout_value)
        self.linear_dropout = nn.Dropout(p=dropout_value)

        self.att_weight = nn.Parameter(torch.randn(1, 2*hidden_dim, 1))
        self.dense = nn.Linear(
            in_features=2*hidden_dim,
            out_features=num_tags,
            bias=True
        )
        self.fc = nn.Linear(2 * hidden_dim, num_tags)

        # initialize weight
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)           # B*H*1
        #print(self.tanh(h).shape)                                           # torch.Size([256, 64, 512])
        #print(att_weight.shape)                                             # torch.Size([256, 256, 1])
        att_score = torch.bmm(self.tanh(h), att_weight)                      # B*L*H  *  B*H*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)  # B*L*1
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))         # B*L*1
        att_weight = F.softmax(att_score, dim=1)                             # B*L*1
        # print(att_weight.shape)

        #reps = torch.bmm(h, att_weight)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = h * att_weight                                                # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)                                               # B*H .transpose(1, 2)  .squeeze(dim=-1)
        return reps

    def forward(self, inputs, is_training=True):
        # word embedding
        input = self.embedding(inputs['input_ids'])

        # You can add features like ET and EEG here
        input = torch.cat([input, inputs['eeg'][:,:,4:6]], dim=-1)

        mask = inputs['attention_mask']
        input, _ = self.bilstm(input)                                        # bilstm layer
        input = self.linear_dropout(torch.relu(self.layernorm(input)))
        reps = self.attention_layer(input, mask)                             # soft attention layer
        output = self.fc(reps)                                               # linear layer

        return output


def att_bl():

    trainLoader = DataLoader(dataset=TextDataSet(train_path, vocab_path), batch_size=batch_size)
    testLoader = DataLoader(TextDataSet(test_path, vocab_path), batch_size=batch_size)

    model = ATTBiLSTM(vocab_size=vocab_size,
                       embed_dim=embed_dim,
                       hidden_dim=hidden_dim).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    best_P ,best_R ,best_F ,best_epoch= 0.0, 0.0, 0.0, 0
    for epoch in range(epochs):
        print("epoch "+ str(epoch + 1) + " is starting!")
        model.train()
        avg_loss = []
        with tqdm(trainLoader) as pbar_train:
            for inputs in pbar_train:
                pred = model(inputs)
                target = inputs['tags']
                optimizer.zero_grad()
                criterion = nn.CrossEntropyLoss()
                loss = criterion(pred.reshape(-1, 6), target.reshape(-1))
                loss.backward()
                optimizer.step()
                pbar_train.set_description('loss: %.3f'%loss.item())
                avg_loss.append(loss.item())
        print("avg_loss: %.2f"%np.average(avg_loss))

        # Test the model
        model.eval()
        y_true, y_pred = [], []

        targets = []
        prediction = []
        with tqdm(testLoader) as pbar_test:
            for inputs in pbar_test:
                logit = model(inputs, is_training=False)
                output = get_outputs(logit)
                outputs = output.tolist()

                out_list = []
                for txt in outputs:
                    txt_list = []
                    for item in txt:
                        if item != 0:
                            txt_list.append(item)
                    out_list.append(txt_list)

                true_tags = [tag[:inputs['text_len'][i]] for i, tag in enumerate(inputs['tags'].cuda().tolist())]
                y_true.extend([i for item in true_tags for i in item])
                y_pred.extend([i for output in out_list for i in output])

                # Get id2word to words_set
                words_set = get_words(inputs, testLoader)
                # Get ground truth keyphrase from true_tags to targets
                targets = get_gold(true_tags, targets, words_set)
                # Get predicted keyphrase from out_list to prediction
                prediction = get_pred(out_list, prediction, words_set)



        # calculate the P, R, F1 value of the test data
        P, R, F = evaluate(prediction, targets)
        if F > best_F:
            best_P = P
            best_R = R
            best_F = F
            best_epoch = epoch
        P_str = "P:" + "\t" + str(P)
        R_str = "R:" + "\t" + str(R)
        F_str = "F:" + "\t" + str(F)

        logging.info("Epoch" + str(epoch + 1) + "'s Evaluation:")
        logging.info("P is :" + P_str)
        logging.info("R is :" + R_str)
        logging.info("F is :" + F_str)


    return best_P, best_R, best_F, best_epoch