# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''
   This python file mainly contains code for training, testing and evaluation of AKE models for Bi-LSTM+CRF.
   The main function is blcrf(), called from main().
      1* Through the TextDataSet (Dataset): build the dictionary for vovabulary and cognitive signals.
      2* Build BiLSTM_CRF() model
      3* Start training and calculate the loss value.
      4* Conduct testing, again read the data first, build the model, in the prediction, evaluate its results.
'''

import numpy as np
import logging
from torch import nn, optim
from tqdm import tqdm
from ..config import *
from ..utils import *
from ..evaluate import *
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader


# Define BiLSTM_CRF() model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, num_tags=6):
        super(BiLSTM_CRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=embed_dim+fs_num,
                              hidden_size=hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=2 * hidden_dim)
        self.tocrf = nn.Linear(2 * hidden_dim, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.dropout  = nn.Dropout(p = dropout_value)



    def forward(self, inputs, is_training=True):

        # word embedding
        input = self.embedding(inputs['input_ids'])

        # You can add features like ET and EEG here
        input = torch.cat([input, inputs['eeg'][:,:,4:6]], dim=-1)
        
        input, _ = self.bilstm(input)                                         # bilstm layer
        input = self.dropout(torch.relu(self.layernorm(input)))               # layernorm layer
        crf_feats = self.tocrf(input)                                         # Linear layer

        if is_training:
            loss = self.crf(emissions=crf_feats,                              # CRF layer
                            tags=inputs['tags'],
                            mask=inputs['attention_mask'],
                            reduction='mean')
            return -loss
        else:
            outputs = self.crf.decode(emissions=crf_feats, mask=inputs['attention_mask'])
            tag_probs = torch.softmax(crf_feats, dim=-1)
            return tag_probs, outputs


# Main Function
def blcrf():

    # Load the data
    trainLoader = DataLoader(dataset=TextDataSet(train_path, vocab_path, max_length, tag2ids), batch_size=batch_size)
    testLoader = DataLoader(TextDataSet(test_path, vocab_path, max_length, tag2ids), batch_size=batch_size)

    # Define the model
    model = BiLSTM_CRF(vocab_size=vocab_size,
                       embed_dim=embed_dim,
                       hidden_dim=hidden_dim,).to(device)
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_P, best_R, best_F = 0.0, 0.0, 0.0
    best_epoch, best_loss = 0, 10
    for epoch in range(epochs):
        print("epoch " + str(epoch + 1) + " is starting!")

        # Train the model
        model.train()
        avg_loss = []
        with tqdm(trainLoader) as pbar_train:
            for inputs in pbar_train:
                loss = model(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar_train.set_description('loss: %.3f' % loss.item())
                avg_loss.append(loss.item())
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
        print("avg_loss: %.2f" % np.average(avg_loss))

        # Test the model
        model.eval()
        y_true, y_pred = [], []

        targets = []
        prediction = []
        with tqdm(testLoader) as pbar_test:
            for inputs in pbar_test:
                _, outputs = model(inputs, is_training=False)
                true_tags = [tag[:inputs['text_len'][i]] for i, tag in enumerate(inputs['tags'].cuda().tolist())]
                y_true.extend([i for item in true_tags for i in item])
                y_pred.extend([i for output in outputs for i in output])

                # Get id2word to words_set
                words_set = get_words(inputs, testLoader)
                # Get ground truth keyphrase from true_tags to targets
                targets = get_gold(true_tags, targets, words_set)
                # Get predicted keyphrase from out_list to prediction
                prediction = get_pred(outputs, prediction, words_set)


        # calculate the P, R, F1 value of the test data
        P, R, F = evaluate(prediction, targets)

        if F > best_F:
            best_P = P
            best_R = R
            best_F = F

        P_str = "P:" + "\t" + str(P)
        R_str = "R:" + "\t" + str(R)
        F_str = "F:" + "\t" + str(F)

        logging.info("Epoch" + str(epoch + 1) + "'s Evaluation:")
        logging.info("P is :" + P_str)
        logging.info("R is :" + R_str)
        logging.info("F is :" + F_str)


    return best_P, best_R, best_F, best_epoch
