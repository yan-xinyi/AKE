# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''
   This python file mainly contains some functions to handle the prediction results data and the real data:
       class TextDataSet(Dataset): mainly for building the dictionary for vovabulary and cognitive signals.
       def get_outputs(feats): To get the position in a vector where the maximum value is located, we use this function. We can use the softmax function to calculate this and can get the regularised value of each element.
       def get_model(model_type): This function is mainly used to facilitate the user's selection of the model to be trained.
       def get_words(inputs, Loader): This function is mainly used to get all the words from the word lexicon into words_set.
       def get_gold(true_tags, targets, words_set): This function is mainly used to get the true values of key phrases from the original tags.
       def get_pred(out_list, prediction, words_set): This function is mainly used to get the prediction result of key phrases from the predicted tag sequence.
'''
import json
import torch
from models.BILSTM import bl
from models.BILSTMCRF import blcrf
from models.ATT_BILSTM import att_bl
from models.SATT_BILSTM import satt_bl
from models.ATT_BILSTMCRF import att_blcrf
from models.SATT_BILSTMCRF import satt_blcrf
from models.SATT_BILSTMCRF_GloVe import satt_blcrf_glove
from models.BERT import BERT
from models.T5 import T5
from torch.utils.data import Dataset


# build the dictionary for vovabulary and cognitive signals.
class TextDataSet(Dataset):

    def __init__(self, data_path, vocab_path, max_length, tag2ids):
        # read the dictionary
        self.word2ids = {word: i for i, word in enumerate(json.load(open(vocab_path, 'r', encoding='utf-8')))}
        self.id2words = {val: key for key, val in self.word2ids.items()}
        self.tag2ids = tag2ids
        self.ml = max_length

        # read the data
        self.datas = list(json.load(open(data_path, 'r', encoding='utf-8')).values())

    def __getitem__(self, item):
        text = self.datas[item]
        word_to_ids, tag_to_ids = [], []
        eeg_list, et_list = [], []
        i, text_len = 0, len(text)
        attention_mask = []
        while i < self.ml:
            if i < text_len:
                word = text[i]
                word_id = self.word2ids['[UNK]']
                if word[0] in self.word2ids.keys():
                    word_id = self.word2ids[word[0]]
                tag_id = self.tag2ids[word[-1]]
                et = list(map(float, word[1: 18]))
                eeg = list(map(float, word[18: -1]))

                word_to_ids.append(word_id)
                tag_to_ids.append(tag_id)
                et_list.append(et)
                eeg_list.append(eeg)
                attention_mask.append(1)
            else:
                word_to_ids.append(self.word2ids['[PAD]'])
                tag_to_ids.append(self.tag2ids['[PAD]'])
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


# Use this function to get the regularised values of each element.
def get_outputs(feats):
    outputs = torch.argmax(torch.softmax(feats,dim=-1),dim=-1)
    return outputs


# This function is mainly used to facilitate the user's selection of the model to be trained.
def get_model(model_type):

    model_name = object
    if model_type == 1:
        model_name = bl

    if model_type == 2:
        model_name = blcrf

    if model_type == 3:
        model_name = att_bl

    if model_type == 4:
        model_name = satt_bl

    if model_type == 5:
        model_name = att_blcrf

    if model_type == 6:
        model_name = satt_blcrf

    if model_type == 7:
        model_name = satt_blcrf_glove

    if model_type == 8:
        model_name = BERT

    if model_type == 9:
        model_name = T5

    return model_name


# This function is mainly used to get all the words from the word lexicon into words_set.
def get_words(inputs, Loader):
    words_set = []
    for item in inputs['input_ids'].tolist():
        word_list = []
        for id in item:
            if id in Loader.dataset.id2words.keys():
                word = Loader.dataset.id2words[id]
                if word != "[PAD]":
                    word_list.append(word)
                else:
                    continue;
        words_set.append(word_list)
    return words_set


# This function is mainly used to get the true values of key phrases from the original tags.
def get_gold(true_tags, targets, words_set):
    for i in range(len(true_tags)):
        kw_list = []
        nkw_list = ""
        for j in range(len(true_tags[i])):
            item = true_tags[i][j]
            if item == 5:
                continue;
            if item == 4:
                kw_list.append(str(words_set[i][j]))
            if item == 1:
                nkw_list += str(words_set[i][j])
            if item == 2:
                nkw_list += " "
                nkw_list += str(words_set[i][j])
            if item == 3:
                nkw_list += " "
                nkw_list += str(words_set[i][j])
                kw_list.append(nkw_list)
                nkw_list = ""
        targets.append(kw_list)
    return targets


# This function is mainly used to get the prediction result of key phrases from the predicted tag sequence.
def get_pred(out_list, prediction, words_set):
    for i in range(len(out_list)):
        kw_list = []
        nkw_list = ""
        for j in range(len(out_list[i])):
            item = out_list[i][j]
            if item == 5:
                continue;
            if item == 4:
                kw_list.append(str(words_set[i][j]))
            if item == 1:
                nkw_list += str(words_set[i][j])
            if item == 2:
                nkw_list += " "
                nkw_list += str(words_set[i][j])
            if item == 3:
                nkw_list += " "
                nkw_list += str(words_set[i][j])
                kw_list.append(nkw_list)
                nkw_list = ""
        prediction.append(kw_list)
    return prediction
