# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan

'''
   This python file mainly contains some functions to handle the prediction results data and the real data:
       def get_outputs(feats): To get the position in a vector where the maximum value is located, we use this function. We can use the softmax function to calculate this and can get the regularised value of each element.
       def get_model(model_type): This function is mainly used to facilitate the user's selection of the model to be trained.
       def get_words(inputs, Loader): This function is mainly used to get all the words from the word lexicon into words_set.
       def get_gold(true_tags, targets, words_set): This function is mainly used to get the true values of key phrases from the original tags.
       def get_pred(out_list, prediction, words_set): This function is mainly used to get the prediction result of key phrases from the predicted tag sequence.
'''

import torch
from models.BILSTM import bl
from models.BILSTMCRF import blcrf
from models.ATT_BILSTM import att_bl
from models.SATT_BILSTM import satt_bl
from models.ATT_BILSTMCRF import att_blcrf
from models.SATT_BILSTMCRF import satt_blcrf
from models.SATT_BILSTMCRF_GloVe import satt_blcrf_glove


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
