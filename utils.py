import torch
from models.BILSTM import bl
from models.BILSTMCRF import blcrf
from models.ATT_BILSTM import att_bl
from models.SATT_BILSTM import satt_bl
from models.ATT_BILSTMCRF import att_blcrf
from models.SATT_BILSTMCRF import satt_blcrf
from models.SATT_BILSTMCRF_GloVe import satt_blcrf_glove

def get_outputs(feats):
    outputs = torch.argmax(torch.softmax(feats,dim=-1),dim=-1)
    return outputs


def get_model(model_type):
    "This function is mainly used to facilitate the user's selection of the model to be trained."
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


def get_words(inputs, testLoader):
    words_set = []
    for item in inputs['input_ids'].tolist():
        word_list = []
        for id in item:
            if id in testLoader.dataset.id2words.keys():
                word = testLoader.dataset.id2words[id]
                if word != "[PAD]":
                    word_list.append(word)
                else:
                    continue;
        words_set.append(word_list)
    return words_set


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