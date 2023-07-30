# -*- coding: utf-8 -*-
# @Time    : 2023/7/30 11:25
# @Author  : Xinyi Yan
'''
   This file is mainly used to calculate the P, R, and F1 values of AKE in terms of keyphrases.
      def read_json_datas(path): mainly for reading json file.
      def evaluate(predict_data, target_data, topk=2): mainly for calculating the P, R and F1 values of the extraction datas.
'''

import os
import json

# Read json file
def read_json_datas(path):
    datas = []
    with open(path, "r", encoding="utf-8") as fp:
        for i in fp.readlines():
            datas.append(json.loads(i))
    return datas



# Calculate the P, R and F1 values of the extraction datas
def evaluate(predict_data, target_data, topk=3):

    TRUE_COUNT, PRED_COUNT, GOLD_COUNT  =  0.0, 0.0, 0.0
    for index, words in enumerate(predict_data):
        y_pred, y_true = None, target_data[index]

        if type(predict_data) == str:
            words = sorted(words.items(), key=lambda item: (-item[1], item[0]))
            y_pred = [i[0] for i in words]
        elif type(predict_data) == list:
            y_pred = words

        y_pred = y_pred[0: topk]
        TRUE_NUM = len(set(y_pred) & set(y_true))
        TRUE_COUNT += TRUE_NUM
        PRED_COUNT += len(y_pred)
        GOLD_COUNT += len(y_true)
    # compute P
    if PRED_COUNT != 0:
        p = (TRUE_COUNT / PRED_COUNT)
    else:
        p = 0
    # compute R
    if GOLD_COUNT != 0:
        r = (TRUE_COUNT / GOLD_COUNT)
    else:
        r = 0
    # compute F1
    if (r + p) != 0:
        f1 = ((2 * r * p) / (r + p))
    else:
        f1 = 0

    p = round(p * 100, 2)
    r = round(r * 100, 2)
    f1 = round(f1 * 100, 2)

    return p, r, f1


