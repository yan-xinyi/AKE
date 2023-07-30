import re
import os
import nltk
import json
import codecs
from tqdm import tqdm
import pandas as pd
from nltk.tag import pos_tag

'''
   This file is mainly used to extract cognitive signals from the Zuco dataset and merge them into the AKE dataset. 
   To run this file, please run main() directly.
'''

# Replace every digit in a string by a zero.
def zero_digits(s):
    return re.sub('\d', '0', s)


# Transfer IOB -> IOBES
def iob_iobes(tags):

    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


# Loading sentences from the Twitter corpus by paragraph
def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            # for i in range(8):
            #     word.pop(18)
            #     i += 1
            word[0] = zero_digits(word[0]) if zeros else word[0]
            # assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# If the tag_scheme does not match the "iboes" format, check and update it.
def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        # if not iob2(tags):
        #     s_str = '\n'.join(' '.join(w) for w in s)
        #     raise Exception('Sentences should be given in IOB format! ' +
        #                     'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


# Compute Eye-tracking ,EEG feature values
def feature_values(sentences):

    all_list = []
    for sentence in sentences:
        for words in sentence:
            all_list.append(words)

    all_word_list, word_list = [], []
    num_list = []
    out_list = []

    for words in all_list:
        all_word_list.append(words[0])
    count = Counter(all_word_list)

    for words in all_list:
        if words[0] not in word_list:
            word_list.append(words[0])
            out_list.append(words)
            num_list.append("1")
        else:
            for item in out_list:
                index = 0
                if item[0] == words[0]:
                    item[1] = int(item[1]) + int(words[1])
                    item[2] = int(item[2]) + int(words[2])
                    item[3] = int(item[3]) + int(words[3])
                    item[4] = int(item[4]) + int(words[4])
                    item[5] = int(item[5]) + int(words[5])
                    item[6] = int(item[6]) + int(words[6])
                    item[7] = int(item[7]) + int(words[7])
                    item[8] = int(item[8]) + int(words[8])
                    item[9] = int(item[9]) + int(words[9])
                    item[10] = int(item[10]) + int(words[10])
                    item[11] = int(item[11]) + int(words[11])
                    item[12] = int(item[12]) + int(words[12])
                    item[13] = int(item[13]) + int(words[13])
                    item[14] = int(item[14]) + int(words[14])
                    item[15] = int(item[15]) + int(words[15])
                    item[16] = int(item[16]) + int(words[16])
                    item[17] = int(item[17]) + int(words[17])
                    item[18] = int(item[18]) + int(words[18])
                    item[19] = int(item[19]) + int(words[19])
                    item[20] = int(item[20]) + int(words[20])
                    item[21] = int(item[21]) + int(words[21])
                    item[22] = int(item[22]) + int(words[22])
                    item[23] = int(item[23]) + int(words[23])
                    item[24] = int(item[24]) + int(words[24])
                    item[25] = int(item[25]) + int(words[25])
                    num_list[index] = int(num_list[index]) + 1
                else:
                    index += 1
    #print(out_list)
    for item in out_list:
        item[1] = int(item[1]) / int(count[item[0]])
        item[2] = int(item[2]) / int(count[item[0]])
        item[3] = int(item[3]) / int(count[item[0]])
        item[4] = int(item[4]) / int(count[item[0]])
        item[5] = int(item[5]) / int(count[item[0]])
        item[6] = int(item[6]) / int(count[item[0]])
        item[7] = int(item[7]) / int(count[item[0]])
        item[8] = int(item[8]) / int(count[item[0]])
        item[9] = int(item[9]) / int(count[item[0]])
        item[10] = int(item[10]) / int(count[item[0]])
        item[11] = int(item[11]) / int(count[item[0]])
        item[12] = int(item[12]) / int(count[item[0]])
        item[13] = int(item[13]) / int(count[item[0]])
        item[14] = int(item[14]) / int(count[item[0]])
        item[15] = int(item[15]) / int(count[item[0]])
        item[16] = int(item[16]) / int(count[item[0]])
        item[17] = int(item[17]) / int(count[item[0]])
        item[18] = int(item[18]) / int(count[item[0]])
        item[19] = int(item[19]) / int(count[item[0]])
        item[20] = int(item[20]) / int(count[item[0]])
        item[21] = int(item[21]) / int(count[item[0]])
        item[22] = int(item[22]) / int(count[item[0]])
        item[23] = int(item[23]) / int(count[item[0]])
        item[24] = int(item[24]) / int(count[item[0]])
        item[25] = int(item[25]) / int(count[item[0]])

    return out_list, word_list


# calculate the unknown words' value
def unfeature_values(all_value_list):

    sum_list = ["unknown",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"O"]
    num = 0
    for i in range(len(all_value_list)):
        for j in range(len(all_value_list[i])):
            sum_list[1] = float(sum_list[1]) + float(all_value_list[i][j][1])
            sum_list[2] = int(sum_list[2]) + int(all_value_list[i][j][2])
            sum_list[3] = int(sum_list[3]) + int(all_value_list[i][j][3])
            sum_list[4] = int(sum_list[4]) + int(all_value_list[i][j][4])
            sum_list[5] = int(sum_list[5]) + int(all_value_list[i][j][5])
            sum_list[6] = int(sum_list[6]) + int(all_value_list[i][j][6])
            sum_list[7] = int(sum_list[7]) + int(all_value_list[i][j][7])
            sum_list[8] = int(sum_list[8]) + int(all_value_list[i][j][8])
            sum_list[9] = int(sum_list[9]) + int(all_value_list[i][j][9])
            sum_list[10] = int(sum_list[10]) + int(all_value_list[i][j][10])
            sum_list[11] = int(sum_list[11]) + int(all_value_list[i][j][11])
            sum_list[12] = int(sum_list[12]) + int(all_value_list[i][j][12])
            sum_list[13] = int(sum_list[13]) + int(all_value_list[i][j][13])
            sum_list[14] = int(sum_list[14]) + int(all_value_list[i][j][14])
            sum_list[15] = int(sum_list[15]) + int(all_value_list[i][j][15])
            sum_list[16] = int(sum_list[16]) + int(all_value_list[i][j][16])
            sum_list[17] = int(sum_list[17]) + int(all_value_list[i][j][17])
            sum_list[18] = int(sum_list[18]) + int(all_value_list[i][j][18])
            sum_list[19] = int(sum_list[19]) + int(all_value_list[i][j][19])
            sum_list[19] = int(sum_list[20]) + int(all_value_list[i][j][20])
            sum_list[21] = int(sum_list[21]) + int(all_value_list[i][j][21])
            sum_list[22] = int(sum_list[22]) + int(all_value_list[i][j][22])
            sum_list[23] = int(sum_list[23]) + int(all_value_list[i][j][23])
            sum_list[24] = int(sum_list[24]) + int(all_value_list[i][j][24])
            sum_list[25] = int(sum_list[25]) + int(all_value_list[i][j][25])
            num += 1

    sum_list[1] = sum_list[1] / num
    sum_list[2] = sum_list[2] / num
    sum_list[3] = sum_list[3] / num
    sum_list[4] = sum_list[4] / num
    sum_list[5] = sum_list[5] / num
    sum_list[6] = sum_list[6] / num
    sum_list[7] = sum_list[7] / num
    sum_list[8] = sum_list[8] / num
    sum_list[9] = sum_list[9] / num
    sum_list[10] = sum_list[10] / num
    sum_list[11] = sum_list[11] / num
    sum_list[12] = sum_list[12] / num
    sum_list[13] = sum_list[13] / num
    sum_list[14] = sum_list[14] / num
    sum_list[15] = sum_list[15] / num
    sum_list[16] = sum_list[16] / num
    sum_list[17] = sum_list[17] / num
    sum_list[18] = sum_list[18] / num
    sum_list[19] = sum_list[19] / num
    sum_list[20] = sum_list[20] / num
    sum_list[21] = sum_list[21] / num
    sum_list[22] = sum_list[22] / num
    sum_list[23] = sum_list[23] / num
    sum_list[24] = sum_list[24] / num
    sum_list[25] = sum_list[25] / num

    return sum_list


# Adding cognitive features to the ake dataset
def add_values(sentences, all_value_list):

    for sentence in sentences:
        for words in sentence:
            for item in all_value_list:
                if words[0] == item[0]:
                    words[1] = item[1]
                    words[2] = item[2]
                    words[3] = item[3]
                    words[4] = item[4]
                    words[5] = item[5]
                    words[6] = item[6]
                    words[7] = item[7]
                    words[8] = item[8]
                    words[9] = item[9]
                    words[10] = item[10]
                    words[11] = item[11]
                    words[12] = item[12]
                    words[13] = item[13]
                    words[14] = item[14]
                    words[15] = item[15]
                    words[16] = item[16]
                    words[17] = item[17]
                    words[18] = item[18]
                    words[19] = item[19]
                    words[20] = item[20]
                    words[21] = item[21]
                    words[22] = item[22]
                    words[23] = item[23]
                    words[24] = item[24]
                    words[25] = item[25]

                if words[26] == "B-PER" or words[26] == "B-LOC" or words[26] == "B-ORG":
                    words[26] = "B"
                if words[26] == "I-PER" or words[26] == "I-LOC" or words[26] == "I-ORG":
                    words[26] = "I"
    return sentences


# Train NLTK pos tagging
def train_pos_tagger(datas):

    word_list = []
    for data in datas:
        for item in data:
            item.append("0")
            item[27] = item[26]
            item[26] = item[25]
            item[25] = item[24]
            item[24] = item[23]
            item[23] = item[22]
            item[22] = item[21]
            item[21] = item[20]
            item[20] = item[19]
            item[19] = item[18]
            item[18] = item[17]
            item[17] = item[16]
            item[16] = item[15]
            item[15] = item[14]
            item[14] = item[13]
            item[13] = item[12]
            item[12] = item[11]
            item[11] = item[10]
            item[10] = item[9]
            item[9] = item[8]
            item[8] = item[7]
            item[7] = item[6]
            item[6] = item[5]
            item[5] = item[4]
            item[4] = item[3]
            item[3] = item[2]
            item[2] = item[1]
            word_list.append(item[0])

    pos_lists = []
    pos_list = pos_tag(word_list)
    for word, pos in pos_list:
        pos_lists.append(pos)
    m = 0
    for data in datas:
        for item in data:
            item[1] = pos_lists[m]
            m += 1
    return datas


# Test NLTK pos tagging
def test_pos_tagger(datas):
    
    word_list = []
    for data in datas:
        for item in data:
            item.append("0")
            item[27] = item[26]
            item[26] = item[25]
            item[25] = item[24]
            item[24] = item[23]
            item[23] = item[22]
            item[22] = item[21]
            item[21] = item[20]
            item[20] = item[19]
            item[19] = item[18]
            item[18] = item[17]
            item[17] = item[16]
            item[16] = item[15]
            item[15] = item[14]
            item[14] = item[13]
            item[13] = item[12]
            item[12] = item[11]
            item[11] = item[10]
            item[10] = item[9]
            item[9] = item[8]
            item[8] = item[7]
            item[7] = item[6]
            item[6] = item[5]
            item[5] = item[4]
            item[4] = item[3]
            item[3] = item[2]
            item[2] = item[1]
            word_list.append(item[0])

    pos_lists = []
    pos_list = pos_tag(word_list)
    for word, pos in pos_list:
        pos_lists.append(pos)
    m = 0
    for data in datas:
        for item in data:
            item[1] = pos_lists[m]
            m += 1
    return datas


# NLTK pos tagging
def pos_tagger(datas):

    word_list = []
    for data in datas:
        for item in data:
            word_list.append(item[0])

    pos_lists = []
    pos_list = pos_tag(word_list)
    for word, pos in pos_list:
        pos_lists.append(pos)
    m = 0
    dataset =[]
    for data in datas:
        sentence = []
        for item in data:
            item_list = []
            item_list.append(item[0])
            item_list.append(pos_lists[m])
            if len(item) == 2:
                item_list.append(item[1])
            else:
                item_list.append("0")
            m += 1
            sentence.append(item_list)
        dataset.append(sentence)
    return dataset


# Discretisation of cognitive eigenvalues
def discretization(dataset):
    for sentence in dataset:
        max_item = []
        min_item = []
        # Calculate the minimum and maximum values of each feature in each sentence
        for i in range(25):
            max = 0.0
            min = 0.0
            for j in range(len(sentence)):
                if min > float(sentence[j][i+2]):
                    min = float(sentence[j][i+2])
                if max < float(sentence[j][i+2]):
                    max = float(sentence[j][i+2])
            min_item.append(min)
            max_item.append(max)


        for word in sentence:
            for i in range(25):
                m = word[i + 2] - min_item[i]
                n = max_item[i] - min_item[i]
                if n == 0:
                    word[i + 2] = 0
                else:
                    word[i + 2] = m / n
    return dataset


# Discretisation of unregistered word feature values
def undiscretization(unknown_value):

    max = 0.0
    min = 0.0
    # Calculate the minimum and maximum values of each feature in each sentence
    for i in range(25):
        if min > float(unknown_value[i + 1]):
            min = float(unknown_value[i + 1])
        if max < float(unknown_value[i + 1]):
            max = float(unknown_value[i + 1])

    for i in range(25):
        m = unknown_value[i + 1] - min
        n = max - min
        if n == 0:
            unknown_value[i + 1] = 0
        else:
            unknown_value[i + 1] = m / n
    return unknown_value


# Merge to get all datas
def merge_data(train_datas, test_datas):
    data = []
    for tr_sentences in train_datas:
        for tr_sentence in tr_sentences:
            if tr_sentence not in data:
                data.append(tr_sentence)
    for te_sentences in test_datas:
        for te_sentence in te_sentences:
            if te_sentence not in data:
                data.append(te_sentence)
    return data


# Transfrom value to string
def to_string(val):
    if type(val) == str:
        return val
    else:
        return str(float(val))
    

# Preprocess datas, add cognitive data into ake dataset
def process_datas(datas, mode, save_folder, zuco_data, word_list, unknown_list, features=None):

    # Data saving path
    save_folder = os.path.join(os.path.abspath(save_folder))
    if not os.path.exists(save_folder): os.mkdir(save_folder)

    if mode == 'train':
        save_str = []
        all_word_num = 0
        unknown_word_num = 0
        for data in datas:
            fs = []
            words, POS, lable = [], [], []
            if data[0] == None: continue
            for item in data:
                words.append(item[0])
                POS.append(item[1])
                if item[2] == "0":   # O
                    lable.append("O")
                if item[2] == "1":   # S
                    lable.append("S")
                if item[2] == "2":   # B
                    lable.append("B")
                if item[2] == "3":   # I
                    lable.append("I")
                if item[2] == "4":   # E
                    lable.append("E")
                all_word_num += 1

            ET1, ET2, ET3, ET4, ET5, ET6, ET7, ET8, ET9, ET10, ET11, ET12, ET13, ET14, ET15, ET16, ET17 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8 = [], [], [], [], [], [], [], []
            fs.append(words)

            for word in words:
                if word != "":
                    if word in word_list:
                        for zuco_word in zuco_data:
                            if word == zuco_word[0]:
                                if "ET" in features:
                                    ET1.append(float(zuco_word[2]))
                                    ET2.append(float(zuco_word[3]))
                                    ET3.append(float(zuco_word[4]))
                                    ET4.append(float(zuco_word[5]))
                                    ET5.append(float(zuco_word[6]))
                                    ET6.append(float(zuco_word[7]))
                                    ET7.append(float(zuco_word[8]))
                                    ET8.append(float(zuco_word[9]))
                                    ET9.append(float(zuco_word[10]))
                                    ET10.append(float(zuco_word[11]))
                                    ET11.append(float(zuco_word[12]))
                                    ET12.append(float(zuco_word[13]))
                                    ET13.append(float(zuco_word[14]))
                                    ET14.append(float(zuco_word[15]))
                                    ET15.append(float(zuco_word[16]))
                                    ET16.append(float(zuco_word[17]))
                                    ET17.append(float(zuco_word[18]))
                                if "EEG" in features:
                                    EEG1.append(float(zuco_word[19]))
                                    EEG2.append(float(zuco_word[20]))
                                    EEG3.append(float(zuco_word[21]))
                                    EEG4.append(float(zuco_word[22]))
                                    EEG5.append(float(zuco_word[23]))
                                    EEG6.append(float(zuco_word[24]))
                                    EEG7.append(float(zuco_word[25]))
                                    EEG8.append(float(zuco_word[26]))
                                break;
                            else:
                                continue;
                    else:
                        ET1.append(float(unknown_list[1]))
                        ET2.append(float(unknown_list[2]))
                        ET3.append(float(unknown_list[3]))
                        ET4.append(float(unknown_list[4]))
                        ET5.append(float(unknown_list[5]))
                        ET6.append(float(unknown_list[6]))
                        ET7.append(float(unknown_list[7]))
                        ET8.append(float(unknown_list[8]))
                        ET9.append(float(unknown_list[9]))
                        ET10.append(float(unknown_list[10]))
                        ET11.append(float(unknown_list[11]))
                        ET12.append(float(unknown_list[12]))
                        ET13.append(float(unknown_list[13]))
                        ET14.append(float(unknown_list[14]))
                        ET15.append(float(unknown_list[15]))
                        ET16.append(float(unknown_list[16]))
                        ET17.append(float(unknown_list[17]))
                        EEG1.append(float(unknown_list[18]))
                        EEG2.append(float(unknown_list[19]))
                        EEG3.append(float(unknown_list[20]))
                        EEG4.append(float(unknown_list[21]))
                        EEG5.append(float(unknown_list[22]))
                        EEG6.append(float(unknown_list[23]))
                        EEG7.append(float(unknown_list[24]))
                        EEG8.append(float(unknown_list[25]))
                        unknown_word_num += 1
                else:
                    continue;

            if "ET" in features:
                fs.append(ET1)
                fs.append(ET2)
                fs.append(ET3)
                fs.append(ET4)
                fs.append(ET5)
                fs.append(ET6)
                fs.append(ET7)
                fs.append(ET8)
                fs.append(ET9)
                fs.append(ET10)
                fs.append(ET11)
                fs.append(ET12)
                fs.append(ET13)
                fs.append(ET14)
                fs.append(ET15)
                fs.append(ET16)
                fs.append(ET17)
            if "EEG" in features:
                fs.append(EEG1)
                fs.append(EEG2)
                fs.append(EEG3)
                fs.append(EEG4)
                fs.append(EEG5)
                fs.append(EEG6)
                fs.append(EEG7)
                fs.append(EEG8)

            fs.append(lable)
            df = pd.DataFrame(fs).T.round(0)
            for index, jj in df.iterrows():
                save_str.append(" ".join([to_string(i) for i in jj.tolist()]))
            save_str.append("")

        print("The number of unknown train words are:", unknown_word_num)
        print("The number of all train words are:", all_word_num)

        with open(os.path.join(save_folder, mode), "w", encoding="utf-8") as fp:
            fp.write("\n".join(save_str).strip())
    return save_str


# Read the intermediate files and turn them into json file which model can cope with
def tojson(save_train_path, save_test_path, train_path, test_path, vocab_path):
    # Biuld the dictionary
    vocab = ['[PAD]', '[UNK]']
    train_list = []
    with open(save_train_path, 'r', encoding='utf-8') as fp:
        texts = fp.read().strip().split("\n\n")
        for index, text in enumerate(tqdm(texts)):
            temp = []
            items = text.split('\n')
            for item in items:
                item = item.strip().split(' ')
                temp.append(item)
                if item[0] not in vocab:
                    vocab.append(item[0])
            train_list.append((index, temp))
    json.dump(dict(train_list), open(train_path, 'w', encoding='utf-8'), ensure_ascii=False)

    test_list = []
    with open(save_test_path, 'r', encoding='utf-8') as fp:
        texts = fp.read().strip().split("\n\n")
        for index, text in enumerate(tqdm(texts)):
            temp = []
            items = text.split('\n')
            for item in items:
                item = item.strip().split(' ')
                temp.append(item)
                if item[0] not in vocab:

                    vocab.append(item[0])
            test_list.append((index, temp))
    json.dump(dict(test_list), open(test_path, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(vocab, open(vocab_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print(len(vocab))


# Mainstream process for dealing with cognitive features
def main():
    zuco_train_path = 'dataset/Zuco/train'                      # Raw file path to the ZUCO training set for the cognitive processing corpus
    zuco_test_path = 'dataset/Zuco/test'                        # Raw file path to the ZUCO testing set for the cognitive processing corpus
    tweet_train_path = 'datset/General-Twitter/train'           # Raw file path to the Tweets training set for the AKE corpus
    tweet_test_path = 'datset/General-Twitter/test'            # Raw file path to the Tweets testing set for the AKE corpus
    save_train_path = 'datset/General-Twitter/train_im'         # Path to store intermediate training files for adding cognitive features
    save_test_path = 'datset/General-Twitter/test_im'           # Path to store intermediate testing files for adding cognitive features
    train_path = 'datset/GT-train.json'                         # Storing the final AKE training corpus containing cognitive features
    test_path = 'datset/GT-test.json'                           # Storing the final AKE testing corpus containing cognitive features
    vocab_path = 'datset/GT-vocab.json'                         # Storing the final AKE vocab corpus

    # Load sentences
    all_sentences = []
    train_sentences = load_sentences(zuco_train_path, 0, 0)
    test_sentences = load_sentences(zuco_test_path, 0, 0)
    tweet_train_list = load_sentences(tweet_train_path, 0, 0)
    tweet_test_list = load_sentences(tweet_test_path, 0, 0)
    for sentence in train_sentences, test_sentences:
        for item in sentence:
            all_sentences.append(item)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, "iobes")
    update_tag_scheme(test_sentences, "iobes")

    # Compute Eye-tracking ,EEG feature values
    all_value_list, word_list = feature_values(all_sentences)

    # calculate the unknown words' value
    unknown_value_list = unfeature_values(all_sentences)
    print(unknown_value_list)

    # add the Eye-tracking ,EEG feature values in every sentences
    all_value_list = add_values(all_sentences, all_value_list)
    train_value_list = add_values(train_sentences, all_value_list)
    test_value_list = add_values(test_sentences, all_value_list)

    # NLTK POS labelling
    train_datas = train_pos_tagger(train_value_list)
    test_datas = test_pos_tagger(test_value_list)
    tweet_train = pos_tagger(tweet_train_list)
    tweet_test = pos_tagger(tweet_test_list)

    # Discretisation of cognitive Features
    train_datas = discretization(train_datas)
    test_datas = discretization(test_datas)
    unknown_list = undiscretization(unknown_value_list)

    # Merge to get all datas
    all_dataset = merge_data(train_datas, test_datas)

    # Store the intermediate files that store the added cognitive features to the corresponding paths
    process_datas(tweet_train, 'train', save_train_path, all_dataset, word_list, unknown_list, features=['ET', 'EEG'])
    process_datas(tweet_test, 'test', save_test_path, all_dataset, word_list, unknown_list, features=['ET', 'EEG'])

    # Read the intermediate files and turn them into json file which model can cope with
    tojson(save_train_path, save_test_path, train_path, test_path, vocab_path)


if __name__ == '__main__':
    main()
