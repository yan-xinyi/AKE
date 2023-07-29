import json
from tqdm import tqdm

# Create the dictionary
vocab = ['[PAD]', '[UNK]']
train_list = []
with open("./dataset/Election-Trec/train", 'r', encoding='utf-8') as fp:
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
json.dump(dict(train_list), open('./dataset/Election-Trec/train.json', 'w', encoding='utf-8'), ensure_ascii=False)

test_list = []
with open("./dataset/Election-Trec/test", 'r', encoding='utf-8') as fp:
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
json.dump(dict(test_list), open('./dataset/Election-Trec/test.json', 'w', encoding='utf-8'), ensure_ascii=False)
json.dump(vocab, open('./dataset/Election-Trec/vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(len(vocab))

