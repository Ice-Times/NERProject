# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import numpy as np
from collections import Counter
from itertools import accumulate
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib as mpl

Path = "./NER/data"
CORPUS_PATH = "%s/train.txt" % Path

KERAS_MODEL_SAVE_PATH = '%s/Bi-LSTM-Model.h5' % Path    #模型
WORD_DICTIONARY_PATH = '%s/word_dictionary.pk' % Path   #字典
InVERSE_WORD_DICTIONARY_PATH = '%s/inverse_word_dictionary.pk' % Path
LABEL_DICTIONARY_PATH = '%s/label_dictionary.pk' % Path
OUTPUT_DICTIONARY_PATH = '%s/output_dictionary.pk' % Path

CONSTANTS = [
             KERAS_MODEL_SAVE_PATH,
             InVERSE_WORD_DICTIONARY_PATH,
             WORD_DICTIONARY_PATH,
             LABEL_DICTIONARY_PATH,
             OUTPUT_DICTIONARY_PATH
             ]

# 加载train.txt中数据
def load_data():
    with open(CORPUS_PATH, 'r') as f:
        text_data = [text.strip() for text in f.readlines()]
    text_data = [text_data[k].split('\t') for k in range(0, len(text_data))]
    index = range(0, len(text_data), 3)
    # print(index)
    # print(len(index))
    input_data = list()
    for i in range(1, len(index)):
        rows = text_data[index[i-1]:index[i]]
        # if i==14000:
        #     print(i)
        #     print(rows)
        sentence_no = np.array([i]*len(rows[0]), dtype=str)
        rows.append(sentence_no)
        rows = np.array(rows).T
        input_data.append(rows)

    rows = text_data[len(text_data)-3:len(text_data)]
    # print("最后一个训练集：")
    # print("1:", rows[0])
    # print("2:", rows[1])
    # print("3:", rows[2])
    # print("1:",len(rows[0]))
    # print("2:", len(rows[1]))
    # print("3:", len(rows[2]))
    sentence_no = np.array([i] * len(rows[0]), dtype=str)
    rows.append(sentence_no)
    rows = np.array(rows).T
    input_data.append(rows)



    input_data = pd.DataFrame(np.concatenate([item for item in input_data]),columns=['word', 'pos', 'tag', 'sent_no'])
   # print(input_data)
    return input_data

# 数据处理
def data_processing():
    input_data = load_data()
    labels, vocabulary = list(input_data['tag'].unique()), list(input_data['word'].unique())

    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i+1 for i, label in enumerate(labels)}
    output_dictionary = {i+1: labels for i, labels in enumerate(labels)}

    dict_list = [word_dictionary, inverse_word_dictionary,label_dictionary, output_dictionary]
    # 保存结果
    for dict_item, path in zip(dict_list, CONSTANTS[1:]):
        with open(path, 'wb') as f:
            pickle.dump(dict_item, f)

def showDatas():
    input_data = load_data()
    with open(CORPUS_PATH, 'r') as f:
        text_data = [text.strip() for text in f.readlines()]
    text_data = [text_data[k].split('\t') for k in range(0, len(text_data))]




    labels, vocabulary = list(input_data['tag'].unique()), list(input_data['word'].unique())
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    print(len(text_data))
    print(word_dictionary)
    llist=[]
    llist.append(len(text_data)/3)
    llist.append(len(word_dictionary))
    return  llist
# data_processing()
#showDatas()
