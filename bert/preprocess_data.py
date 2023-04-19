# -*- coding: utf-8 -*-


import os
from collections import Counter
from utils import *
from config.base_config import BaseConfig

base_config = BaseConfig()
train_data_path = base_config.train_data_path


def preprocess_data():
    if not os.path.exists("./model"):
        os.mkdir("./model")
    if not os.path.exists("./log"):
        os.mkdir("./log")

    label2idx = {}
    words = []
    with open(train_data_path) as f:
        for line in f:
            line = json.loads(line)
            label = line["label"]
            text = line["sentence"]
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            words.extend(text.split(' '))
    words_counter = Counter(words).most_common(base_config.vocab_size - 2)
    words = ["<PAD>", "<UNK>"] + [w for w, c in words_counter]
    word_map = {word: idx for idx, word in enumerate(words)}
    dump_json(word_map, base_config.word2idx_path)
    dump_json(label2idx, base_config.label2idx_path)

    # 添加word异构节点到数据集
    # label2idx = {}
    # words = []
    # with open(train_data_path) as f:
    #     for line in f:
    #         line = json.loads(line)
    #         label = line["label"]
    #         text = line["sentence"]
    #         if label not in label2idx:
    #             label2idx[label] = len(label2idx)
    #         words.extend(text.split(' '))
    # # 添加word标签
    # label2idx['WORD'] = len(label2idx)
    # words_counter = Counter(words).most_common(base_config.vocab_size - 2)
    # words = ["<PAD>", "<UNK>"] + [w for w, c in words_counter]
    # word_map = {word: idx for idx, word in enumerate(words)}
    # dump_json(word_map, base_config.word2idx_path)
    # dump_json(label2idx, base_config.label2idx_path)
    # # 将word追加到训练集
    # with open(base_config.train_data_path, 'a', encoding='utf-8') as f:
    #     for word in words:
    #         word_node = {"sentence": word, "label": 'WORD'}
    #         json.dump(word_node, f, ensure_ascii=False)
    #         f.write('\n')


if __name__ == '__main__':
    preprocess_data()
