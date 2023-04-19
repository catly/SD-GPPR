# -*- coding: utf-8 -*-



class BaseConfig():
    def __init__(self):
        self.train_data_path = "./dataset/train.json"
        self.test_data_path = "./dataset/test.json"
        self.dev_data_path = "./dataset/dev.json"
        self.words_data_path = "./dataset/words.json"
        self.label2idx_path = "./dataset/label2idx.json"
        self.word2idx_path = "./dataset/word2idx.json"
        self.vocab_size = 6500+2
        self.max_seq_len = 128             #100

baseconfig = BaseConfig()