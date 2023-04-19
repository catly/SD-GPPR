from torch.utils.data import DataLoader
from bert import Bert
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.bert_config import BertConfig
from utils import *
from collections import defaultdict
from collections import Counter, defaultdict
import numpy as np

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig()

def get_words_freq(text_list):
    """根据text_list获取词-词频"""
    dataset = []
    for text in text_list:
        dataset += text.split(' ')
    words_freq = Counter(dataset)
    return words_freq

def encode_fn(text_list):
    """将输入句子编码成BERT需要格式"""
    tokenizers = tokenizer.batch_encode_plus(
        text_list,
        padding=True,
        truncation=True,
        max_length=config.base_config.max_seq_len,
        return_tensors='pt',  # 返回的类型为pytorch tensor
        is_split_into_words=True
    )
    input_ids = tokenizers['input_ids']
    token_type_ids = tokenizers['token_type_ids']
    attention_mask = tokenizers['attention_mask']
    return input_ids, token_type_ids, attention_mask


class BertDataSet(Dataset):
    def __init__(self, data_path):
        texts, labels = [], []
        label2idx = config.label2idx
        with open(data_path) as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                labels.append(label2idx[line["label"]])
                texts.append(line["sentence"])
        self.labels = torch.tensor(labels)
        self.texts = texts
        self.input_ids, self.token_type_ids, self.attention_mask = encode_fn(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index], \
               self.texts[index]


class WordDataSet(Dataset):
    def __init__(self, data_path):
        texts, labels = [], []
        with open(data_path) as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                labels.append(line["label"])
                texts.append(line["sentence"])
        self.labels = labels
        self.texts = texts
        self.input_ids, self.token_type_ids, self.attention_mask = encode_fn(texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index], \
               self.texts[index]


def dev(model, data_loader, config):
    device = config.device
    idx2label = {idx: label for label, idx in config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, labels, texts = batch[0].to(device), batch[1].to(device), batch[
                2].to(device), batch[3].to(device), batch[4]
            logits, mid_feature = model(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]

    tmp_true_labels = []
    for i in true_labels:
        if i == 3:
            tmp_true_labels.append('WORD')
        else:
            tmp_true_labels.append(idx2label[i])
    true_labels = tmp_true_labels

    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    table = classification_report(true_labels, pred_labels)
    f = f1_score(true_labels, pred_labels, average=None)
    f_avg = (f[0] + f[1]) / 2
    return acc, f_avg, table, mid_feature, texts


def get_data(dataset):
    dataset_dir = "corpus/" + dataset

    with open(dataset_dir + '_sentences_clean.txt', 'r') as f_node:
        node_list = f_node.readlines()
        for i in range(len(node_list)):
            if node_list[i][-1] == '\n':
                node_list[i] = node_list[i][:-1]

    with open(dataset_dir + '_labels.txt', 'r') as f_label:
        label_list = f_label.readlines()
        for i in range(len(label_list)):
            if label_list[i][-1] == '\n':
                label_list[i] = label_list[i][:-1]

    with open(dataset_dir + '_opinion_towards.txt', 'r') as f_opto:
        opto_list = f_opto.readlines()
        for i in range(len(opto_list)):
            if opto_list[i][-1] == '\n':
                opto_list[i] = opto_list[i][:-1]

    # with open(dataset_dir + '_targets.txt', 'r') as f_target:
    with open(dataset_dir + '_targets_clean.txt', 'r') as f_target:
        target_list = f_target.readlines()
        for i in range(len(target_list)):
            if target_list[i][-1] == '\n':
                target_list[i] = target_list[i][:-1]

    with open(dataset_dir + '_data_split_tag.txt', 'r') as f_tvt:
        tvt_list = f_tvt.readlines()
        for i in range(len(tvt_list)):
            if tvt_list[i][-1] == '\n':
                tvt_list[i] = tvt_list[i][:-1]

    return node_list, label_list, opto_list, target_list, tvt_list


def get_word_mid_feature(model, data_loader, config):
    device = config.device
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, labels, texts = batch[0].to(device), batch[1].to(device), batch[
                2].to(device), batch[3], batch[4]
            logits, mid_feature = model(input_ids, token_type_ids, attention_mask)

    return mid_feature, texts


def train():
    config = BertConfig()

    logger = get_logger(config.log_path, "berttrain")

    train_dataset = BertDataSet(config.base_config.train_data_path)
    dev_dataset = BertDataSet(config.base_config.dev_data_path)
    test_dataset = BertDataSet(config.base_config.test_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=2814, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1249, shuffle=False)

    best_model = Bert(config)
    save_path = './model/'
    best_model.load_state_dict(torch.load(save_path + "bert.pth"))
    all_data = {}
    acc, f_avg, cls_report, mid_feature, texts = dev(best_model, train_dataloader, config)
    for i in range(len(mid_feature)):
        all_data[texts[i]] = mid_feature[i]
    acc, f_avg, cls_report, mid_feature, texts = dev(best_model, dev_dataloader, config)
    for i in range(len(mid_feature)):
        all_data[texts[i]] = mid_feature[i]
    acc, f_avg, cls_report, mid_feature, texts = dev(best_model, test_dataloader, config)
    for i in range(len(mid_feature)):
        all_data[texts[i]] = mid_feature[i]

    logger.info("TEST: ACC:{}".format(acc))
    print("TEST: ACC:{}".format(acc))
    logger.info("TEST: F_AVG:{}".format(f_avg))
    print("TEST: F_AVG:{}".format(f_avg))
    logger.info("TEST classification report:\n{}".format(cls_report))
    print("TEST classification report:\n{}".format(cls_report))

    dataset = "twitter_target"
    node_list, label_list, opto_list, target_list, tvt_list = get_data(dataset)

    texts_e = []
    for i in range(len(node_list)):
        # node_key = node_list[i]
        node_key = node_list[i] + ' ' + target_list[i]
        features = all_data[node_key]
        texts_e.append(features)

    # 获取词特征

    # 在config.base_config.words_data_path存储词作为节点的json数据
    words = load_json("./dataset/word2idx.json").keys()
    with open(config.base_config.words_data_path, 'w', encoding='utf-8') as f:
        for word in words:
            word_node = {"sentence": word, "label": 'WORD'}
            json.dump(word_node, f, ensure_ascii=False)
            f.write('\n')

    with open('feature_words.txt', 'r') as s_f:
        # FAVOR
        # NONE
        # AGAINST
        # 1259*3
        selected_word = s_f.readline().strip().split(' ')[1259 * 2:1259 * 3]
        # selected_word = s_f.readline().strip().split(' ')[:1259]

    print(selected_word)
    print(len(selected_word))

    with open('select_example.txt', 'w') as exp_f:
        text_node_id = range(0, 4163)
        all_ww = []
        for i in text_node_id:
            t = 0
            ws = []
            text_words = node_list[i].strip().split(' ')
            for w in text_words:
                if w in selected_word:
                    ws.append(w)
                    t = t + 1
            if t > 10 and label_list[i] == 'AGAINST' and target_list[i] == 'atheism':
                print(node_list[i])
                print(ws)
                print(t)
                all_ww += ws
        print(get_words_freq(all_ww))

    '''
    433
    #Lord Jesus, You are altogether lovely! O Lord #Jesus, You are altogether lovely! -LifeAndBuildingAsPortrayedInTheSongOfSongs  #SemST
    lord jesus altogether lovely lord jesus altogether lovely life building portrayed song songs
    ['lord', 'jesus', 'altogether', 'lovely', 'lord', 'jesus', 'altogether', 'lovely', 'life', 'portrayed', 'song', 'songs']
    421
    Good morning. Praise the LORD; For His mercy endures FOREVER!!! #Repent #USA #JesusIsLord #Hallelujah #PraisetheLord #homosexuality #SemST
    good morning praise lord mercy endures forever repent usa jesus lord hallelujah praise lord homosexuality
    ['good', 'praise', 'lord', 'mercy', 'forever', 'usa', 'jesus', 'lord', 'hallelujah', 'praise', 'lord', 'homosexuality']
    272
    Let us offer through Jesus a continual sacrafice of praise to God, proclaiming our allegiance to his name.  Hebrews13:15NLT #God #SemST
    let offer jesus continual sacra ice praise god proclaiming allegiance name hebrews god
    ['let', 'jesus', 'continual', 'sacra', 'ice', 'praise', 'god', 'proclaiming', 'allegiance', 'name', 'god']
    '''
    # e1 = [

    #     'today verse hears hears rejects rejects jesus luke born team gic bible holy spirit',
    #     'never get old hear christians lead others ultimate love hope disciples share gospel jesus changes everything',
    #     'good morning praise lord mercy endures forever repent usa jesus lord hallelujah praise lord homosexuality',
    #     'cor stand firm always give fully work lord labor lord vain god'
    # ]
    # for q in e1:
    #     s = q.split(' ')
    #     print(q)
    #     lln = []
    #     for sw in s:
    #         if sw in selected_word:
    #             lln.append(sw)
    #     print(lln)


if __name__ == "__main__":
    train()
