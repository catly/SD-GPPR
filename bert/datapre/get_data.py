import os
from os.path import join, exists
from os.path import dirname, abspath, join, exists
import wordninja
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import re
import json
from collections import Counter


class GetDataset():
    def __init__(self, dirs):
        self.topic = []
        self.label = []
        self.text = []
        self.opinion_towards = []
        self.data_split_tag = []
        dirs = [dirs]
        for dir in dirs:
            split_tag = dir[:-4].split('/')[-1]
            with open(dir, 'r') as self.f:
                self.header = next(self.f)
                for row in self.f:
                    data = row.split('\t')
                    tmp_topic = ""
                    for i in data[1].lower().split(' '):
                        tmp_topic = tmp_topic + i + ''
                    self.topic.append(tmp_topic.strip())
                    self.text.append(data[2])
                    self.label.append(data[3])
                    self.opinion_towards.append(data[4])
                    self.data_split_tag.append(split_tag)
        self.length = len(self.label)


def clean_doc(string):
    string = string.lower()
    string = re.sub(r"#semst", "", string)
    string = re.sub(r"@\S*", " ", string)

    # ??   U.S.
    string = re.sub(r"[^a-z\']", " ", string)

    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"[\W\d_]", " ", string)
    string = re.sub(r" [a-z] ", " ", string)
    string = re.sub(r" [a-z] ", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = "".join(string)
    string = wordninja.split(string)

    return string


def p_clean(sentence):
    clean_sentence = clean_doc(sentence.strip())
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    words = [word for word in clean_sentence if word not in stop_words and len(word) >= 3]
    clean_sentence = ' '.join(words).strip()
    return clean_sentence


def get_sub_dataset(dataset):
    dataset_dir = "./StanceData/" + dataset + '.txt'
    sub_dataset = GetDataset(dataset_dir)
    save_sub_dataset_dir = "../dataset/" + dataset + '.json'

    with open(save_sub_dataset_dir, 'w') as sub_f:
        labels = sub_dataset.label
        texts = sub_dataset.text
        topics = sub_dataset.topic
        opinion_towards = sub_dataset.opinion_towards
        length = sub_dataset.length
        dict = {

        }
        for i in range(length):
            # dict['sentence'] = p_clean(texts[i])
            dict['sentence'] = p_clean(texts[i] + ' ' + topics[i])
            dict['label'] = labels[i]
            print(dict)
            json.dump(dict, sub_f)
            sub_f.write('\n')


def main():
    get_sub_dataset('train')
    get_sub_dataset('test')
    get_sub_dataset('dev')
    with open('../dataset/dev.json', 'r') as df:
        with open('../dataset/val.json', 'w') as vf:
            t_data = df.readlines()
            for one_data in t_data:
                vf.write(one_data)


main()
