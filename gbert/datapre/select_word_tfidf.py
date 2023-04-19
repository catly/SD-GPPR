import time
from collections import Counter, defaultdict
from math import log


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

    with open(dataset_dir + '_targets.txt', 'r') as f_target:
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


def run():
    dataset = "twitter_target"
    node_list, label_list, opto_list, target_list, tvt_list = get_data(dataset)
    # label_list  反对-1 支持1 不支持不反对0
    # opto_list   有关0 其他1 不表达2

    # 获取{词:词频}
    dataset = []
    for node in node_list:
        dataset += node.split(' ')
    words_freq = Counter(dataset)
    vocabs = words_freq.keys()
    print(vocabs)
    id_word = {}
    word_id = {}
    for id, word in enumerate(vocabs):
        word_id[word] = id
        id_word[id] = word
    # 1607  2686
    vocabs_size = len(word_id)
    # TF-IDF
    # 文档-词共现的次数 (node_id,word_id):num
    node_word_freq = defaultdict(int)
    for i, node in enumerate(node_list):
        node_words = node.split()
        for word in node_words:
            w_id = word_id[word]
            doc_word_str = (i, w_id)
            node_word_freq[doc_word_str] += 1

    # {词:文档} set
    words_in_node = defaultdict(set)
    for i, node in enumerate(node_list):
        node_words = node.split()
        #  存储当前文档计算过idf的词
        for word in node_words:
            words_in_node[word].add(i)

    with open('node', 'w') as f_tfidf:
        for i, node in enumerate(node_list):
            node_words = node.split()
            #  存储当前文档计算过idf的词
            node_word_set = set()
            print('文档{}:{}  标签：{}'.format(i, node, label_list[i]))
            f_tfidf.write(str(i) + '\t')
            # 8500
            node_one_hot_feature = [0.0] * vocabs_size
            for word in node_words:
                if word in node_word_set:
                    # 同一个文档中计算过的词不再计算
                    continue
                w_id = word_id[word]
                # tf = 词出现在当前节点的数量/文档节点总词数  当前文档节点和某个词的共现次数
                freq = node_word_freq[(i, w_id)]
                tf = freq / len(node_words)
                # idf = 文档总数/出现词的文档数
                idf = log(len(node_list) /
                          len(words_in_node[id_word[w_id]]) + 1)
                # log(文档总数/出现该词的文档数+1)
                tfidf = freq * idf
                node_one_hot_feature[w_id] = tfidf
                print('文档{}中，词{:10s}的TF-IDF值为{}'.format(i, word, tfidf))
            for f in node_one_hot_feature:
                f_tfidf.write(str(f) + '\t')
            f_tfidf.write(str(label_list[i]) + '\n')
    print()


run()
