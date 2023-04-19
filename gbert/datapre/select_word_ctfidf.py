from collections import Counter, defaultdict
from math import log
from tqdm import tqdm


def get_data(dataset):
    dataset_dir = "corpus/" + dataset

    with open(dataset_dir + '_sentences_clean.txt', 'r') as f_node:
        node_list = f_node.readlines()
        for i in range(len(node_list)):
            node_list[i] = node_list[i][:-1]

    with open(dataset_dir + '_labels.txt', 'r') as f_label:
        label_list = f_label.readlines()
        for i in range(len(label_list)):
            label_list[i] = label_list[i][:-1]

    with open(dataset_dir + '_opinion_towards.txt', 'r') as f_opto:
        opto_list = f_opto.readlines()
        for i in range(len(opto_list)):
            opto_list[i] = opto_list[i][:-1]

    with open(dataset_dir + '_targets.txt', 'r') as f_target:
        target_list = f_target.readlines()
        for i in range(len(target_list)):
            target_list[i] = target_list[i][:-1]

    with open(dataset_dir + '_data_split_tag.txt', 'r') as f_tvt:
        tvt_list = f_tvt.readlines()
        for i in range(len(tvt_list)):
            tvt_list[i] = tvt_list[i][:-1]

    return node_list, label_list, opto_list, target_list, tvt_list


def get_words_freq(text_list):
    """根据text_list获取词-词频"""
    dataset = []
    for text in text_list:
        dataset += text.split(' ')
    words_freq = Counter(dataset)
    return words_freq


def run():
    dataset = "twitter_target"
    node_list, label_list, opto_list, target_list, tvt_list = get_data(dataset)
    # label_list  反对-1 支持1 不支持不反对0
    # opto_list   有关0 其他1 不表达2
    # 4163 = test 0-1248 + train 1249-4062  + trial  4063-4162

    words_freq = get_words_freq(node_list)
    vocabs = words_freq.keys()
    id_word = {}
    word_id = {}
    for id, word in enumerate(vocabs):
        word_id[word] = id
        id_word[id] = word

    labels = set(label_list)
    id_label = {}
    label_id = {}
    for id, label in enumerate(labels):
        id_label[id] = label
        label_id[label] = id

    # 获取不同label对应的语料库的词频
    label_nodes = defaultdict(list)
    # {label:[nodes]}
    # 测试集验证集的label不能用
    for i in range(1249, 4063):
        for label in labels:
            if label == label_list[i]:
                label_nodes[label].append(node_list[i])

    # {label:{word:freq}}
    label_corpus_word_freq = defaultdict(dict)
    for l_id, label in id_label.items():
        label_corpus_word_freq[label] = get_words_freq(label_nodes[label])

    # {word:{label:freq}}
    word_corpus_label_freq = defaultdict(dict)
    for w_id, word in id_word.items():
        tmp_dic = defaultdict(int)
        for l, i in label_id.items():
            if label_corpus_word_freq[l][word]:
                tmp_dic[l] = label_corpus_word_freq[l][word]
            else:
                # 类中未出现某个词 该词的频率记为0
                tmp_dic[l] = 0
        word_corpus_label_freq[word] = tmp_dic

    # {label:{words}}
    label_feature_word = defaultdict(set)
    for l, word_freq in label_corpus_word_freq.items():
        word_ctfidf = defaultdict(float)
        for w, f in word_freq.items():
            tf_w_l = f
            a = sum(word_freq.values()) / len(word_freq.keys())
            tf_w = sum(word_corpus_label_freq[w].values())
            c_tfidf = tf_w_l * log(1 + a / tf_w)
            word_ctfidf[w] = c_tfidf
        print(l)
        tmp = sorted(word_ctfidf.items(), key=lambda x: x[1], reverse=True)[:20]
        #前1259*3=3777个词 没有交集
        print(tmp)
        label_feature_word[l] = set(w for (w,w_c_tfidf) in tmp)
    # print(label_feature_word)
    w_set = set()
    for l,l_w_set in label_feature_word.items():
        w_set.update(l_w_set)

    # with open('feature_words','w') as f_w:
    #     for w in w_set:
    #         f_w.write(w+' ')
    # print(len(w_set))

    # with open('node', 'w') as f_n:
    #     for node in node_list:
    #         pass
    #
    #     id_feature = defaultdict(str)
    #     feature_id = defaultdict(int)
    #     for i,feature in enumerate(w_set):
    #         id_feature[i] = feature
    #         feature_id[feature] = i
    #
    #     with open('node', 'w') as f_n:
    #         for i, node in enumerate(node_list):
    #             node_words = node.split()
    #             node_one_hot_feature = [0] * len(w_set)
    #             for word in node_words:
    #                 if word in w_set:
    #                     f_id = feature_id[word]
    #                     node_one_hot_feature[f_id] = 1
    #             print(len(node_one_hot_feature))
    #             f_n.write(str(i)+ '\t')
    #             for f in node_one_hot_feature:
    #                 f_n.write(str(f) + '\t')
    #             f_n.write(str(label_list[i]) + '\n')



'''
'

'''



run()
