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


def run():
    dataset = "twitter_target"
    node_list, label_list, opto_list, target_list, tvt_list = get_data(dataset)
    # label_list  反对-1 支持1 不支持不反对0
    # opto_list   有关0 其他1 不表达2
    # 4163 = test 0-1248 + train 1249-4062  + trial  4063-4162

    dataset = []
    for node in node_list:
        dataset += node.split(' ')
    words_freq = Counter(dataset)
    vocabs = words_freq.keys()
    id_word = {}
    word_id = {}
    for id, word in enumerate(vocabs):
        word_id[word] = id
        id_word[id] = word

    # log-odds-ratio获取label-word之间信息
    labels = set(label_list)
    id_label = {}
    label_id = {}
    for id, label in enumerate(labels):
        id_label[id] = label
        label_id[label] = id

    def get_word_freq(text_list):
        """根据text_list获取词-词频"""
        word_freq = defaultdict(int)
        for doc_words in text_list:
            words = doc_words.split()
            for word in words:
                word_freq[word] += 1
        return word_freq

    # 获取不同label对应的语料库的大小
    label_nodes = defaultdict(list)
    # {label:[nodes]}
    # 测试集验证集的label不能用
    for i in range(1249, 4063):
        for label in labels:
            if label == label_list[i]:
                label_nodes[label].append(node_list[i])

    label_corpus_dic = defaultdict(int)
    for l_id, label in id_label.items():
        label_corpus_word_freq = get_word_freq(label_nodes[label])
        label_corpus_size = sum([num for word, num in label_corpus_word_freq.items()])
        label_corpus_dic[l_id] = label_corpus_size

    # 获取整个语料库大小
    corpus_size = sum([size for l, size in label_corpus_dic.items()])

    # word_labels {word:[label_id]}
    word_labels = defaultdict(list)
    for i in range(1249, 4063):
        words = node_list[i].split()
        for w in words:
            word_labels[w].append(label_id[label_list[i]])

    # word_labels_freq {word:[{label:label_freq}]}
    word_labels_freq = {}
    for word, label_id_list in word_labels.items():
        label_freq = {}
        # 词word在某个label上的词频
        for label, i in label_id.items():
            label_freq[i] = len([1 for id in label_id_list if id == i])
        word_labels_freq[word] = label_freq

    print(word_labels_freq)

    for t_id in range(3):
        word_lor = defaultdict(float)
        for w, l_freq in tqdm(word_labels_freq.items()):
            # w在label语料库t_id中的频率
            y_w = l_freq[t_id]
            # w在整个语料库i中的频率
            a_w = sum([v for k, v in l_freq.items()])
            # 语料库i的大小
            n_i = label_corpus_dic[t_id]
            # 整个语料库的大小
            a = corpus_size
            lor = log(y_w + a_w) / (n_i + a - y_w - a_w)
            word_lor[w] = lor
            # print('log({}+{})/({}+{}-{}-{})'.format(y_w,a_w,n_i,a,y_w,a_w))
            # if lor == 0:
            #     # 2000+词只出现一次
            #     print('log({}+{})/({}+{}-{}-{})'.format(y_w, a_w, n_i, a, y_w, a_w))
            # print('词{:10s}在label:{}中的lor为{}'.format(w, id_label[t_id], lor))

        order_word_lor = sorted(word_lor.items(), key=lambda x: x[1], reverse=True)
        print(order_word_lor[:30])


run()
