import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm


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
    # with open('link', 'w') as f_link:
    #     for i in tqdm(range(len(node_list))):
    #         for j in range(i + 1, len(node_list)):
    #             if target_list[i] == target_list[j] and opto_list[i] == opto_list[j] and \
    #                     label_list[i] == label_list[j] and tvt_list[i] == 'train' and tvt_list[j] == 'train':
    #                 f_link.write(str(i) + '\t' + str(j) + '\n')

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    texts_e = []
    for text in tqdm(node_list):
        input_ids = torch.tensor([tokenizer.encode(text)])
        with torch.no_grad():
            features = bertweet(input_ids)
            texts_e.append(features[1])

    data = np.array(texts_e)
    # 存储边关系
    similar_matrix = torch.zeros((len(data), len(data)))
    for i in tqdm(range(len(data))):
        for j in range(i + 1, len(data)):
            similar_matrix[i][j] = torch.cosine_similarity(data[i], data[j])

    with open('link', 'w') as f:
        t = 0
        for i in range(len(similar_matrix)):
            for j in range(len(similar_matrix[i])):
                if similar_matrix[i][j] > 0.987 and target_list[i] == target_list[j] and \
                        opto_list[i] == opto_list[j] and label_list[i] == label_list[j] and \
                        tvt_list[i] == 'train' and tvt_list[j] == 'train':
                    print(str(t) + ':  ' + str(similar_matrix[i][j]))
                    t = t + 1
                    f.write(str(i) + '\t' + str(j) + '\n')
            # f.write(str(i) + '\t' + str(int(torch.argmax(similar_matrix[i]))) + '\n')
            # print(torch.argmax(similar_matrix[i]))
            # similar_matrix[i][int(torch.argmax(similar_matrix[i]))] = 0
            # f.write(str(i) + '\t' + str(int(torch.argmax(similar_matrix[i]))) + '\n')
            # print(torch.argmax(similar_matrix[i]))
        print(t)


run()
