import torch
from transformers import AutoModel, AutoTokenizer
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

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    texts_e = []
    for node in tqdm(node_list):
        input_ids = torch.tensor([tokenizer.encode(node)])
        with torch.no_grad():
            features = bertweet(input_ids)
            texts_e.append(features[1])

    with open('node', 'w') as f:
        for i in tqdm(range(len(texts_e))):
            print(i)
            line = str(i) + '\t'
            for num in texts_e[i][0]:
                line = line + str(float(num)) + '\t'
            line = line + str(label_list[i]) + '\n'
            f.write(line)


run()
