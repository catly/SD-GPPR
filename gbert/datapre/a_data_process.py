import os
import wordninja


class GetDataset():
    def __init__(self, dirs):
        self.topic = []
        self.label = []
        self.text = []
        self.opinion_towards = []
        self.data_split_tag = []
        for dir in dirs:
            split_tag = dir[:-4].split('/')[-1]
            with open(dir, 'r') as self.f:
                self.header = next(self.f)
                for row in self.f:
                    data = row.split('\t')
                    print(data)
                    tmp_topic = ""
                    for i in data[1].lower().split(' '):
                        tmp_topic = tmp_topic + i + ''
                    self.topic.append(tmp_topic.strip())
                    self.text.append(data[2])
                    self.label.append(data[3])
                    self.opinion_towards.append(data[4])
                    self.data_split_tag.append(split_tag)

        self.length = len(self.label)



def main():
    root_dir = "../datapre/StanceData/"
    data_dirs = [root_dir + i for i in os.listdir(root_dir)]

    dataset = GetDataset(data_dirs)

    texts = dataset.text
    labels = dataset.label
    topics = dataset.topic
    opinion_towards = dataset.opinion_towards
    data_split_tags = dataset.data_split_tag
    length = dataset.length

    with open('../datapre/corpus/twitter_target_sentences.txt', 'w') as f:
        for text in texts:
            f.write(text + '\n')

    with open('../datapre/corpus/twitter_target_targets.txt', 'w') as f:
        for topic in topics:
            f.write(" ".join(wordninja.split(topic)) + '\n')

    with open('../datapre/corpus/twitter_target_labels.txt', 'w') as f:
        for label in labels:
            f.write(label + '\n')

    with open('../datapre/corpus/twitter_target_opinion_towards.txt', 'w') as f:
        for opinion_toward in opinion_towards:
            f.write(opinion_toward + '\n')

    with open('../datapre/corpus/twitter_target_data_split_tag.txt', 'w') as f:
        for tag in data_split_tags:
            f.write(tag + '\n')

    # print(length)
    # 有关1 其他2 不表达3
    # 反对0 支持1 及不反对也不支持2

    l1 = [0, 0, 0]
    l2 = [0, 0, 0]
    l3 = [0, 0, 0]

    test_a_f_n = [0, 0, 0]
    for i in range(length):
        if data_split_tags[i] == "train":
            print(i)
            if opinion_towards[i] == 'TARGET':
                # -1:1535  0:9  1:996
                if labels[i] == 'FAVOR':
                    l1[0] += 1
                    print("与target有关TARGET,label=FAVOR" + texts[i] + ' ' + topics[i])
                elif labels[i] == 'NONE':
                    l1[1] += 1
                    print("与target有关TARGET,label=NONE" + texts[i] + ' ' + topics[i])
                else:  # labels[i] == 'AGAINST'
                    l1[2] += 1
                    print("与target有关TARGET,label=AGAINST" + texts[i] + ' ' + topics[i])
            elif opinion_towards[i] == 'OTHER':
                # -1:560  0:792  1:54
                if labels[i] == 'FAVOR':
                    l2[0] += 1
                    print("与其他target有关OTHER,label=FAVOR" + texts[i] + ' ' + topics[i])
                elif labels[i] == 'NONE':
                    l2[1] += 1
                    print("与其他target有关OTHER,label=NONE" + texts[i] + ' ' + topics[i])
                else:  # labels[i] == 'AGAINST'
                    l2[2] += 1
                    print("与其他target有关OTHER,label=AGAINST" + texts[i] + ' ' + topics[i])
            else: # opinion_towards[i] == 'NO ONE'
                if labels[i] == 'FAVOR':
                    l3[0] += 1
                    print("与target有关2,label=FAVOR" + texts[i] + ' ' + topics[i])
                elif labels[i] == 'NONE':
                    l3[1] += 1
                    print("与target有关2,label=NONE" + texts[i] + ' ' + topics[i])
                else:  # labels[i] == 'AGAINST'
                    l3[2] += 1
                    print("与target有关2,label=AGAINST" + texts[i] + ' ' + topics[i])
        if data_split_tags[i] == 'test':
            if labels[i] == 'AGAINST':
                test_a_f_n[0] += 1
            elif labels[i] == 'FAVOR':
                test_a_f_n[1] += 1
            else:
                test_a_f_n[2] += 1
    print(test_a_f_n)

    l = [l1, l2, l3]
    #    -1 0 1
    # 0
    # 1
    # 2
    print(l)

    del texts
    del labels
    del topics
    del opinion_towards


main()
