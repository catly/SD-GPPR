import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModel, AutoTokenizer

np.random.seed(1)
torch.manual_seed(1)
batch_size = 16
max_len = 50
lstm_h_in = 768
lstm_h_out = 300
lstm_num_layers = 2


class MyDataset(Dataset):
    def __init__(self, node_list, label_list, opto_list, target_list, tvt_list, tvt):
        self.total_len = len(node_list)
        self.nodes = []
        self.labels = []
        for i in range(self.total_len):
            if tvt_list[i] == tvt:
                # self.nodes.append(node_list[i] + target_list[i])
                self.nodes.append(node_list[i])
                self.labels.append(label_list[i])
            self.len_dataset = len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        label = self.labels[idx]

        sample = {'node': node, 'label': label}
        # print(sample)
        return sample

    def __len__(self):
        return self.len_dataset


def collate_fn(batch):
    node = []
    label = torch.zeros((batch_size, 3))
    for i in range(batch_size):
        node.append(batch[i]['node'])
        if batch[i]['label'] == '-1':
            label[i] = torch.LongTensor([[1.0, 0, 0]])
        elif batch[i]['label'] == '1':
            label[i] = torch.LongTensor([[0, 1.0, 0]])
        else:
            label[i] = torch.LongTensor([[0, 0, 1.0]])

    return node, label


class bert_lstm_liner(nn.Module):
    def __init__(self):
        super(bert_lstm_liner, self).__init__()

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        self.encoder = nn.LSTM(lstm_h_in, lstm_h_out, lstm_num_layers, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 50),
            nn.Tanh(),
            nn.Linear(50, 3),
            nn.Softmax()
        )
        self.liner = nn.Sequential(nn.Linear(600, 3))

    def forward(self, x):
        batch_features = torch.zeros((batch_size, max_len, 768))
        for i in range(batch_size):
            input_ids = torch.tensor([self.tokenizer.encode(x[i])])
            features = self.bertweet(input_ids)
            batch_features[i][:len(features[0][0])] = features[0][0]  # len*768
        h0 = torch.randn(2 * lstm_num_layers, batch_size, lstm_h_out)
        c0 = torch.randn(2 * lstm_num_layers, batch_size, lstm_h_out)
        output, (hn, cn) = self.encoder(batch_features.transpose(0, 1), (h0, c0))
        # 第二层正向的隐藏状态
        f_last = hn[-2, :, :]
        # 第二层反向的隐藏状态
        b_last = hn[-1, :, :]
        out = torch.cat((f_last, b_last), dim=-1)

        y = self.mlp(out)
        return y

    # def forward(self, x):
    #     batch_features = torch.zeros((batch_size, 1, 768))
    #     for i in range(batch_size):
    #         input_ids = torch.tensor([self.tokenizer.encode(x[i])])
    #         features = self.bertweet(input_ids)[1]
    #         batch_features[i] = features  # 1*768
    #     h0 = torch.randn(2 * lstm_num_layers, batch_size, lstm_h_out)
    #     c0 = torch.randn(2 * lstm_num_layers, batch_size, lstm_h_out)
    #     output, (hn, cn) = self.encoder(batch_features.transpose(0, 1), (h0, c0))
    #     y = self.liner(output)[0]
    #
    #     return y


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


def main():
    dataset = "twitter_target"
    node_list, label_list, opto_list, target_list, tvt_list = get_data(dataset)

    train_data = MyDataset(node_list, label_list, opto_list, target_list, tvt_list, 'train')  # 20715条
    validation_data = MyDataset(node_list, label_list, opto_list, target_list, tvt_list, 'trial')  # 3635条
    test_data = MyDataset(node_list, label_list, opto_list, target_list, tvt_list, 'test')  # 4870条

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True,
                                  collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_data, shuffle=True, batch_size=batch_size, drop_last=True,
                                       collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)

    model = bert_lstm_liner()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train
    total_train_step = 0
    total_text_step = 0
    epoch = 20
    acc = 0

    for i in range(epoch):
        total_num = 0
        acc_num = 0
        print('--------第{}个eopch训练开始--------'.format(i))
        for batch in train_dataloader:
            node, label = batch
            output = model(node)
            loss = loss_fn(output, label)
            total_num += 1

            if torch.argmax(output).item() == torch.argmax(label).item():
                acc += 1

            # 优化器优化模型
            # 梯度清零
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 1 == 0:
                print('第{}次训练，Loss:{}'.format(total_train_step, loss.item()))

            acc = acc_num / total_num
            # print('第{}个eopch训练准确率：{}'.format(i, acc))

        # 测试
        total_loss = 0
        total_num = 0
        acc_num = 0
        with torch.no_grad():
            for batch in test_dataloader:
                node, label = batch
                output = model(node)

                loss = loss_fn(output, label)
                total_num += 1

                if torch.argmax(output).item() == torch.argmax(label).item():
                    acc += 1
                # 优化器优化模型
                # 梯度清零
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_text_step += 1
                if total_text_step % 20 == 0:
                    print('第{}次测试，Loss:{}'.format(total_train_step, loss.item()))

            acc = acc_num / total_num
            print('第{}个eopch训练准确率：{}'.format(i, acc))

        print('保存模型')
        torch.save(model, './save_model/lstm_mlp{}.pth'.format(i))


main()
