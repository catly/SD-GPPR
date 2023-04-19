import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from gbert.mycode.MethodGraphBert import MethodGraphBert

import time
import numpy as np

from gbert.mycode.EvaluateAcc import EvaluateAcc

from sklearn.metrics import classification_report, f1_score
import warnings
import os

warnings.filterwarnings("ignore")

BertLayerNorm = torch.nn.LayerNorm
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4  # 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
        residual_h, residual_y = self.residual_term()
        if idx is not None:
            if residual_h is None:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx],
                                    residual_h=None)
            else:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx],
                                    residual_h=residual_h[idx])
                residual_y = residual_y[idx]
        else:
            if residual_h is None:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None)
            else:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)

        # print('time1')
        # print(time.strftime('%H:%M:%S',time.localtime(time.time())))
        # 11-5 原

        sequence_output = 0
        for i in range(self.config.k + 1):
            sequence_output += outputs[0][:, i, :]
        sequence_output /= float(self.config.k + 1)

        # # 11-5 改
        # # print(outputs[0].shape)  # (data_num)*(k+1)*(hidden_size)
        # sequence_output = outputs[0][:, 0, :]
        # data_num = sequence_output.shape[0]
        # sequence_output_k = torch.ones(data_num,1)
        # cos_similar = torch.nn.CosineSimilarity(0)
        # for i in range(self.config.k):
        #     former_node = outputs[0][:, i, :]   # (data_num)*(hidden_size)
        #     latter_node = outputs[0][:, i+1, :]
        #     for j in range(data_num):
        #         similar_score = cos_similar(former_node[j],latter_node[j])
        #         # print("{}&{} similar score:".format(i,i+1)+str(similar_score))
        #         if similar_score >= 0.2 and sequence_output_k[j] == i+1:
        #             sequence_output[j] += latter_node[j]
        #             sequence_output_k[j] += 1
        # # for t in sequence_output_k:
        # #     print(t)
        # sequence_output /= sequence_output_k
        # # 11-5 改

        # sequence_output += outputs[0][:, 0, :]
        labels = self.cls_y(sequence_output)

        # print('fin')
        # print(time.strftime('%H:%M:%S', time.localtime(time.time())))

        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)

    def residual_term(self):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(self.data['X']), self.res_y(self.data['X'])
        elif self.config.residual_type == 'graph_raw':
            return torch.spmm(self.data['A'], self.res_h(self.data['X'])), torch.spmm(self.data['A'],
                                                                                      self.res_y(self.data['X']))

    def train_model(self, max_epoch):

        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')

        # max_score = 0.0
        # if (os.path.exists('class_report')):
        #     # 存在，则删除文件
        #     os.remove("class_report")

        for epoch in range(max_epoch):
            # print(epoch)
            # with open('class_report', 'a') as ff:
            #     ff.write('Epoch:' + str(epoch) + '\n')
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
                                  self.data['hop_embeddings'], self.data['idx_train'])
            loss_train = F.cross_entropy(output, self.data['y'][self.data['idx_train']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output[:, :3].max(1)[1]}

            acc_train, f1_train, favg_train, report_train = accuracy.evaluate()
            # print(loss_train)
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
                                  self.data['hop_embeddings'], self.data['idx_val'])

            loss_val = F.cross_entropy(output, self.data['y'][self.data['idx_val']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_val']],
                             'pred_y': output[:, :3].max(1)[1]}
            acc_val, f1_val, favg_val, report_val = accuracy.evaluate()

            # -------------------------
            # ---- keep records for drawing convergence plots ----
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
                                  self.data['hop_embeddings'], self.data['idx_test'])
            loss_test = F.cross_entropy(output, self.data['y'][self.data['idx_test']])
            accuracy.data = {"id":self.data['idx_test'],
                             'true_y': self.data['y'][self.data['idx_test']],
                             'pred_y': output[:, :3].max(1)[1]                             }
            # print("000000000")
            # print(epoch)
            # id_l =  accuracy.data["id"]
            # ty_list =  accuracy.data['true_y']
            # py_list =  accuracy.data["pred_y"]
            # for ii in id_l:
            #     if ty_list[ii]!=py_list[ii]:
            #         print(ii,ty_list[ii],py_list[ii])

            acc_test, f1_test, favg_test, report_test = accuracy.evaluate()

            # # 添加分类的f值
            # class_targets = []
            # with open('class', 'r') as f:
            #     for t in f.readlines():
            #         class_targets.append(t[:-1])
            #
            # labels_dict = {0: 'AGAINST', 1: 'FAVOR', 2: 'NONE'}
            # class_true_labels = []
            # class_pred_labels = []
            # for i in range(len(accuracy.data['true_y'])):
            #     class_pred_labels.append(labels_dict[accuracy.data['pred_y'][i].item()])
            #     class_true_labels.append(labels_dict[accuracy.data['true_y'][i].item()])
            #
            # target_set = set(class_targets)
            #
            # with open('class_report','a') as ff:
            #     for t in target_set:
            #         sub_pre = []
            #         sub_true = []
            #         for i in range(len(class_targets)):
            #             if class_targets[i] == t:
            #                 sub_pre.append(class_pred_labels[i])
            #                 sub_true.append(class_true_labels[i])
            #
            #         table = classification_report(sub_true, sub_pre)
            #         f = f1_score(sub_true, sub_pre, average=None)
            #         f_avg = (f[0] + f[1]) / 2
            #         ff.write(table)
            #         ff.write('\n')
            #         ff.write(str(f))
            #         ff.write('\n')
            #         ff.write(str(t) + "   " + str(f_avg))
            #         ff.write('\n')

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'acc_train': acc_train.item(),
                                                'report_train':report_train,
                                                'loss_val': loss_val.item(), 'acc_val': acc_val.item(),
                                                'favg_val': favg_val,'report_val':report_val,
                                                'loss_test': loss_test.item(), 'acc_test': acc_test.item(),
                                                'f1_test': f1_test, 'favg_test': favg_test,'report_test':report_test,
                                                'time': time.time() - t_epoch_begin}

            # -------------------------

            if epoch % 10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_test: {:.4f}'.format(acc_test.item()),
                      'f1_test: {}'.format(f1_test),
                      'favg_test: {:.4f}'.format(favg_test),
                      'report_test\n{}'.format(report_test),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))


        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin) +
              ', best testing performance {: 4f}'.format(
                  np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])) +
              ', minimun loss {: 4f}'.format(
                  np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])) +
              ', max_favg {: 4f}'.format(
                  np.max([self.learning_record_dict[epoch]['favg_test'] for epoch in self.learning_record_dict]))
              )

        # for i in range(len(self.learning_record_dict)):
        #     print(i)
        #     print(self.learning_record_dict[i]['favg_val'])
        #     print(self.learning_record_dict[i]['f1_test'])
        #     print(self.learning_record_dict[i]['favg_test'])

        print("---------------------------------------------------------")


        e = np.argmax([self.learning_record_dict[epoch]['favg_test'] for epoch in self.learning_record_dict])
        print("test_favg_max epoch=" + str(e))
        f_avg = self.learning_record_dict[e]['favg_val']
        acc = self.learning_record_dict[e]['acc_val']
        print("val    acc:{}    f_avg:{}".format(acc, f_avg))

        f = self.learning_record_dict[e]['f1_test']
        f_avg = self.learning_record_dict[e]['favg_test']
        acc = self.learning_record_dict[e]['acc_test']

        print("test   acc:{}    f_avg:{}".format(acc, f_avg))
        print("{}".format(f))
        print(self.learning_record_dict[e]['report_test'])

        print("---------------------------------------------------------")


        e = np.argmax([self.learning_record_dict[epoch]['favg_val'] for epoch in self.learning_record_dict])
        print("val_favg_max epoch=" + str(e))
        f_avg = self.learning_record_dict[e]['favg_val']
        acc = self.learning_record_dict[e]['acc_val']
        print("val    acc:{}    f_avg:{}".format(acc, f_avg))

        f = self.learning_record_dict[e]['f1_test']
        f_avg = self.learning_record_dict[e]['favg_test']
        acc = self.learning_record_dict[e]['acc_test']

        print("test   acc:{}    f_avg:{}".format(acc, f_avg))
        print("{}".format(f))
        print(self.learning_record_dict[e]['report_test'])

        print("---------------------------------------------------------")

        e = np.argmax([self.learning_record_dict[epoch]['acc_val'] for epoch in self.learning_record_dict])
        print("val_acc_max epoch=" + str(e))
        f_avg = self.learning_record_dict[e]['favg_val']
        acc = self.learning_record_dict[e]['acc_val']
        print("val    acc:{}    f_avg:{}".format(acc, f_avg))

        f = self.learning_record_dict[e]['f1_test']
        f_avg = self.learning_record_dict[e]['favg_test']
        acc = self.learning_record_dict[e]['acc_test']

        print("test   acc:{}    f_avg:{}".format(acc, f_avg))
        print("{}".format(f))
        print(self.learning_record_dict[e]['report_test'])
        print("---------------------------------------------------------")

        return time.time() - t_begin, np.max(
            [self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])

    def run(self):

        self.train_model(self.max_epoch)

        return self.learning_record_dict
