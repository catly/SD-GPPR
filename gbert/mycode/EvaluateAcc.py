'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD
import sklearn.metrics

from gbert.mycode.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class EvaluateAcc(evaluate):
    data = None
    def evaluate(self):
        x, y = self.data['true_y'], self.data['pred_y']
        # f1 = f1_score(x,y,average='micro')
        # f1 = f1_score(x, y, average='macro')

        f1 = f1_score(x, y, average=None)
        f1_avg_af = (f1[0] + f1[1]) / 2

        # f2 = f1_score(x,y,labels=[0,1],average=None)
        # print(f1)
        # print(f2)
        # print('////////////')
        return accuracy_score(self.data['true_y'], self.data['pred_y']), f1, f1_avg_af, sklearn.metrics.classification_report(x, y)
