'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class EvaluateAcc(evaluate):
    data = None
    weighted_tag = False
    
    def evaluate(self):
        if self.weighted_tag:
            return accuracy_score(self.data['true_y'], self.data['pred_y']), f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        else:
            return accuracy_score(self.data['true_y'], self.data['pred_y']), f1_score(self.data['true_y'], self.data['pred_y'])
        