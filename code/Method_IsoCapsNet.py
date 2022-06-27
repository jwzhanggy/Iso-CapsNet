from code.base_class.method import method
from code.Method_IsoCapsNet_Modules import IsoCapsuleNet, CapsuleLoss
from code.Evaluate_Acc import EvaluateAcc
from torch import nn
import torch.optim as optim
import torch
import math
import time
import numpy as np


class Method_IsoCapsNet(method, nn.Module):
    data = None
    hyper_parameters = None
    P_set = None

    def __init__(self, mName, mDescription, hyper_parameters):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.hyper_parameters = hyper_parameters
        self.model = IsoCapsuleNet(hyper_parameters)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(sorted(classes))}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def forward(self, X, y=None):
        features = self.model(X, y=None)
        return features

    def train_model(self, train_loader, test_loader):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_parameters['lr'], weight_decay=self.hyper_parameters['weight_decay'])
        accuracy = EvaluateAcc('', '')
        accuracy.weighted_tag=True
        loss_function = CapsuleLoss()

        for epoch in range(self.hyper_parameters['max_epoch']):
            for iteration, (X, y) in enumerate(train_loader):
                t_epoch_begin = time.time()
                self.train()
                optimizer.zero_grad()

                y_true = torch.eye(self.hyper_parameters['nclass']).index_select(dim=0, index=torch.LongTensor(y))
                y_pred, reconstructions = self.forward(X, y_true)
                loss_train = loss_function(X, y_true, y_pred, reconstructions)

                accuracy.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                acc_train = accuracy.evaluate()

                loss_train.backward()
                optimizer.step()

                self.test_model(test_loader, print_result=False)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'Iteration: {:04d}'.format(iteration + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: ', acc_train,
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!", "Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    def test_model(self, test_loader, print_result=False):
        t_begin = time.time()
        self.eval()

        accuracy = EvaluateAcc('', '')
        accuracy.weighted_tag = True

        y_pred_list = []
        y_true_list = []
        for iteration, (X, y_true) in enumerate(test_loader):
            t_iteration_begin = time.time()
            y_pred, reconstruction = self.forward(X)
            y_pred_list.extend(y_pred.max(1)[1])
            y_true_list.extend(y_true)
            accuracy.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            acc_test = accuracy.evaluate()
            if print_result:
                print('Iteration: {:04d}'.format(iteration + 1),
                      'acc_test: ', acc_test,
                      'time: {:.4f}s'.format(time.time() - t_iteration_begin))

        accuracy.data = {'true_y': y_true_list, 'pred_y': y_pred_list}
        acc_test = accuracy.evaluate()
        if print_result:
            print('Overall Testing Accuracy: ', acc_test,
                  'Overall Time: {:.4f}s'.format(time.time() - t_begin))
        return y_pred_list, y_true_list

    def run(self):
        print('***** start training *****')
        self.train_model(self.data['train'], self.data['test'])
        print('***** start testing *****')
        y_pred, y_true = self.test_model(self.data['test'], print_result=True)
        print('Prediction Results & Ground-Truth:')
        print({'pred_y': y_pred, 'true_y': y_true})
        return {'pred_y': y_pred, 'true_y': y_true}