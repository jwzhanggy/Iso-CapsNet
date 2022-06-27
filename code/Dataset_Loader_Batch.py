'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


from code.base_class.dataset import dataset
from typing import Dict, Union
import pickle
import torch
import numpy as np

class Brain_Graph_Dataset(torch.utils.data.Dataset):
    data = None
    hyper_parameters = None

    def __init__(self, data, hyper_parameters):
        self.data = data
        self.hyper_parameters = hyper_parameters
        self.size = len(self.data['y'])

    def __len__(self) -> int:
        return self.size

    def adjust_y(self, label):
        if label == -1:
            return 0
        else:
            return label

    def __getitem__(self, index: int) -> Dict[str, Union[int, np.ndarray]]:
        return (
            torch.reshape(torch.FloatTensor(np.array(self.data['X'][index])), (-1, self.hyper_parameters['num_node'], self.hyper_parameters['num_node'])),
            self.adjust_y(self.data['y'][index]),
        )

class DatasetLoader(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    load_type = 'Processed'
    hyper_parameters = None

    def __init__(self, dName=None, dDescription=None, hyper_parameters=None):
        super(DatasetLoader, self).__init__(dName, dDescription)
        self.hyper_parameters = hyper_parameters

    def load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        train_loader = torch.utils.data.DataLoader(
            Brain_Graph_Dataset(data['train'], self.hyper_parameters),
            batch_size=min(self.hyper_parameters['batch_size'], len(data['train']['X'])),
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            Brain_Graph_Dataset(data['test'], self.hyper_parameters),
            batch_size=min(self.hyper_parameters['batch_size'], len(data['test']['X'])),
            shuffle=True,
        )
        return {'train': train_loader, 'test': test_loader}
