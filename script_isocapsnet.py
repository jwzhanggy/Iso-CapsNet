from code.Dataset_Loader_Batch import DatasetLoader
from code.Method_IsoCapsNet import Method_IsoCapsNet
from code.Result_Saving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch
import warnings
import random



#---- CapsNet on MNIST ----

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
warnings.filterwarnings("ignore", category=UserWarning)

# HIV_fMRI
# BP_fMRI
# HIV_DTI
# ADHD_fMRI

dataset_name = 'HIV_fMRI'

graph_size_dict = {
    'HIV_fMRI': 90,
    'BP_fMRI': 82,
    'HIV_DTI': 90,
    'ADHD_fMRI': 116
}

if 1:
    for fold in range(1, 4):
        #---- hyper-parameters ----
        hyper_parameters = {
            'k': 4,
            'lr': 1e-2,
            'weight_decay': 5e-4,
            'max_epoch': 17,
            'nclass': 2,
            'num_node': graph_size_dict[dataset_name],
            'batch_size': 64,
        }
        hyper_parameters['nfeature'] = (hyper_parameters['num_node'] - hyper_parameters['k'] + 1)**2
        # --------------------------

        print('************ Start Fold: ' + str(fold) + ' ************')
        # ---- objection initialization section ---------------
        data_obj = DatasetLoader(dataset_name, '')
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_source_file_name = 'fold_' + str(fold)
        data_obj.hyper_parameters = hyper_parameters

        method_obj = Method_IsoCapsNet('IsoCapsNet', '', hyper_parameters)

        result_obj = ResultSaving('saver', '')
        result_obj.result_destination_folder_path = './result/' + dataset_name + '/'
        result_obj.result_destination_file_name = 'fold_' + str(fold)

        setting_obj = Settings('regular setting', '')

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
#------------------------------------


# HIV_fMRI
# 0.8695652173913043, 0.8531139835487661
# 0.7727272727272727, 0.7879539815023686
# 0.7727272727272727, 0.7633477633477633