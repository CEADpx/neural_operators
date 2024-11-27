import sys
import torch

from torch_pcanet import PCANet
src_path = '../../'
sys.path.append(src_path + 'data/')
from dataMethods import DataProcessor

def load_data_and_pcanet(data_path, model_path):
    
    nn = torch.load(model_path)

    # load data (it is needed for encoding and decoding of new inputs and outputs)
    data = DataProcessor(data_path, \
                        nn.metadata['num_train'], \
                        nn.metadata['num_test'], \
                        nn.metadata['num_inp_fn_points'], \
                        nn.metadata['num_out_fn_points'], \
                        nn.metadata['num_Y_components'], \
                        nn.metadata['num_inp_red_dim'], \
                        nn.metadata['num_out_red_dim'])

    return data, nn