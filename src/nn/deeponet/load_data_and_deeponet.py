import sys
import torch

from torch_deeponet import DeepONet
src_path = '../../'
sys.path.append(src_path + 'data/')
from dataMethods import DataProcessor

def load_data_and_deeponet(data_path, model_path):
    
    nn = torch.load(model_path, weights_only=False)

    # load data (it is needed for encoding and decoding of new inputs and outputs)
    data = DataProcessor(data_path, \
                        nn.metadata['num_train'], \
                        nn.metadata['num_test'], \
                        nn.metadata['num_inp_fn_points'], \
                        nn.metadata['num_out_fn_points'], \
                        nn.metadata['num_Y_components'])

    return data, nn