import sys
import torch

from torch_fno2d import FNO2D
src_path = '../../'
sys.path.append(src_path + 'data/')
from dataMethods import DataProcessorFNO

def load_data_and_fno(data_path, model_path):
    
    nn = torch.load(model_path, weights_only=False)

    # load data (it is needed for encoding and decoding of new inputs and outputs)
    data = DataProcessorFNO(data_path, \
                        nn.metadata['num_train'], \
                        nn.metadata['num_test'], \
                        nn.metadata['num_Y_components'], \
                        nn.metadata['coarsen_grid_factor'])

    return data, nn