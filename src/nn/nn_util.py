import os
import sys
import pickle
import numpy as np

sys.path.insert(0, "../mcmc")
sys.path.insert(0, "deeponet")
sys.path.insert(0, "pcanet")
sys.path.insert(0, "fno")

from surrogateModel import SurrogateModel, SurrogateModelFNO
from load_data_and_deeponet import load_data_and_deeponet
from load_data_and_fno import load_data_and_fno
from load_data_and_pcanet import load_data_and_pcanet

def get_surrogate_specs(name, data_prefix, model_path):
    data_file, model_file, load_data_and_model = None, None, None

    if name == "DeepONet":
        data_file = os.path.join(model_path, "data", f"{data_prefix}_samples.npz")
        model_file = os.path.join(model_path, name, "Results", "model.pkl")
        load_data_and_model = load_data_and_deeponet
    elif name == "PCANet":
        data_file = os.path.join(model_path, "data", f"{data_prefix}_samples.npz")
        model_file = os.path.join(model_path, name, "Results", "model.pkl")
        load_data_and_model = load_data_and_pcanet
    elif name == "FNO":     
        data_file = os.path.join(model_path, "data", f"{data_prefix}_FNO_samples.npz")
        model_file = os.path.join(model_path, name, "Results", "model.pkl")
        load_data_and_model = load_data_and_fno

    return data_file, model_file, load_data_and_model

def load_data_and_model(name, data_prefix, model_path):
    
    df, mf, ldam = get_surrogate_specs(name, data_prefix, model_path)
    
    missing = []
    if not os.path.isfile(df):
        missing.append(('data', df))
    if not os.path.isfile(mf):
        missing.append(('model', mf))
    if missing:
        for kind, path in missing:
            print('{} model not loaded; missing {} file: {}'.format(name, kind, path))
        return None

    return ldam(df, mf)

def load_surrogate_model(name, data_prefix, model_path, model, u_comps=None):
    loaded = load_data_and_model(name, data_prefix, model_path)
    if loaded is None:
        return None
    data, nn = loaded

    if name == 'FNO':
        u_nodes = model.u_nodes
        grid_x = data.grid_x_test[0, :, :, 0]
        grid_y = data.grid_y_test[0, :, :, 0]
        if u_comps is None:
            raise Exception('u_comps is not set')
        nn_surrogate = SurrogateModelFNO(model, nn, data, u_nodes, grid_x, grid_y, u_comps)
    else:
        nn_surrogate = SurrogateModel(model, nn, data)

    print('{} nn_data: '.format(name), data)
    print('{} nn_model: '.format(name), nn)
    return nn_surrogate
