import numpy as np
import torch
from torch.utils.data import Dataset

class DataHandler(Dataset):

    def __init__(self, X_train_, X_trunk_, Y_train_, \
                 convert_to_tensor=True):

        self.X_train = self.convert_np_to_tensor(X_train_) if convert_to_tensor else X_train_

        if X_trunk_ is None:
            self.X_trunk = None
        else:
            self.X_trunk = self.convert_np_to_tensor(X_trunk_) if convert_to_tensor else X_trunk_
        
        self.Y_train = self.convert_np_to_tensor(Y_train_) if convert_to_tensor else Y_train_

    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array
    
    def __len__(self):
        return len(self.Y_train) 

    def __getitem__(self, index):
        if self.X_trunk is None:
            return self.X_train[index,:], self.Y_train[index,:]
        else:
            return self.X_train[index,:], self.X_trunk, self.Y_train[index,:]
        
class DataProcessor:

    def __init__(self, data_file_name \
                        = '../problems/poisson/data/Poisson_samples.npz', \
                 num_train = 1900, num_test = 100, \
                 num_inp_fn_points = 2601, num_out_fn_points = 2601, \
                 num_Y_components = 1, \
                 num_inp_red_dim = None, \
                 num_out_red_dim = None):
        
        # load data from file
        self.data = np.load(data_file_name)

        self.num_train = num_train
        self.num_test = num_test
        self.num_inp_fn_points = num_inp_fn_points
        self.num_out_fn_points = num_out_fn_points
        self.num_Y_components = num_Y_components 
        self.num_inp_red_dim = num_inp_red_dim
        self.num_out_red_dim = num_out_red_dim
        self.tol = 1.0e-9

        self.load_X_data(self.data)
        self.load_Y_data(self.data)
            
    def get_data_to_save(self):
        data_to_save = {}
        data_to_save['num_train'] = self.num_train
        data_to_save['num_test'] = self.num_test
        data_to_save['num_inp_fn_points'] = self.num_inp_fn_points
        data_to_save['num_out_fn_points'] = self.num_out_fn_points
        data_to_save['num_Y_components'] = self.num_Y_components
        data_to_save['num_inp_red_dim'] = self.num_inp_red_dim
        data_to_save['num_out_red_dim'] = self.num_out_red_dim

        data_to_save['X_trunk'] = self.X_trunk
        data_to_save['X_trunk_min'] = self.X_trunk_min
        data_to_save['X_trunk_max'] = self.X_trunk_max
        data_to_save['X_train'] = self.X_train
        data_to_save['X_test'] = self.X_test
        data_to_save['X_train_mean'] = self.X_train_mean
        data_to_save['X_train_std'] = self.X_train_std
        data_to_save['X_train_svd_projector'] = self.X_train_svd_projector
        data_to_save['X_train_s_values'] = self.X_train_s_values
        data_to_save['Y_train'] = self.Y_train
        data_to_save['Y_test'] = self.Y_test
        data_to_save['Y_train_mean'] = self.Y_train_mean
        data_to_save['Y_train_std'] = self.Y_train_std
        data_to_save['Y_train_svd_projector'] = self.Y_train_svd_projector
        data_to_save['Y_train_s_values'] = self.Y_train_s_values
        data_to_save['u_mesh_dirichlet_boundary_nodes'] = self.u_mesh_dirichlet_boundary_nodes
        return data_to_save

    def load_X_data(self, data):

        # trunk input data ('xi' coordinates)
        self.X_trunk = data['u_mesh_nodes']
        self.X_trunk_min = np.min(self.X_trunk, axis = 0)
        self.X_trunk_max = np.max(self.X_trunk, axis = 0)
        
        # branch input data ('m' functions)
        self.X_train = data['m_samples'][:self.num_train,:]
        self.X_test = data['m_samples'][self.num_train:(self.num_train + self.num_test),:]

        self.X_train_mean = np.mean(self.X_train, 0)
        self.X_train_std = np.std(self.X_train, 0)

        self.X_train = (self.X_train - self.X_train_mean)/(self.X_train_std + self.tol)
        self.X_test = (self.X_test - self.X_train_mean)/(self.X_train_std + self.tol)

        if self.num_inp_red_dim is not None:
            # compute SVD of input data 
            self.X_train_svd_projector, self.X_train_s_values = self.compute_svd(self.X_train, self.num_inp_red_dim, is_data_centered = True)

            # define training and testing data in the reduced dimension
            self.X_train = np.dot(self.X_train, self.X_train_svd_projector.T)
            self.X_test = np.dot(self.X_test, self.X_train_svd_projector.T)
        else:
            self.X_train_svd_projector = None
            self.X_train_s_values = None
    
    def load_Y_data(self, data):

        # output data ('u' functions)
        self.Y_train = data['u_samples'][:self.num_train,:]
        self.Y_test = data['u_samples'][self.num_train:(self.num_train + self.num_test),:]

        if self.num_out_fn_points * self.num_Y_components != self.Y_train.shape[1]:
            raise ValueError('num_out_fn_points does not match the number of output function points in the data')
        
        self.Y_train_mean = np.mean(self.Y_train, 0)
        self.Y_train_std = np.std(self.Y_train, 0)

        self.Y_train = (self.Y_train - self.Y_train_mean)/(self.Y_train_std + self.tol)
        self.Y_test = (self.Y_test - self.Y_train_mean)/(self.Y_train_std + self.tol)

        if self.num_out_red_dim is not None:
            # compute SVD of output data 
            self.Y_train_svd_projector, self.Y_train_s_values = self.compute_svd(self.Y_train, self.num_out_red_dim, is_data_centered = True)

            # define training and testing data in the reduced dimension
            self.Y_train = np.dot(self.Y_train, self.Y_train_svd_projector.T)
            self.Y_test = np.dot(self.Y_test, self.Y_train_svd_projector.T)
        else:
            self.Y_train_svd_projector = None
            self.Y_train_s_values = None

        # read indices corresponding to the Dirichlet boundary conditions
        self.u_mesh_dirichlet_boundary_nodes = data['u_mesh_dirichlet_boundary_nodes']
        
    def encoder_Y(self, x):
        x = (x - self.Y_train_mean)/(self.Y_train_std + self.tol)
        if self.Y_train_svd_projector is not None:
            return self.project_SVD(x, self.Y_train_svd_projector)
        else:
            return x
    
    def decoder_Y(self, x):
        # first lift the data to the original dimension
        if self.Y_train_svd_projector is not None:
            x = self.lift_SVD(x, self.Y_train_svd_projector)

        x = x*(self.Y_train_std + self.tol) + self.Y_train_mean
        return x
    
    def encoder_X(self, x):
        x = (x - self.X_train_mean)/(self.X_train_std + self.tol)
        if self.X_train_svd_projector is not None:
            return self.project_SVD(x, self.X_train_svd_projector)
        else:
            return x
    
    def decoder_X(self, x):
        # first lift the data to the original dimension
        if self.X_train_svd_projector is not None:
            x = self.lift_SVD(x, self.X_train_svd_projector)

        x = x*(self.X_train_std + self.tol) + self.X_train_mean
        return x
    
    def compute_svd(self, data, num_red_dim, is_data_centered = False):
        if is_data_centered == False:
            data_mean = np.mean(data, 0)
            data = data - data_mean
        U, S, _ = np.linalg.svd(data.T, full_matrices = False)
        projector = U[:, :num_red_dim].T # size num_red_dim x dim(X_train[0])
        return projector, S
    
    def project_SVD(self, data, Pi):
        return np.dot(data, Pi.T)
    
    def lift_SVD(self, data, Pi):
        return np.dot(data, Pi)
    

class DataProcessorTF(DataProcessor):
    def __init__(self, batch_size = 100, \
                 data_file_name = \
                    '../problems/poisson/data/Poisson_samples.npz', \
                 num_train = 1900, num_test = 100, \
                 num_inp_fn_points = 2601, num_out_fn_points = 2601, \
                 num_Y_components = 1, \
                 num_inp_red_dim = None, num_out_red_dim = None):
        
        self.batch_size = batch_size
        super().__init__(data_file_name, num_train, num_test, \
                         num_inp_fn_points, num_out_fn_points, \
                         num_Y_components, num_inp_red_dim, num_out_red_dim)
        
        # reshaping data to be compatible with TensorFlow
        ## X_train data
        self.X_train = self.X_train.reshape(-1, 1, self.num_inp_fn_points)  
        self.X_test = self.X_test.reshape(-1, 1, self.num_inp_fn_points)
        self.X_train_mean = self.X_train_mean.reshape(1, 1, self.num_inp_fn_points)
        self.X_train_std = self.X_train_std.reshape(1, 1, self.num_inp_fn_points)

        ## Y_train data
        self.Y_train = self.Y_train.reshape(-1, self.num_out_fn_points * self.num_Y_components, 1)
        self.Y_test = self.Y_test.reshape(-1, self.num_out_fn_points * self.num_Y_components, 1)
        self.Y_train_mean = self.Y_train_mean.reshape(1, self.num_out_fn_points * self.num_Y_components, 1)
        self.Y_train_std = self.Y_train_std.reshape(1, self.num_out_fn_points * self.num_Y_components, 1)

    def get_data_to_save(self):
        data_to_save = super().get_data_to_save()
        data_to_save['batch_size'] = self.batch_size
        return data_to_save
        
    def encoder_Y(self, x):
        x = (x - self.Y_train_mean)/(self.Y_train_std + self.tol)
        if self.Y_train_svd_projector is not None:
            x = self.project_SVD(x[:, :, 0], self.Y_train_svd_projector)
            return x.reshape(x.shape[0], x.shape[1], 1)
        else:
            return x
    
    def decoder_Y(self, x):
        # first lift the data to the original dimension
        if self.Y_train_svd_projector is not None:
            x = self.lift_SVD(x[:, :, 0], self.Y_train_svd_projector)
            x = x.reshape(x.shape[0], x.shape[1], 1)

        x = x*(self.Y_train_std + self.tol) + self.Y_train_mean
        return x
    
    def encoder_X(self, x):
        x = (x - self.X_train_mean)/(self.X_train_std + self.tol)
        if self.X_train_svd_projector is not None:
            x = self.project_SVD(x[:, 0, :], self.X_train_svd_projector)
            return x.reshape(x.shape[0], 1, x.shape[1])
        else:
            return x
    
    def decoder_X(self, x):
        # first lift the data to the original dimension
        if self.X_train_svd_projector is not None:
            x = self.lift_SVD(x[:, 0, :], self.X_train_svd_projector)
            x = x.reshape(x.shape[0], 1, x.shape[1])

        x = x*(self.X_train_std + self.tol) + self.X_train_mean
        return x
    
    def minibatch(self):

        batch_id = np.random.choice(self.X_train.shape[0], self.batch_size, replace=False)

        X_train = [self.X_train[i:i+1] for i in batch_id]
        X_train = np.concatenate(X_train, axis=0)
        Y_train = [self.Y_train[i:i+1] for i in batch_id]
        Y_train = np.concatenate(Y_train, axis=0)

        X_trunk_train = self.X_trunk
        X_trunk_min = self.X_trunk_min
        X_trunk_max = self.X_trunk_max

        return X_train, X_trunk_train, Y_train, X_trunk_min, X_trunk_max

    def testbatch(self, num_test):
        batch_id = np.arange(num_test)
        X_test = [self.X_test[i:i+1] for i in batch_id]
        X_test = np.concatenate(X_test, axis=0)
        Y_test = [self.Y_test[i:i+1] for i in batch_id]
        Y_test = np.concatenate(Y_test, axis=0)
        X_trunk_test = self.X_trunk

        return X_test, X_trunk_test, Y_test
    

class DataProcessorFNO:
    def __init__(self, data_file_name \
                        = '../problems/poisson/data/Poisson_FNO_samples.npz', \
                 num_train = 1900, num_test = 100, \
                 num_Y_components = 1, \
                 coarsen_grid_factor = 2):
        
        # load data from file
        self.data = np.load(data_file_name)
        self.tol = 1.0e-9
        
        self.num_train = num_train
        self.num_test = num_test
        self.num_Y_components = num_Y_components 
        self.coarsen_grid_factor = coarsen_grid_factor
        
        self.load_X_data(self.data)
        self.load_Y_data(self.data)

    def get_data_to_save(self):
        data_to_save = {}
        data_to_save['num_train'] = self.num_train
        data_to_save['num_test'] = self.num_test
        data_to_save['num_Y_components'] = self.num_Y_components
        data_to_save['coarsen_grid_factor'] = self.coarsen_grid_factor

        data_to_save['num_grid_x'] = self.num_grid_x
        data_to_save['num_grid_y'] = self.num_grid_y
        data_to_save['grid_x_train'] = self.grid_x_train
        data_to_save['grid_y_train'] = self.grid_y_train
        data_to_save['grid_x_test'] = self.grid_x_test
        data_to_save['grid_y_test'] = self.grid_y_test
        data_to_save['X_train'] = self.X_train
        data_to_save['X_test'] = self.X_test
        data_to_save['X_train_mean'] = self.X_train_mean
        data_to_save['X_train_std'] = self.X_train_std
        data_to_save['Y_train'] = self.Y_train
        data_to_save['Y_test'] = self.Y_test
        data_to_save['Y_train_mean'] = self.Y_train_mean
        data_to_save['Y_train_std'] = self.Y_train_std
        data_to_save['u_grid_dirichlet_boundary_nodes'] = self.u_grid_dirichlet_boundary_nodes

        return data_to_save

    def load_X_data(self, data, tol = 1.0e-9):

        # grid coordinates data
        # we select every coarsen_grid_factor-th point so that we can coarsen the grid
        grid_x = data['grid_x'][::self.coarsen_grid_factor, ::self.coarsen_grid_factor]
        grid_y = data['grid_y'][::self.coarsen_grid_factor, ::self.coarsen_grid_factor]

        self.num_grid_x = grid_x.shape[0]
        self.num_grid_y = grid_x.shape[1]

        # grid coordinates data
        self.grid_x_train = np.tile(grid_x, \
                               (self.num_train, 1, 1)\
                               ).reshape(self.num_train, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        self.grid_y_train = np.tile(grid_y, \
                               (self.num_train, 1, 1)\
                               ).reshape(self.num_train, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        self.grid_x_test = np.tile(grid_x, \
                               (self.num_test, 1, 1)\
                               ).reshape(self.num_test, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        self.grid_y_test = np.tile(grid_y, \
                               (self.num_test, 1, 1)\
                               ).reshape(self.num_test, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        # branch input data ('m' functions)
        self.X_train = data['grid_m_samples'][:self.num_train]
        self.X_train = self.X_train[:, ::self.coarsen_grid_factor, ::self.coarsen_grid_factor]
        self.X_train = self.X_train.reshape(self.num_train, self.num_grid_x, self.num_grid_y, 1)

        self.X_test = data['grid_m_samples'][self.num_train:(self.num_train + self.num_test)]
        self.X_test = self.X_test[:, ::self.coarsen_grid_factor, ::self.coarsen_grid_factor]
        self.X_test = self.X_test.reshape(self.num_test, self.num_grid_x, self.num_grid_y, 1)

        self.X_train_mean = np.mean(self.X_train, 0)
        self.X_train_std = np.std(self.X_train, 0)

        # center and scale data
        self.X_train = (self.X_train - self.X_train_mean)/(self.X_train_std + tol)
        self.X_test = (self.X_test - self.X_train_mean)/(self.X_train_std + tol)

        # combine grid coordinates and function m values
        self.X_train = np.concatenate((self.X_train, self.grid_x_train, self.grid_y_train), axis = -1)
        self.X_test = np.concatenate((self.X_test, self.grid_x_test, self.grid_y_test), axis = -1)

        self.X_train = torch.from_numpy(self.X_train).to(torch.float32)
        self.X_test = torch.from_numpy(self.X_test).to(torch.float32)
    
    def load_Y_data(self, data, tol = 1.0e-9):

        # output data ('u' functions)
        self.Y_train = data['grid_u_samples'][:self.num_train]
        self.Y_train = self.Y_train[:, ::self.coarsen_grid_factor, ::self.coarsen_grid_factor]
        self.Y_train = self.Y_train.reshape(self.num_train, self.num_grid_x, self.num_grid_y, self.num_Y_components)

        self.Y_test = data['grid_u_samples'][self.num_train:(self.num_train + self.num_test)]
        self.Y_test = self.Y_test[:, ::self.coarsen_grid_factor, ::self.coarsen_grid_factor]
        self.Y_test = self.Y_test.reshape(self.num_test, self.num_grid_x, self.num_grid_y, self.num_Y_components)

        self.Y_train_mean = np.mean(self.Y_train, 0)
        self.Y_train_std = np.std(self.Y_train, 0)

        # center and scale data
        self.Y_train = (self.Y_train - self.Y_train_mean)/(self.Y_train_std + tol)
        self.Y_test = (self.Y_test - self.Y_train_mean)/(self.Y_train_std + tol)

        self.Y_train = torch.from_numpy(self.Y_train).to(torch.float32)
        self.Y_test = torch.from_numpy(self.Y_test).to(torch.float32)

        # read indices corresponding to the Dirichlet boundary conditions
        self.u_grid_dirichlet_boundary_nodes = data['u_grid_dirichlet_boundary_nodes']
        
    def encoder_Y(self, x):
        return (x - self.Y_train_mean)/(self.Y_train_std + self.tol)
    
    def decoder_Y(self, x):
        return x*(self.Y_train_std + self.tol) + self.Y_train_mean
    
    def encoder_X(self, x):
        return (x - self.X_train_mean)/(self.X_train_std + self.tol)
            
    def decoder_X(self, x):
        return x*(self.X_train_std + self.tol) + self.X_train_mean