import numpy as np
import torch
from torch.utils.data import Dataset

class DataHandler(Dataset):
    def __init__(self, X_train_, X_trunk_, Y_train_, convert_to_tensor=True):

        self.X_train = self.convert_np_to_tensor(X_train_) if convert_to_tensor else X_train_

        if X_trunk_ is None:
            self.X_trunk = None
        else:
            self.X_trunk = self.convert_np_to_tensor(X_trunk_) if convert_to_tensor else X_trunk_
        
        self.Y_train = self.convert_np_to_tensor(Y_train_) if convert_to_tensor else Y_train_

    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array
    
    def __len__(self):
        return len(self.Y_train)  # Assuming X_train and X_trunk have the same length as y

    def __getitem__(self, index):
        if self.X_trunk is None:
            return self.X_train[index,:], self.Y_train[index,:]
        else:
            return self.X_train[index,:], self.X_trunk, self.Y_train[index,:]
        
class DataProcessor:
    def __init__(self, data_file_name = '../problems/poisson/data/Poisson_samples.npz', \
                 num_train = 1900, num_test = 100, \
                 num_inp_fn_points = 2601, num_out_fn_points = 2601, \
                 num_Y_components = 1, \
                 num_inp_red_dim = None, num_out_red_dim = None):
        
        self.data = np.load(data_file_name)
        self.num_train = num_train
        self.num_test = num_test
        self.num_inp_fn_points = num_inp_fn_points
        self.num_out_fn_points = num_out_fn_points
        self.num_Y_components = num_Y_components 
        self.num_inp_red_dim = num_inp_red_dim
        self.num_out_red_dim = num_out_red_dim

        self.data, self.X_trunk, self.X_train, self.X_test, \
            self.X_train_mean, self.X_train_std, \
            self.X_train_svd_projector, self.X_train_s_values, \
            self.tol = self.load_X_data(self.data)
        
        self.Y_train, self.Y_test, \
            self.Y_train_mean, self.Y_train_std, \
            self.Y_train_svd_projector, self.Y_train_s_values \
            = self.load_Y_data(self.data)

        self.X_trunk_min = np.min(self.X_trunk, axis = 0)
        self.X_trunk_max = np.max(self.X_trunk, axis = 0)

    def load_X_data(self, data, tol = 1.0e-9):

        # trunk input data ('xi' coordinates)
        X_trunk = data['u_mesh_nodes']
        
        # branch input data ('m' functions)
        X_train = data['m_samples'][:self.num_train,:]
        X_test = data['m_samples'][self.num_train:(self.num_train + self.num_test),:]

        X_train_mean = np.mean(X_train, 0)
        X_train_std = np.std(X_train, 0)

        X_train = (X_train - X_train_mean)/(X_train_std + tol)
        X_test = (X_test - X_train_mean)/(X_train_std + tol)

        if self.num_inp_red_dim is not None:
            # compute SVD of input data 
            X_train_svd_projector, X_train_s_values = self.compute_svd(X_train, self.num_inp_red_dim, is_data_centered = True)

            # define training and testing data in the reduced dimension
            X_train = np.dot(X_train, X_train_svd_projector.T)
            X_test = np.dot(X_test, X_train_svd_projector.T)
        else:
            X_train_svd_projector = None
            X_train_s_values = None

        return data, X_trunk, X_train, X_test, \
               X_train_mean, X_train_std, \
               X_train_svd_projector, X_train_s_values, \
               tol 
    
    def load_Y_data(self, data, tol = 1.0e-9):

        # trunk input data ('xi' coordinates)
        X_trunk = data['u_mesh_nodes']
        
        # output data ('u' functions)
        Y_train = data['u_samples'][:self.num_train,:]
        Y_test = data['u_samples'][self.num_train:(self.num_train + self.num_test),:]

        if self.num_out_fn_points * self.num_Y_components != Y_train.shape[1]:
            raise ValueError('num_out_fn_points does not match the number of output function points in the data')
        
        Y_train_mean = np.mean(Y_train, 0)
        Y_train_std = np.std(Y_train, 0)

        Y_train = (Y_train - Y_train_mean)/(Y_train_std + tol)
        Y_test = (Y_test - Y_train_mean)/(Y_train_std + tol)

        if self.num_out_red_dim is not None:
            # compute SVD of output data 
            Y_train_svd_projector, Y_train_s_values = self.compute_svd(Y_train, self.num_out_red_dim, is_data_centered = True)

            # define training and testing data in the reduced dimension
            Y_train = np.dot(Y_train, Y_train_svd_projector.T)
            Y_test = np.dot(Y_test, Y_train_svd_projector.T)
        else:
            Y_train_svd_projector = None
            Y_train_s_values = None
        
        return Y_train, Y_test, \
               Y_train_mean, Y_train_std, \
               Y_train_svd_projector, Y_train_s_values
        
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