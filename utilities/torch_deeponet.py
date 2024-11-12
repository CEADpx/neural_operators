import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataMethods import DataHandler, DataProcessor
from torch_mlp import MLP

class DeepONet(nn.Module):
    
    def __init__(self, num_layers, num_neurons, act, \
                 num_br_outputs, num_tr_outputs, \
                 num_inp_fn_points, \
                 out_coordinate_dimension, \
                 num_Y_components):

        super(DeepONet, self).__init__()

        self.num_inp_fn_points = num_inp_fn_points
        self.num_br_outputs = num_br_outputs
        self.num_tr_outputs = num_tr_outputs
        self.out_coordinate_dimension = out_coordinate_dimension
        self.num_Y_components = num_Y_components

        # creating the branch network
        self.branch_net = MLP(input_size=num_inp_fn_points, \
                              hidden_size=num_neurons, \
                              num_classes=num_br_outputs, \
                              depth=num_layers, \
                              act=act)
        self.branch_net.float()

        # creating the trunk network
        self.trunk_net = MLP(input_size=out_coordinate_dimension, \
                             hidden_size=num_neurons, \
                             num_classes=num_tr_outputs, \
                             depth=num_layers, \
                             act=act)
        self.trunk_net.float()
        
        self.bias = [nn.Parameter(torch.ones((1,)),requires_grad=True) for i in range(num_Y_components)]

        self.num_Y_components = num_Y_components

        # Logger
        self.train_loss_log = []
        self.test_loss_log = []
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    
    def forward(self, batch):

        X_train = self.convert_np_to_tensor(batch['X_train'])
        X_trunk = self.convert_np_to_tensor(batch['X_trunk'])
        
        branch_out = self.branch_net.forward(X_train)
        trunk_out = self.trunk_net.forward(X_trunk,final_act=True)

        if self.num_Y_components > 1:
            # for multiple output components we need to split the output of the branch network
            output = []
            for i in range(self.num_Y_components):
                output.append(branch_out[:,i*self.num_tr_outputs:(i+1)*self.num_tr_outputs] @ trunk_out.t() + self.bias[i])
            
            # stack and reshape to have output in batch_size x output_components format
            output = torch.stack(output, dim=-1)
            output = output.reshape(-1, X_trunk.shape[0] * self.num_Y_components)

            # output_1 = branch_out[:,:self.num_tr_outputs] @ trunk_out.t() + self.bias[0]
            # output_2 = branch_out[:,self.num_tr_outputs:] @ trunk_out.t() + self.bias[1]
            # output = torch.stack((output_1, output_2), dim=-1)
            # output = output.reshape(-1, X_trunk.shape[0] * X_trunk.shape[1])
        else:
            output = branch_out @ trunk_out.t() + self.bias[0]

        return output
    
    def train(self, train_data, test_data, batch_size=32, epochs = 1000, \
              lr=0.001, log=True, \
              loss_print_freq = 100):

        self.epochs = epochs
        self.batch_size = batch_size

        dataset = DataHandler(train_data['X_train'], \
                              train_data['X_trunk'], train_data['Y_train'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        X_trunk = dataset.X_trunk

        test_data_tensor = \
            {'X_train': self.convert_np_to_tensor(test_data['X_train']), \
             'X_trunk': self.convert_np_to_tensor(test_data['X_trunk']), \
             'Y_train': self.convert_np_to_tensor(test_data['Y_train']) }

        # MSE loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train_loss_log = np.zeros((epochs, 1))
        self.test_loss_log = np.zeros((epochs, 1))

        # training loop
        for epoch in range(epochs):
            
            train_losses = []
            test_losses = []

            start_time = time.perf_counter()

            for X_train, _, Y_train in dataloader:

                # print(X_train.shape, X_trunk.shape, Y_train.shape)
                batch = {'X_train': X_train, 'X_trunk': X_trunk, 'Y_train': Y_train}
                
                # removing previous gradients
                optimizer.zero_grad()

                # forward pass through model
                Y_train_pred = self.forward(batch)

                # compute loss
                loss = criterion(Y_train_pred, Y_train)

                # backward pass
                loss.backward()

                # calculate avg loss across batches
                train_losses.append(loss.item())

                # update parameters
                optimizer.step()

                # compute test loss
                Y_test_pred = self.forward(test_data_tensor)
                test_loss = criterion(Y_test_pred, test_data_tensor['Y_train']).item()
                test_losses.append(test_loss)

            end_time = time.perf_counter()
            epoch_time = end_time - start_time

            self.train_loss_log[epoch, 0] = np.mean(train_losses)
            self.test_loss_log[epoch, 0] = np.mean(test_losses)

            if log == True and (epoch % loss_print_freq == 0 or epoch == epochs - 1):
                print('='*30)
                print('Epoch: {:5d}, Train Loss (l2 squared): {:.3e}, Test Loss (l2 squared): {:.3e}, Time (sec): {:.3f}'.format(epoch, \
                                                  np.mean(train_losses), \
                                                  np.mean(test_losses), \
                                                  epoch_time))
                print('='*30)
    
    def predict(self, test_data):
        test_data_tensor = \
            {'X_train': self.convert_np_to_tensor(test_data['X_train']), \
             'X_trunk': self.convert_np_to_tensor(test_data['X_trunk']), \
             'Y_train': self.convert_np_to_tensor(test_data['Y_train']) }
        
        return self.forward(test_data_tensor)