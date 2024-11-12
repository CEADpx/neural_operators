import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataMethods import DataHandler
from torch_mlp import MLP



class PCANet(nn.Module):
    
    def __init__(self, num_layers, num_neurons, act, \
                 num_inp_red_dim, num_out_red_dim):

        super(PCANet, self).__init__()

        # creating the branch network
        self.net = MLP(input_size=num_inp_red_dim, \
                              hidden_size=num_neurons, \
                              num_classes=num_out_red_dim, \
                              depth=num_layers, \
                              act=act)
        self.net.float()

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
        return self.net.forward(X_train)
    
    def train(self, train_data, test_data, batch_size=32, epochs = 1000, \
              lr=0.001, log=True, \
              loss_print_freq = 100):

        self.epochs = epochs
        self.batch_size = batch_size

        dataset = DataHandler(train_data['X_train'], None, train_data['Y_train'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        test_data_tensor = \
            {'X_train': self.convert_np_to_tensor(test_data['X_train']), \
             'Y_train': self.convert_np_to_tensor(test_data['Y_train']) }

        # MSE loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train_loss_log = np.zeros((epochs, 1))
        self.test_loss_log = np.zeros((epochs, 1))

        # Main training loop
        for epoch in range(epochs):
            
            train_losses = []
            test_losses = []

            start_time = time.perf_counter()

            for X_train, Y_train in dataloader:

                # print(X_train.shape, X_trunk.shape, Y_train.shape)
                batch = {'X_train': X_train, 'Y_train': Y_train}
                
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
             'Y_train': self.convert_np_to_tensor(test_data['Y_train']) }
        
        return self.forward(test_data_tensor)