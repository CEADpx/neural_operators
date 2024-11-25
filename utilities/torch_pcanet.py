import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local utility methods
from dataMethods import DataHandler
from torch_mlp import MLP

class PCANet(nn.Module):
    
    def __init__(self, num_layers, num_neurons, act, \
                 num_inp_red_dim, num_out_red_dim, save_file=None):

        super(PCANet, self).__init__()
        self.name = 'PCANet'
        self.save_file = save_file
        if save_file is None:
            self.save_file = './PCANet_model/model.pkl'

        # branch network
        self.net = MLP(input_size=num_inp_red_dim, \
                              hidden_size=num_neurons, \
                              num_classes=num_out_red_dim, \
                              depth=num_layers, \
                              act=act)
        self.net.float()

        # record losses
        self.train_loss_log = []
        self.test_loss_log = []
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    def forward(self, X):
        X = self.convert_np_to_tensor(X)
        return self.net.forward(X)
    
    def train(self, train_data, test_data, \
              batch_size=32, epochs = 1000, \
              lr=0.001, log=True, \
              loss_print_freq = 100, \
              save_model = False, save_file = None, save_epoch = 100):

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_epoch = save_epoch
        
        if save_file is not None:
            self.save_file = save_file

        # train and test dataloaders to sample batches of data
        dataset = DataHandler(train_data['X_train'], None, \
                              train_data['Y_train'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        test_dataset = DataHandler(test_data['X_train'], None, \
                                   test_data['Y_train'])
        test_dataloader = DataLoader(test_dataset, \
                                     batch_size=batch_size, shuffle=True)

        # loss and optimizer setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        self.train_loss_log = np.zeros((epochs, 1))
        self.test_loss_log = np.zeros((epochs, 1))

        # training and testing loop
        start_time = time.perf_counter()

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print('-'*50)
        print('Starting training with {} trainable parameters...'.format(self.trainable_params))
        print('-'*50)
        
        for epoch in range(1, epochs+1):

            train_losses = []
            test_losses = []

            t1 = time.perf_counter()

            # training loop
            for X_train, Y_train in dataloader:

                # clear gradients
                optimizer.zero_grad()

                # forward pass through model
                Y_train_pred = self.forward(X_train)

                # compute and save loss
                loss = criterion(Y_train_pred, Y_train)
                train_losses.append(loss.item())

                # backward pass
                loss.backward()

                # update parameters
                optimizer.step()

            # update learning rate
            scheduler.step()

            # testing loop
            with torch.no_grad():
                for X_test, Y_test in test_dataloader:
                    
                    # forward pass through model
                    Y_test_pred = self.forward(X_test)

                    # compute and save test loss
                    test_loss = criterion(Y_test_pred, Y_test)
                    test_losses.append(test_loss.item())

            # log losses
            self.train_loss_log[epoch-1, 0] = np.mean(train_losses)
            self.test_loss_log[epoch-1, 0] = np.mean(test_losses)

            # print loss and time
            t2 = time.perf_counter()
            epoch_time = t2 - t1

            if log == True and (epoch % loss_print_freq == 0 or epoch == epochs or epoch == 1):
                print('-'*50)
                print('Epoch: {:5d}, Train Loss (l2 squared): {:.3e}, Test Loss (l2 squared): {:.3e}, Time (sec): {:.3f}'.format(epoch, \
                                    np.mean(self.train_loss_log[epoch-1, 0]), \
                                    np.mean(self.test_loss_log[epoch-1, 0]), \
                                    epoch_time))
                print('-'*50)

            # check if we need to save model parameters
            if save_model == True and (epoch % save_epoch == 0 or epoch == epochs):
                torch.save(self, self.save_file)
                print('-'*50)
                print('Model parameters saved at epoch {}'.format(epoch))
                print('-'*50)

        # print final message
        end_time = time.perf_counter()
        print('-'*50)
        print('Train time: {:.3f}, Epochs: {:5d}, Batch Size: {:5d}, Final Train Loss (l2 squared): {:.3e}, Final Test Loss (l2 squared): {:.3e}'.format(end_time - start_time, \
                    epochs, batch_size, \
                    self.train_loss_log[-1, 0], \
                    self.test_loss_log[-1, 0]))
        print('-'*50)
    
    def predict(self, X):
        with torch.no_grad():
            return self.forward(X)