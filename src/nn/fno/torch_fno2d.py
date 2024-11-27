import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# local utility methods
from torch_fno2dlayer import FNO2DLayer

class FNO2D(nn.Module):
    
    def __init__(self, num_layers, width, \
                 fourier_modes1, fourier_modes2, \
                 num_Y_components, save_file=None):

        super(FNO2D, self).__init__()
        self.name = 'FNO'
        self.save_file = save_file
        if save_file is None:
            self.save_file = './FNO_model/model.pkl'

        self.num_layers = num_layers
        self.width = width # dimension of hidden output space 
        self.fourier_modes1 = fourier_modes1
        self.fourier_modes2 = fourier_modes2
        
        # dimension d_o of the pointwise value of the target function (u(x) \in R^d_o)
        self.num_Y_components = num_Y_components

        # create hidden layers (FNO layers)
        self.fno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fno_layers.append(FNO2DLayer(self.width, \
                                            self.width, \
                                            self.fourier_modes1, \
                                            self.fourier_modes2))
        
        # no activation in the last hidden layer
        self.fno_layers[-1].apply_act = False 

        # define input-to-hidden projector
        # input has 3 components: m(x,y), x_1, x_2
        self.input_projector = nn.Linear(3, self.width)

        # define hidden-to-output projector 
        # project to the dimension of u(x) \in R^d_o
        self.output_projector = nn.Linear(self.width, self.num_Y_components)

        # record losses
        self.train_loss_log = []
        self.test_loss_log = []

        # metadata (used in applications, e.g., bayesian inversion)
        self.metadata = None
    
    def convert_np_to_tensor(self,array):
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32)
        else:
            return array

    def forward(self, X):
        x = self.convert_np_to_tensor(X)

        # input-to-hidden projector
        x = self.input_projector(x)
        
        # rearrange x so that it has the shape (batch, width, x, y)
        x = x.permute(0, 3, 1, 2)
        
        # pass through hidden layers
        for i in range(self.num_layers):
            x = self.fno_layers[i](x)
        
        # rearrange x so that it has the shape (batch, x, y, width)
        x = x.permute(0, 2, 3, 1)

        # hidden-to-output projector
        x = self.output_projector(x)

        return x
    
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

        # X_train has the shape (num_train, num_grid_x, num_grid_y, 3) 
        # given first 3 indices, each row of X_train has (m(x,y), x, y)
        # Y_train has the shape (num_train, num_grid_x, num_grid_y, num_Y_components) 
        # given first 3 indices, each row is u_1(x,y), u_2(x,y), ..., u_{num_Y_components}(x,y) (u_i being the ith component of u(x))
        # train and test dataloaders to sample batches of data
        dataloader = DataLoader(TensorDataset(train_data['X_train'], \
                                    train_data['Y_train']), \
                                batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(TensorDataset(test_data['X_train'], \
                                    test_data['Y_train']), \
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
                loss = criterion(Y_train_pred.view(batch_size, -1), \
                                 Y_train.view(batch_size, -1)) 
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
                    test_loss = criterion(Y_test_pred.view(batch_size, -1), \
                                          Y_test.view(batch_size, -1))
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