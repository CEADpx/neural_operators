import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local utility methods
src_path = '../../'
sys.path.append(src_path + 'data/')
from dataMethods import DataHandler

sys.path.append(src_path + 'nn/mlp/')
from torch_mlp import MLP

class DeepONet(nn.Module):
    
    def __init__(self, num_layers, num_neurons, act, \
                 num_br_outputs, num_tr_outputs, \
                 num_inp_fn_points, \
                 out_coordinate_dimension, \
                 num_Y_components, save_file=None):

        super(DeepONet, self).__init__()
        self.name = 'DeepONet'

        self.num_inp_fn_points = num_inp_fn_points
        self.num_br_outputs = num_br_outputs
        self.num_tr_outputs = num_tr_outputs
        self.out_coordinate_dimension = out_coordinate_dimension
        self.num_Y_components = num_Y_components
        self.save_file = save_file
        if save_file is None:
            self.save_file = './DeepONet_model/model.pkl'

        # branch network
        self.branch_net = MLP(input_size=num_inp_fn_points, \
                              hidden_size=num_neurons, \
                              num_classes=num_br_outputs, \
                              depth=num_layers, \
                              act=act)
        self.branch_net.float()

        # trunk network
        self.trunk_net = MLP(input_size=out_coordinate_dimension, \
                             hidden_size=num_neurons, \
                             num_classes=num_tr_outputs, \
                             depth=num_layers, \
                             act=act)
        self.trunk_net.float()
        
        # bias added to the product of branch and trunk networks
        self.bias = [nn.Parameter(torch.ones((1,)),requires_grad=True) for i in range(num_Y_components)]

        # dimension d_o of the pointwise value of the target function (u(x) \in R^d_o)
        self.num_Y_components = num_Y_components

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
    
    def forward(self, X, X_trunk):

        X = self.convert_np_to_tensor(X)
        X_trunk = self.convert_np_to_tensor(X_trunk)
        
        branch_out = self.branch_net.forward(X)
        trunk_out = self.trunk_net.forward(X_trunk,final_act=True)

        if self.num_Y_components == 1:
            output = branch_out @ trunk_out.t() + self.bias[0]
        else:
            # when d_o > 1, split the branch output and compute the product
            output = []
            for i in range(self.num_Y_components):
                output.append(branch_out[:,i*self.num_tr_outputs:(i+1)*self.num_tr_outputs] @ trunk_out.t() + self.bias[i])
            
            # stack and reshape 
            output = torch.stack(output, dim=-1)
            output = output.reshape(-1, X_trunk.shape[0] * self.num_Y_components)

        return output
    
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
        dataset = DataHandler(train_data['X_train'], \
                              train_data['X_trunk'], train_data['Y_train'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        test_dataset = DataHandler(test_data['X_train'], \
                                   test_data['X_trunk'], test_data['Y_train'])
        test_dataloader = DataLoader(test_dataset, \
                                     batch_size=batch_size, shuffle=True)

        # store coordinates for trunk network
        X_trunk = dataset.X_trunk
        
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
            for X_train, _, Y_train in dataloader:

                # clear gradients
                optimizer.zero_grad()

                # forward pass through model
                Y_train_pred = self.forward(X_train, X_trunk)

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
                for X_test, _, Y_test in test_dataloader:
                    
                    # forward pass through model
                    Y_test_pred = self.forward(X_test, X_trunk)

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
    
    def predict(self, X, X_trunk):
        with torch.no_grad():
            return self.forward(X, X_trunk)