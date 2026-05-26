import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

class SurrogateModel:
    def __init__(self, true_model, model, data):
        self.true_model = true_model
        self.model = model
        self.data = data
        self.model_type = self.model.name
        self.w_to_m = None
        self.u = None

    def solveFwd(self, w):
        if self.model_type == 'DeepONet':
            self.w_to_m = self.true_model.transform_gaussian_pointwise(w)
            if self.data.num_Y_components > 1:
                self.w_to_m = self.w_to_m.reshape(1, -1)
            self.u = self.data.decoder_Y(self.model.predict(self.data.encoder_X(self.w_to_m), self.data.X_trunk).detach().numpy())
            if self.data.num_Y_components > 1:
                self.u = self.u.reshape(-1)
        elif self.model_type == 'PCANet':
            self.w_to_m = self.true_model.transform_gaussian_pointwise(w)
            self.u = self.data.decoder_Y(self.model.predict(self.data.encoder_X(self.w_to_m)).detach().numpy())
        else:
            raise Exception('Unknown model type')

        return self.u
    
class SurrogateModelFNO(SurrogateModel):
    def __init__(self, true_model, model, data, nodes, grid_x, grid_y, u_comps = 1):
        super().__init__(true_model, model, data)
        self.nodes = nodes
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.u_comps = u_comps

        self.num_nodes = len(nodes)
        self.num_grid_x = len(grid_x)
        self.num_grid_y = len(grid_x.T)

        # create a input for FNO
        num_test = 1
        self.m_test = np.zeros((1, self.num_grid_x, self.num_grid_y, 1))

        self.grid_x_test = np.tile(self.grid_x, \
                               (num_test, 1, 1)\
                               ).reshape(num_test, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        self.grid_y_test = np.tile(self.grid_y, \
                               (num_test, 1, 1)\
                               ).reshape(num_test, \
                                         self.num_grid_x, \
                                         self.num_grid_y, 1) # exra dim
        
        self.X_test = np.concatenate((self.m_test, self.grid_x_test, self.grid_y_test), axis=-1)


    def nodes_to_grid_m(self, m):
        return griddata(self.nodes, m, (self.grid_x, self.grid_y), method='linear')
        
    def grid_to_nodes_u(self, u):
        if self.u_comps == 1:
            interp = RegularGridInterpolator((self.grid_x[:,0], self.grid_y[0,:]), u[:, :, 0])
            return interp(self.nodes)
        else:
            u_interp = np.zeros(self.u_comps*self.num_nodes)
            for i in range(self.u_comps):
                interp = RegularGridInterpolator((self.grid_x[:,0], self.grid_y[0,:]), u[:, :, i])
                u_interp[i*self.num_nodes:(i+1)*self.num_nodes] = interp(self.nodes)
            
            return u_interp
                
    def solveFwd(self, w):
        if self.model_type == 'FNO':
            self.w_to_m = self.true_model.transform_gaussian_pointwise(w)
            self.m_test[0, :, :, 0] = self.nodes_to_grid_m(self.w_to_m)
            
            # encode input m
            self.m_test = self.data.encoder_X(self.m_test)

            # combine encoded m with grid
            self.X_test[0, :, :, 0] = self.m_test[0, :, :, 0]

            # predict u
            u_grid = self.data.decoder_Y(self.model.predict(self.X_test).detach().numpy())[0, :, :, :]

            self.u = self.grid_to_nodes_u(u_grid)

        else:
            raise Exception('Unknown model type')

        return self.u