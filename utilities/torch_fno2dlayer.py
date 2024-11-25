import torch
import torch.nn as nn
import torch.nn.functional as nnF

class FNO2DLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, \
                 modes1, modes2, \
                 apply_act = True, \
                 act = nnF.gelu):
        
        super(FNO2DLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.apply_act = apply_act
        self.act = act

        # parameters in nonlocal transformation
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))

        # parameters in linear transformation
        self.w = nn.Conv2d(self.out_channels, self.out_channels, 1)

    # complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = torch.einsum("bixy,ioxy->boxy",a,b)
        return op

    def fourier_transform(self, x):
        batchsize = x.shape[0]
        
        # compute Fourier coeffcients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # return to physical space
        x = torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1)))
        return x
    
    def linear_transform(self, x):
        return self.w(x)
    
    def forward(self, x):
        x = self.fourier_transform(x) + self.linear_transform(x)
        if self.apply_act:
            return self.act(x)
        else:
            return x