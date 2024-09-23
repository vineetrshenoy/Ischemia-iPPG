import os
import sys
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.utils.weight_norm as weight_norm
import torch.autograd as autograd
import mlflow
from complexPyTorch.complexLayers import ComplexReLU, NaiveComplexBatchNorm1d
from hand_ischemia.models.timeScaleNetwork import TiscMlpN
from hand_ischemia.models.network import ResNetUNet
from complexPyTorch.complexFunctions import complex_relu
from complexPyTorch.complexLayers import ComplexReLU, ComplexMaxPool1d, NaiveComplexBatchNorm1d
from hand_ischemia.models import PhysNet
__all__ = ['build_model']


class Spectrum_CLS(nn.Module):

    def __init__(self):
        super(Spectrum_CLS, self).__init__()
        self.mtype = 'NN'
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=2, dtype=torch.cfloat),
            ComplexReLU(),
            NaiveComplexBatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, dtype=torch.cfloat),
            ComplexReLU(),
            NaiveComplexBatchNorm1d(64),
            nn.Conv1d(64, 1, kernel_size=16, stride=2, dtype=torch.cfloat),
            ComplexMaxPool1d(16),
            
        )
        self.last_linear = nn.Linear(10, 2)
        self.sig_act = nn.Sigmoid()

    def forward(self, x, **kwargs):

        out = self.layers(x)

        out = torch.squeeze(torch.abs(out))
        out = self.last_linear(out)
        out = self.sig_act(out)
        # Skip connection
        #out = out + x
        out = out

        return out


def build_model(cfg):

    model = PhysNet(2, in_ch=3)
    
    if cfg.TIME_SCALE_PPG.CLS_MODEL_TYPE == 'TiSc': 
        classifier = TiscMlpN([[2,256],100,50,20,10,2], length_input=256, tisc_initialization='white')
    elif cfg.TIME_SCALE_PPG.CLS_MODEL_TYPE == 'SPEC':
        classifier = Spectrum_CLS()
        

    
    return model, classifier


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(m.bias)


def weights_init_complex(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.009)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('NaiveComplexBatchNorm1d') != -1:
        torch.nn.init.normal_(m.bn_i.weight, mean=0.0, std=0.009)
        torch.nn.init.zeros_(m.bn_i.bias)

        torch.nn.init.normal_(m.bn_r.weight, mean=0.0, std=0.009)
        torch.nn.init.zeros_(m.bn_r.bias)


if __name__ == "__main__":

    print('Hello')

    x = torch.randn(100, 1, 2561, dtype=torch.cfloat)
    model = Denoiser_cls()
    #model.apply(weights_init)
    output = model(x)
    temp = 5
    #model = Denoiser()
    #model.apply(weights_init_complex)
    #x = torch.randn(100, 5, 1251, dtype=torch.cfloat)
    #output = model(x)


    #model = DEQ_Layer()
    #model.apply(weights_init_complex)
    #x = torch.randn(100, 5, 1251, dtype=torch.cfloat)
    #z0 = torch.zeros_like(x)
    #output = model(z0, x)



    '''
    x = torch.randn(1, 5, 1251, dtype=torch.cfloat)
    model = DenoiserReal()
    model.apply(weights_init)    
    output = model(x)
    '''
    #print('output.shape {} ; x.shape {}'.format(output.shape, x.shape))
    #residual = torch.linalg.norm(output)
    #print('The norm of the residual is {} '.format(residual))
