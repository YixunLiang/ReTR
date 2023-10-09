import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2d import ConvBnReLU, ResidualBlock

class FPN_FeatureExtractor(nn.Module):
    """
    Feature pyramid network
    """
    def __init__(self, out_ch=16):
        super(FPN_FeatureExtractor, self).__init__()

        self.in_planes = 16

        self.out_ch = out_ch

        self.conv1 = ConvBnReLU(3,16)
        self.layer1 = self._make_layer(32, stride=2)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        
        self.inner = nn.Conv2d(64, 96, 1, stride=1, padding=0, bias=True)

        # output convolution

        # self.output_feat3 = nn.Conv2d(96, 96//4, 1, stride=1, padding=0)
        # self.output_feat2 = nn.Conv2d(64, 64//4, 1, stride=1, padding=0)
        # self.output_feat1 = nn.Conv2d(32, 32//4, 1, stride=1, padding=0)
        self.output_feat3 = nn.Conv2d(96, 96//4, 3, stride=1, padding=1)

        self.output_feat2 = nn.Conv2d(64, 64//4, 3, stride=1, padding=1)

        self.output_feat1 = nn.Conv2d(32, 32//2, 3, stride=1, padding=1)

        self.output = nn.Conv2d(96, out_ch, 3, stride=1, padding=1)

        self.gradients = []

    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        
        return nn.Sequential(*layers)

    def forward(self, x):
        fea0 = self.conv1(x)
        # _ = fea0.register_hook(self.activations_hook)
        fea1 = self.layer1(fea0)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)
        intra_feat = F.interpolate(fea3, scale_factor=2, mode="bilinear") + self.inner(fea2)
        x_out = self.output(intra_feat)
        # _ = x_out.register_hook(self.activations_hook)

        #3D feat
        feat1_proj = self.output_feat1(fea1)
        # _ = feat1_proj.register_hook(self.activations_hook)
        feat2_proj = self.output_feat2(fea2)
        # _ = feat2_proj.register_hook(self.activations_hook)
        feat3_proj = self.output_feat3(fea3) 
        # _ = feat3_proj.register_hook(self.activations_hook)

        return x_out,[feat1_proj,feat2_proj,feat3_proj]#, fea0#, [fea1,fea2,fea3]
    
    def activations_hook(self, grad):
        self.gradients.append(grad.cpu().detach())

    def get_activations_gradient(self):
        return self.gradients
    
    def reset_gradient(self):
        self.gradients = []
        print('Reset Gradients')