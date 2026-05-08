import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class EquiDermNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EquiDermNet, self).__init__()
        # Backbone
        base = models.densenet121(pretrained=True)
        self.features = base.features
        self.feat_dim = 1024
        
        # Heads
        self.proj_l = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.proj_s = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())
        
        self.edl_head = nn.Linear(512, num_classes)
        
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, alpha=1.0):
        f = self.features(x)
        f = F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)
        
        z_l = self.proj_l(f)
        z_s = self.proj_s(f)
        
        logits = self.edl_head(z_l)
        
        z_l_rev = GradientReversalFunction.apply(z_l, alpha)
        skin_pred = self.discriminator(z_l_rev)
        
        return logits, skin_pred, z_l, z_s
