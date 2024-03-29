import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input).cuda()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class Regularization(nn.Module):
    def __init__(self):
        super(Regularization, self).__init__()

    def tv(self, image):
        x_loss = torch.mean((torch.abs(image[:,:,1:,:] - image[:,:,:-1,:])))
        y_loss = torch.mean((torch.abs(image[:,:,:,1:] - image[:,:,:,:-1])))

        return (x_loss + y_loss)

    def regionLoss(self, image):
        mask_mean = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

    def absoulteRegionLoss(self, image):
        image       = torch.abs(image)
        mask_mean   = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

    def powerRegionLoss(self, image):
        image       = torch.pow(image, 2)
        mask_mean   = F.avg_pool2d(image, image.size(2), stride=1).squeeze().mean()

        return mask_mean

# ---------------------------------------------------------------------------
# function for computing the gradient penalty
# ---------------------------------------------------------------------------
def compute_gradient_norm(input_data, output_prediction):
   
    number_data     = len(input_data)
    gradient        = torch.autograd.grad(outputs=output_prediction.sum(), inputs=input_data, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_square = gradient.pow(2)
    gradient_norm   = gradient_square.reshape(number_data, -1).sum(1) 

    return gradient_norm.mean()