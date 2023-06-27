import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Reconstruction_Loss(nn.Module):
    def __init__(self):
        super(Reconstruction_Loss, self).__init__()

    def forward(self, image, label):
        loss = torch.mean(torch.sum(torch.abs(image - label), [1, 2, 3]), dim=0)
        return loss


class list_Pyramid_Loss(nn.Module):

    def __init__(self,num_high):
        super(Pyramid_Loss, self).__init__()
        self.num_high = num_high

    def forward(self, list1, list2):
        loss = 0
        for step in range(self.num_high+1):
            if step == self.num_high:
                loss = loss + torch.mean(4*(step+1)*(step+1)*torch.sum(torch.abs(list1[step] - list2[step]), [1, 2, 3]), dim=0)
                break
            loss = loss + (step+1)*(step+1)*torch.mean(4 * torch.sum(torch.abs(list1[step] - list2[step]), [1, 2, 3]), dim=0)
        return loss

class Pyramid_Loss(nn.Module):

    def __init__(self):
        super(Pyramid_Loss, self).__init__()

    def forward(self, x_1, x_2, x_3, label_1, label_2, label_3):
        loss1 = torch.mean(4 * torch.sum(torch.abs(x_1 - label_1), [1, 2, 3]), dim=0)
        loss2 = torch.mean(2 * torch.sum(torch.abs(x_2 - label_2), [1, 2, 3]), dim=0)
        loss3 = torch.mean(torch.sum(torch.abs(x_3 - label_3), [1, 2, 3]), dim=0)
        loss = loss1 + loss2 + loss3
        return loss

"""
if __name__ == "__main__":
    x_1 = torch.rand(size=(2, 3, 224, 224))
    x_2 = torch.rand(size=(2, 3, 224, 224))
    x_3 = torch.rand(size=(2, 3, 224, 224))
    x_4 = torch.rand(size=(2, 3, 224, 224))
    x_5 = torch.rand(size=(2, 3, 224, 224))
    x_6 = torch.rand(size=(2, 3, 224, 224))
    x_7 = torch.rand(size=(2, 3, 224, 224))
    x_8 = torch.rand(size=(2, 3, 224, 224))
    loss1 = Pyramid_Loss()
    print(loss1(x_1, x_2,x_3, x_4, x_5, x_6))
    loss2 = Reconstruction_Loss()
    print(loss1(x_1, x_2,x_3, x_4, x_5, x_6)+loss2(x_7, x_8))
"""

"""
if __name__ == "__main__":
    x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).view(1, 3, 2, 2)
    y = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12]).view(1, 3, 2, 2)
    loss = Reconstruction_Loss()
    print(loss(x, y))
"""

