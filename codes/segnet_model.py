from __future__ import print_function, division
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
import math


class Segnet(nn.Module):
	def __init__(self, in_dim, out_dim1, out_dim2, out_dim3):
		super(Segnet, self).__init__()
		self.in_dim = in_dim
		self.out_dim1 = out_dim1
		self.out_dim2 = out_dim2
		self.out_dim3 = out_dim3
		
		filters = [64, 128, 256, 512, 1024]
		value = 0.2

		self.dn1 = nn.Conv2d(self.in_dim, filters[0], kernel_size= 3, stride=1, padding=1)
		self.dn2 = nn.Conv2d(filters[0], filters[0], kernel_size= 3, stride=1, padding=1)
		self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn3 = nn.Conv2d(filters[0], filters[1], kernel_size= 3, stride=1, padding=1)
		self.dn4 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn5 = nn.Conv2d(filters[1], filters[2], kernel_size= 3, stride=1, padding=1)
		self.dn6 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.dn7 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn8 = nn.Conv2d(filters[2], filters[3], kernel_size= 3, stride=1, padding=1)
		self.dn9 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.dn10 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn11 = nn.Conv2d(filters[3], filters[4], kernel_size= 3, stride=1, padding=1)
		self.dn12 = nn.Conv2d(filters[4], filters[4], kernel_size= 3, stride=1, padding=1)
		self.dn13 = nn.Conv2d(filters[4], filters[4], kernel_size= 3, stride=1, padding=1)
		self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)


		self.unpool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up1 = nn.Conv2d(filters[4], filters[3], kernel_size= 3, stride=1, padding=1)
		self.up2 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.up3 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)

		self.unpool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up4 = nn.Conv2d(filters[3], filters[2], kernel_size= 3, stride=1, padding=1)
		self.up5 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.up6 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)

		self.unpool_3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up7 = nn.Conv2d(filters[2], filters[1], kernel_size= 3, stride=1, padding=1)
		self.up8 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.up9 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)

		self.unpool_4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up10 = nn.Conv2d(filters[1], filters[0], kernel_size= 3, stride=1, padding=1)
		self.up11 = nn.Conv2d(filters[0], filters[0], kernel_size= 3, stride=1, padding=1)

		self.unpool_5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up12 = nn.Conv2d(filters[0], self.out_dim1, kernel_size= 3, stride=1, padding=1)
		self.up13 = nn.Conv2d(self.out_dim1, self.out_dim1, kernel_size= 3, stride=1, padding=1)