from __future__ import print_function, division
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from load_dataset_multi import *
from torch.autograd import Variable

class Segnet(nn.Module):
	def __init__(self, in_dim, out_dim1, out_dim2):
		super(Segnet, self).__init__()
		self.in_dim = in_dim
		self.out_dim1 = out_dim1
		self.out_dim2 = out_dim2
		
		filters = [64, 128, 256, 512, 1024]
		value = 0.2
#-----------------------------------------------------------Encoder part-----------------------------------------------
		self.dn1 = nn.Conv2d(self.in_dim, filters[0], kernel_size= 3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(filters[0])
		self.ac1 = nn.LeakyReLU(value, inplace=True)		
		self.dn2 = nn.Conv2d(filters[0], filters[0], kernel_size= 3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(filters[0])
		self.ac2 = nn.LeakyReLU(value, inplace=True)
		self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn3 = nn.Conv2d(filters[0], filters[1], kernel_size= 3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(filters[1])
		self.ac3 = nn.LeakyReLU(value, inplace=True)
		self.dn4 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(filters[1])
		self.ac4 = nn.LeakyReLU(value, inplace=True)
		self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn5 = nn.Conv2d(filters[1], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(filters[2])
		self.ac5 = nn.LeakyReLU(value, inplace=True)
		self.dn6 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn6 = nn.BatchNorm2d(filters[2])
		self.ac6 = nn.LeakyReLU(value, inplace=True)
		self.dn7 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn7 = nn.BatchNorm2d(filters[2])
		self.ac7 = nn.LeakyReLU(value, inplace=True)
		self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn8 = nn.Conv2d(filters[2], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn8 = nn.BatchNorm2d(filters[3])
		self.ac8 = nn.LeakyReLU(value, inplace=True)
		self.dn9 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn9 = nn.BatchNorm2d(filters[3])
		self.ac9 = nn.LeakyReLU(value, inplace=True)
		self.dn10 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn10 = nn.BatchNorm2d(filters[3])
		self.ac10 = nn.LeakyReLU(value, inplace=True)
		self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

		self.dn11 = nn.Conv2d(filters[3], filters[4], kernel_size= 3, stride=1, padding=1)
		self.bn11 = nn.BatchNorm2d(filters[4])
		self.ac11 = nn.LeakyReLU(value, inplace=True)
		self.dn12 = nn.Conv2d(filters[4], filters[4], kernel_size= 3, stride=1, padding=1)
		self.bn12 = nn.BatchNorm2d(filters[4])
		self.ac12 = nn.LeakyReLU(value, inplace=True)
		self.dn13 = nn.Conv2d(filters[4], filters[4], kernel_size= 3, stride=1, padding=1)
		self.bn13 = nn.BatchNorm2d(filters[4])
		self.ac13 = nn.LeakyReLU(value, inplace=True)
		self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

#-----------------------------------------------------------------bottlenech---------------------------------------------------------------------
		self.bottle_conv1 = nn.Conv2d(filters[4], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bottle_bn1 = nn.BatchNorm2d(filters[3])
		self.bottle_ac1 = nn.LeakyReLU(value, inplace=True)		
		self.bottle_conv2 = nn.Conv2d(filters[3], filters[4], kernel_size= 3, stride=1, padding=1)
		self.bottle_bn2 = nn.BatchNorm2d(filters[4])
		self.bottle_ac2 = nn.LeakyReLU(value, inplace=True)

#---------------------------------------------------------------semantic decoder ------------------------------------------------------------------
		self.unpool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up1 = nn.Conv2d(filters[4], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn14 = nn.BatchNorm2d(filters[3])
		self.ac14 = nn.LeakyReLU(value, inplace=True)
		self.up2 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn15 = nn.BatchNorm2d(filters[3])
		self.ac15 = nn.LeakyReLU(value, inplace=True)
		self.up3 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.bn16 = nn.BatchNorm2d(filters[3])
		self.ac16 = nn.LeakyReLU(value, inplace=True)
		self.att_1 = self.att_layer(filters[3], filters[3], filters[3])

		self.unpool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up4 = nn.Conv2d(filters[3], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn17 = nn.BatchNorm2d(filters[2])
		self.ac17 = nn.LeakyReLU(value, inplace=True)
		self.up5 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn18 = nn.BatchNorm2d(filters[2])
		self.ac18 = nn.LeakyReLU(value, inplace=True)
		self.up6 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.bn19 = nn.BatchNorm2d(filters[2])
		self.ac19 = nn.LeakyReLU(value, inplace=True)
		self.att_2 = self.att_layer(filters[2], filters[2], filters[2])

		self.unpool_3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up7 = nn.Conv2d(filters[2], filters[1], kernel_size= 3, stride=1, padding=1)
		self.bn20 = nn.BatchNorm2d(filters[1])
		self.ac20 = nn.LeakyReLU(value, inplace=True)
		self.up8 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.bn21 = nn.BatchNorm2d(filters[1])
		self.ac21 = nn.LeakyReLU(value, inplace=True)
		self.up9 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.bn22 = nn.BatchNorm2d(filters[1])
		self.ac22 = nn.LeakyReLU(value, inplace=True)
		self.att_3 = self.att_layer(filters[1], filters[1], filters[1])

		self.unpool_4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up10 = nn.Conv2d(filters[1], filters[0], kernel_size= 3, stride=1, padding=1)
		self.bn23 = nn.BatchNorm2d(filters[0])
		self.ac23 = nn.LeakyReLU(value, inplace=True)
		self.up11 = nn.Conv2d(filters[0], filters[0], kernel_size= 3, stride=1, padding=1)
		self.bn24 = nn.BatchNorm2d(filters[0])
		self.ac24 = nn.LeakyReLU(value, inplace=True)
		self.att_4 = self.att_layer(filters[0], filters[0], filters[0])

		self.unpool_5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.up12 = nn.Conv2d(filters[0], 16, kernel_size= 3, stride=1, padding=1)
		self.bn25 = nn.BatchNorm2d(16)
		self.ac25 = nn.LeakyReLU(value, inplace=True)
		self.sem_output = nn.Conv2d(16, self.out_dim1, kernel_size= 3, stride=1, padding=1)
		self.att_5 = self.att_layer(16, 16, 16)

#-----------------------------------------------------------Decoder for depth estimation--------------------------------
		self.depth_unpool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.depth_up1 = nn.Conv2d(filters[4], filters[3], kernel_size= 3, stride=1, padding=1)
		self.depth_bn14 = nn.BatchNorm2d(filters[3])
		self.depth_ac14 = nn.LeakyReLU(value, inplace=True)
		self.depth_up2 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.depth_bn15 = nn.BatchNorm2d(filters[3])
		self.depth_ac15 = nn.LeakyReLU(value, inplace=True)
		self.depth_up3 = nn.Conv2d(filters[3], filters[3], kernel_size= 3, stride=1, padding=1)
		self.depth_bn16 = nn.BatchNorm2d(filters[3])
		self.depth_ac16 = nn.LeakyReLU(value, inplace=True)
		self.depth_att_1 = self.att_layer(filters[3], filters[3], filters[3])

		self.depth_unpool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.depth_up4 = nn.Conv2d(filters[3], filters[2], kernel_size= 3, stride=1, padding=1)
		self.depth_bn17 = nn.BatchNorm2d(filters[2])
		self.depth_ac17 = nn.LeakyReLU(value, inplace=True)
		self.depth_up5 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.depth_bn18 = nn.BatchNorm2d(filters[2])
		self.depth_ac18 = nn.LeakyReLU(value, inplace=True)
		self.depth_up6 = nn.Conv2d(filters[2], filters[2], kernel_size= 3, stride=1, padding=1)
		self.depth_bn19 = nn.BatchNorm2d(filters[2])
		self.depth_ac19 = nn.LeakyReLU(value, inplace=True)
		self.depth_att_2 = self.att_layer(filters[2], filters[2], filters[2])

		self.depth_unpool_3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.depth_up7 = nn.Conv2d(filters[2], filters[1], kernel_size= 3, stride=1, padding=1)
		self.depth_bn20 = nn.BatchNorm2d(filters[1])
		self.depth_ac20 = nn.LeakyReLU(value, inplace=True)
		self.depth_up8 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.depth_bn21 = nn.BatchNorm2d(filters[1])
		self.depth_ac21 = nn.LeakyReLU(value, inplace=True)
		self.depth_up9 = nn.Conv2d(filters[1], filters[1], kernel_size= 3, stride=1, padding=1)
		self.depth_bn22 = nn.BatchNorm2d(filters[1])
		self.depth_ac22 = nn.LeakyReLU(value, inplace=True)
		self.depth_att_3 = self.att_layer(filters[1], filters[1], filters[1])

		self.depth_unpool_4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.depth_up10 = nn.Conv2d(filters[1], filters[0], kernel_size= 3, stride=1, padding=1)
		self.depth_bn23 = nn.BatchNorm2d(filters[0])
		self.depth_ac23 = nn.LeakyReLU(value, inplace=True)
		self.depth_up11 = nn.Conv2d(filters[0], filters[0], kernel_size= 3, stride=1, padding=1)
		self.depth_bn24 = nn.BatchNorm2d(filters[0])
		self.depth_ac24 = nn.LeakyReLU(value, inplace=True)
		self.depth_att_4 = self.att_layer(filters[0], filters[0], filters[0])

		self.depth_unpool_5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.depth_up12 = nn.Conv2d(filters[0], 16, kernel_size= 3, stride=1, padding=1)
		self.depth_bn25 = nn.BatchNorm2d(16)
		self.depth_ac25 = nn.LeakyReLU(value, inplace=True)
		self.depth_output = nn.Conv2d(16, self.out_dim2, kernel_size= 3, stride=1, padding=1)
		self.depth_att_5 = self.att_layer(16, 16, 16)


	def att_layer(self, in_dim1, in_dim2, out_dim):
		att = nn.Sequential(
         		nn.Conv2d(in_dim1, in_dim2, kernel_size=1, padding=0),
         		nn.BatchNorm2d(in_dim1),
         		nn.ReLU(inplace= True),
         		nn.Conv2d(in_dim2, out_dim ,kernel_size=1, padding=0),
         		nn.BatchNorm2d(out_dim),
         		nn.Sigmoid()
        	)
		return att


	def forward(self, x):
		dn1 = self.dn1(x)
		bn1 = self.bn1(dn1)
		ac1 = self.ac1(bn1)
		dn2 = self.dn2(ac1)
		bn2 = self.bn2(dn2)
		ac2 = self.ac2(bn2)
		mp1, idx1 = self.mp1(ac2)

		dn3 = self.dn3(mp1)
		bn3 = self.bn3(dn3)
		ac3 = self.ac3(bn3)
		dn4 = self.dn4(ac3)
		bn4 = self.bn4(dn4)
		ac4 = self.ac4(bn4)
		mp2, idx2 = self.mp2(ac4)

		dn5 = self.dn5(mp2)
		bn5 = self.bn5(dn5)
		ac5 = self.ac5(bn5)
		dn6 = self.dn6(ac5)
		bn6 = self.bn6(dn6)
		ac6 = self.ac6(bn6)
		dn7 = self.dn7(ac6)
		bn7 = self.bn7(dn7)
		ac7 = self.ac7(bn7)
		mp3, idx3 = self.mp3(ac7)

		dn8 = self.dn8(mp3)
		bn8 = self.bn8(dn8)
		ac8 = self.ac8(bn8)
		dn9 = self.dn9(ac8)
		bn9 = self.bn9(dn9)
		ac9 = self.ac9(bn9)
		dn10 = self.dn10(ac9)
		bn10 = self.bn10(dn10)
		ac10 = self.ac10(bn10)
		mp4, idx4 = self.mp4(ac10)

		dn11 = self.dn11(mp4)
		bn11 = self.bn11(dn11)
		ac11 = self.ac11(bn11)
		dn12 = self.dn12(ac11)
		bn12 = self.bn12(dn12)
		ac12 = self.ac12(bn12)
		dn13 = self.dn13(ac12)
		bn13 = self.bn13(dn13)
		ac13 = self.ac13(bn13)
		mp5, idx5 = self.mp4(ac13)

		bottle_conv1 = self.bottle_conv1(mp5)
		bottle_bn1 = self.bottle_bn1(bottle_conv1)
		bottle_ac1 = self.bottle_ac1(bottle_bn1)
		bottle_conv2 = self.bottle_conv2(bottle_ac1)
		bottle_bn2 = self.bottle_bn2(bottle_conv2)
		bottle_ac2 = self.bottle_ac2(bottle_bn2)

#-------------------------------------------forward method for semantic segmentation-----------------------------------
		unpool_1 = self.unpool_1(bottle_ac2, idx5)
		up1 = self.up1(unpool_1)
		bn14 = self.bn14(up1)
		ac14 = self.ac14(bn14)
		att_1 = self.att_1(ac14)
		up2 = self.up2(ac14)
		bn15 = self.bn15(up2)
		ac15 = self.ac15(bn15)
		att_up1 = ac15*att_1
		up3 = self.up3(att_up1)
		bn16 = self.bn16(up3)
		ac16 = self.ac16(bn16)

		unpool_2 = self.unpool_2(ac16, idx4)
		up4 = self.up4(unpool_2)
		bn17 = self.bn17(up4)
		ac17 = self.ac17(bn17)
		att_2 = self.att_2(ac17)
		up5 = self.up5(ac17)
		bn18 = self.bn18(up5)
		ac18 = self.ac18(bn18)
		att_up2 = ac18*att_2
		up6 = self.up6(att_up2)
		bn19 = self.bn19(up6)
		ac19 = self.ac19(bn19)

		unpool_3 = self.unpool_3(ac19, idx3)
		up7 = self.up7(unpool_3)
		bn20 = self.bn20(up7)
		ac20 = self.ac20(bn20)
		att_3 = self.att_3(ac20)
		up8 = self.up8(ac20)
		bn21 = self.bn21(up8)
		ac21 = self.ac21(bn21)
		att_up3 = ac21*att_3
		up9 = self.up9(att_up3)
		bn22 = self.bn22(up9)
		ac22 = self.ac22(bn22)

		unpool_4 = self.unpool_4(ac22, idx2)
		up10 = self.up10(unpool_4)
		bn23 = self.bn23(up10)
		ac23 = self.ac23(bn23)
		att_4 = self.att_4(ac23)
		up11 = self.up11(ac23)
		bn24 = self.bn24(up11)
		ac24 = self.ac24(bn24)
		att_up4 = ac24*att_4

		unpool_5 = self.unpool_5(att_up4, idx1)
		up12 = self.up12(unpool_5)
		bn25 = self.bn25(up12)
		ac25 = self.ac25(bn25)
		sem_output = self.sem_output(ac25)

#-----------------------------------------------------forward method for depth -----------------------------------------
		depth_unpool_1 = self.depth_unpool_1(bottle_ac2, idx5)
		depth_up1 = self.depth_up1(depth_unpool_1)
		depth_bn14 = self.depth_bn14(depth_up1)
		depth_ac14 = self.depth_ac14(depth_bn14)
		depth_att_1 = self.depth_att_1(depth_ac14)
		depth_up2 = self.depth_up2(depth_ac14)
		depth_bn15 = self.depth_bn15(depth_up2)
		depth_ac15 = self.depth_ac15(depth_bn15)
		depth_att_up1 = depth_ac15*depth_att_1
		depth_up3 = self.depth_up3(depth_att_up1)
		depth_bn16 = self.depth_bn16(depth_up3)
		depth_ac16 = self.depth_ac16(depth_bn16)

		depth_unpool_2 = self.depth_unpool_2(ac16, idx4)
		depth_up4 = self.depth_up4(depth_unpool_2)
		depth_bn17 = self.depth_bn17(depth_up4)
		depth_ac17 = self.depth_ac17(depth_bn17)
		depth_att_2 = self.depth_att_2(depth_ac17)
		depth_up5 = self.depth_up5(depth_ac17)
		depth_bn18 = self.depth_bn18(depth_up5)
		depth_ac18 = self.depth_ac18(depth_bn18)
		depth_att_up2 = depth_ac18*depth_att_2
		depth_up6 = self.depth_up6(depth_att_up2)
		depth_bn19 = self.depth_bn19(depth_up6)
		depth_ac19 = self.depth_ac19(depth_bn19)

		depth_unpool_3 = self.depth_unpool_3(ac19, idx3)
		depth_up7 = self.depth_up7(depth_unpool_3)
		depth_bn20 = self.depth_bn20(depth_up7)
		depth_ac20 = self.depth_ac20(depth_bn20)
		depth_att_3 = self.depth_att_3(depth_ac20)
		depth_up8 = self.depth_up8(depth_ac20)
		depth_bn21 = self.depth_bn21(depth_up8)
		depth_ac21 = self.depth_ac21(depth_bn21)
		depth_att_up3 = depth_ac21*depth_att_3
		depth_up9 = self.depth_up9(depth_att_up3)
		depth_bn22 = self.depth_bn22(depth_up9)
		depth_ac22 = self.depth_ac22(depth_bn22)

		depth_unpool_4 = self.unpool_4(ac22, idx2)
		depth_up10 = self.depth_up10(depth_unpool_4)
		depth_bn23 = self.depth_bn23(depth_up10)
		depth_ac23 = self.depth_ac23(depth_bn23)
		depth_att_4 = self.depth_att_4(depth_ac23)
		depth_up11 = self.depth_up11(depth_ac23)
		depth_bn24 = self.depth_bn24(depth_up11)
		depth_ac24 = self.depth_ac24(depth_bn24)
		depth_att_up4 = depth_ac24*depth_att_4

		depth_unpool_5 = self.depth_unpool_5(depth_att_up4, idx1)
		depth_up12 = self.depth_up12(depth_unpool_5)
		depth_bn25 = self.depth_bn25(depth_up12)
		depth_ac25 = self.depth_ac25(depth_bn25)
		depth_output = self.depth_output(depth_ac25)	

		return sem_output, depth_output



		