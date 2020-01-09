import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
	def __init__(self, root, mode="train"):
		if mode == "train":
			self.path_1 = os.path.join(root, "train/images")
			self.list_image = [os.path.join(self.path_1, s) for s in os.listdir(self.path_1)]
			self.list_masks = [w.replace('images', 'masks') for w in self.list_image]
			self.list_masks = [w.replace('_image', '_mask') for w in self.list_masks]
			self.list_depth = [w.replace('images', 'depths') for w in self.list_image]
			self.list_depth = [w.replace('_image.jpg', '_depth.png') for w in self.list_depth]
			
		else:
			self.path_1 = os.path.join(root, "val/images")
			self.list_image = [os.path.join(self.path_1, s) for s in os.listdir(self.path_1)]
			self.list_masks = [w.replace("images", "masks") for w in self.list_image]
			self.list_masks = [w.replace('_image', '_mask') for w in self.list_masks]
			self.list_depth = [w.replace('images', 'depths') for w in self.list_image]
			self.list_depth = [w.replace('_image.jpg', '_depth.png') for w in self.list_depth]

		# self.mapping = {
		# 				0: 0,  # unlabeled
		# 				1: 0,  # ego vehicle
		# 				2: 0,  # rect border
		# 				3: 0,  # out of roi
		# 				4: 0,  # static
		# 				5: 0,  # dynamic
		# 				6: 0,  # ground
		# 				7: 1,  # road
		# 				8: 2,  # sidewalk
		# 				9: 0,  # parking
		# 				10: 0,  # rail track
		# 				11: 0,  # building
		# 				12: 0,  # wall
		# 				13: 0,  # fence
		# 				14: 0,  # guard rail
		# 				15: 0,  # bridge
		# 				16: 0,  # tunnel
		# 				17: 0,  # pole
		# 				18: 0,  # polegroup
		# 				19: 3,  # traffic light
		# 				20: 0,  # traffic sign
		# 				21: 0,  # vegetation
		# 				22: 0,  # terrain
		# 				23: 4,  # sky
		# 				24: 5,  # person
		# 				25: 0,  # rider
		# 				26: 6,  # car
		# 				27: 7,  # truck
		# 				28: 0,  # bus
		# 				29: 0,  # caravan
		# 				30: 0,  # trailer
		# 				31: 0,  # train
		# 				32: 0,  # motorcycle
		# 				33: 0,  # bicycle
		# 				34: 0  # licenseplate
		# }
		# self.mappingrgb = {
		# 				0: (255, 0, 0),  # unlabeled
		# 				1: (255, 0, 0),  # ego vehicle
		# 				2: (255, 0, 0),  # rect border
		# 				3: (255, 0, 0),  # out of roi
		# 				4: (255, 0, 0),  # static
		# 				5: (255, 0, 0),  # dynamic
		# 				6: (255, 0, 0),  # ground
		# 				7: (128, 64, 128),  # road
		# 				8: (244, 35, 232),  # sidewalk
		# 				9: (255, 0, 0),  # parking
		# 				10: (255, 0, 0),  # rail track
		# 				11: (255, 0, 0),  # building
		# 				12: (255, 0, 0),  # wall
		# 				13: (255, 0, 0),  # fence
		# 				14: (255, 0, 0),  # guard rail
		# 				15: (255, 0, 0),  # bridge
		# 				16: (255, 0, 0),  # tunnel
		# 				17: (255, 0, 0),  # pole
		# 				18: (255, 0, 0),  # polegroup
		# 				19: (255, 0, 100),  # traffic light
		# 				20: (255, 0, 0),  # traffic sign
		# 				21: (255, 0, 0),  # vegetation
		# 				22: (255, 0, 0),  # terrain
		# 				23: (0, 0, 255),  # sky
		# 				24: (220, 20, 60),  # person
		# 				25: (255, 0, 0),  # rider
		# 				26: (255, 255, 0),  # car
		# 				27: (255, 80, 190),  # truck
		# 				28: (255, 0, 0),  # bus
		# 				29: (255, 0, 0),  # caravan
		# 				30: (255, 0, 0),  # trailer
		# 				31: (255, 0, 0),  # train
		# 				32: (255, 0, 0),  # motorcycle
		# 				33: (255, 0, 0),  # bicycle
		# 				34: (255, 0, 0)  # licenseplate
		# }

		self.mapping = {
			0: 0,  # unlabeled
			1: 0,  # ego vehicle
			2: 0,  # rect border
			3: 0,  # out of roi
			4: 0,  # static
			5: 0,  # dynamic
			6: 0,  # ground
			7: 1,  # road
			8: 1,  # sidewalk
			9: 0,  # parking
			10: 0,  # rail track
			11: 2,  # building
			12: 2,  # wall
			13: 2,  # fence
			14: 0,  # guard rail
			15: 0,  # bridge
			16: 0,  # tunnel
			17: 3,  # pole
			18: 0,  # polegroup
			19: 3,  # traffic light
			20: 3,  # traffic sign
			21: 4,  # vegetation
			22: 4,  # terrain
			23: 5,  # sky
			24: 6,  # person
			25: 6,  # rider
			26: 7,  # car
			27: 7,  # truck
			28: 7,  # bus
			29: 7,  # caravan
			30: 7,  # trailer
			31: 7,  # train
			32: 7,  # motorcycle
			33: 7,  # bicycle
			34: 7  # licenseplate
		}
		self.mappingrgb = {
			0: (0, 0, 0),  # unlabeled
			1: (0, 0, 0),  # ego vehicle
			2: (0, 0, 0),  # rect border
			3: (0, 0, 0),  # out of roi
			4: (0, 0, 0),  # static
			5: (0, 0, 0),  # dynamic
			6: (0, 0, 0),  # ground
			7: (128, 64, 128),  # road
			8: (128, 64, 128),  # sidewalk
			9: (0, 0, 0),  # parking
			10: (0, 0, 0),  # rail track
			11: (70, 70, 70),  # building
			12: (70, 70, 70),  # wall
			13: (70, 70, 70),  # fence
			14: (0, 0, 0),  # guard rail
			15: (0, 0, 0),  # bridge
			16: (0, 0, 0),  # tunnel
			17: (153, 153, 153),  # pole
			18: (0, 0, 0),  # polegroup
			19: (153, 153, 153),  # traffic light
			20: (153, 153, 153),  # traffic sign
			21: (107, 142, 35),  # vegetation
			22: (107, 142, 35),  # terrain
			23: (70, 130, 180),  # sky
			24: (220, 20, 60),  # person
			25: (220, 20, 60),  # rider
			26: (0, 0, 142),  # car
			27: (0, 0, 142),  # truck
			28: (0, 0, 142),  # bus
			29: (0, 0, 142),  # caravan
			30: (0, 0, 142),  # trailer
			31: (0, 0, 142),  # train
			32: (0, 0, 142),  # motorcycle
			33: (0, 0, 142),  # bicycle
			34: (0, 0, 142)  # licenseplate
		}

	def mask_to_class(self, mask):
		maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
		for k in self.mapping:
			maskimg[mask == k] = self.mapping[k]
		return maskimg

	def mask_to_rgb(self, mask):
		rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
		for k in self.mappingrgb:
			rgbimg[0][mask == k] = self.mappingrgb[k][0]
			rgbimg[1][mask == k] = self.mappingrgb[k][1]
			rgbimg[2][mask == k] = self.mappingrgb[k][2]
		return rgbimg

	def class_to_rgb(self, mask):
		mask2class = dict((v, k) for k, v in self.mapping.items())
		rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
		for k in mask2class:
			rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
			rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
			rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
		return rgbimg

	def __getitem__(self, index):
		image_A = Image.open(self.list_image[index % len(self.list_image)])
		image_B = Image.open(self.list_depth[index % len(self.list_depth)])
		image_C = Image.open(self.list_masks[index % len(self.list_masks)])
	
		# image_B = TF.resize(image_B, size=(64,128), interpolation=Image.BILINEAR)
		# image_A = TF.resize(image_A, size=(64,128), interpolation=Image.BILINEAR)
		# image_C = TF.resize(image_C, size=(64,128), interpolation=Image.NEAREST)
		
		image_A = transforms.functional.to_tensor(image_A)
		image_B = transforms.functional.to_tensor(image_B)
		image_C = torch.from_numpy(np.array(image_C, dtype=np.uint8))

		targetrgb = self.mask_to_rgb(image_C)
		targetrgb = targetrgb.long()
		targetmask = self.mask_to_class(image_C)
		targetmask = targetmask.long()

		return image_A.type(torch.FloatTensor), image_B.type(torch.FloatTensor), targetmask, targetrgb

	def __len__(self):
		return len(self.list_image)