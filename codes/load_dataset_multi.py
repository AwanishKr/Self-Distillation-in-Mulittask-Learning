import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
	def __init__(self, root, mode="train"):
		if mode == "train":
			self.path_1 = os.path.join(root, "train/depths")
			self.list_depth = [os.path.join(self.path_1, s) for s in os.listdir(self.path_1)]
			self.list_image = [w.replace('depths', 'images') for w in self.list_depth]
			self.list_image = [w.replace('_depth.png', '_image.jpg') for w in self.list_image]
			# self.list_image = [w.replace('png', 'tif') for w in self.list_image]
			# self.list_label= [w.replace('depths', 'labels') for w in self.list_depth]
			# self.list_label = [w.replace('_depth', '_label') for w in self.list_label]
			# self.list_label = [w.replace('png', 'tif') for w in self.list_label]
		# else:
		# 	self.path_1 = os.path.join(root, r"val/depths")
		# 	self.list_depth = [os.path.join(r"data/val/depths", s) for s in os.listdir(self.path_1)]
		# 	self.list_image = [w.replace('depths', 'images') for w in self.list_depth]
		# 	self.list_image = [w.replace('_depth', '_image') for w in self.list_image]
		# 	# self.list_image = [w.replace('png', 'tif') for w in self.list_image]
		# 	self.list_label= [w.replace('depths', 'labels') for w in self.list_depth]
		# 	self.list_label = [w.replace('_depth', '_label') for w in self.list_label]
			# self.list_label = [w.replace('png', 'tif') for w in self.list_label]
		
		# self.merged_list = list(zip(self.list_image, self.list_label, self.list_depth))
		self.merged_list = list(zip(self.list_image, self.list_depth))

		# self.mapping = {
		# 	0: 0,  # Clutter/background
		# 	1: 1,  # Low vegetation
		# 	2: 2,  # Tree 
		# 	3: 3,  # Car
		# 	4: 4,  # Building 
		# 	5: 5,  # Impervious surfaces
		# }
		# self.mappingrgb = {
		# 	0: (255, 0, 0),  # Clutter/background
		# 	1: (0, 255, 255),  # Low vegetation
		# 	2: (0, 255, 0),  # Tree 
		# 	3: (255, 255, 0),  # Car
		# 	4: (0, 0, 255),  # Building 
		# 	5: (255, 255, 255),  # Impervious surfaces
		# }
		# self.num_classes = 6

	# def mask_to_class(self, mask):
	# 	maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
	# 	for k in self.mapping:
	# 		maskimg[mask == k] = self.mapping[k]
	# 	return maskimg

	# def mask_to_rgb(self, mask):
	# 	rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
	# 	for k in self.mappingrgb:
	# 		rgbimg[0][mask == k] = self.mappingrgb[k][0]
	# 		rgbimg[1][mask == k] = self.mappingrgb[k][1]
	# 		rgbimg[2][mask == k] = self.mappingrgb[k][2]
	# 	return rgbimg

	# def class_to_rgb(self, mask):
	# 	mask2class = dict((v, k) for k, v in self.mapping.items())
	# 	rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
	# 	for k in mask2class:
	# 		rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
	# 		rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
	# 		rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
	# 	return rgbimg


	def __getitem__(self, index):
		image_A = Image.open(self.list_depth[index % len(self.list_depth)])
		image_B = Image.open(self.list_image[index % len(self.list_image)])
		# image_C = Image.open(self.list_label[index % len(self.list_label)])

		# image_A = TF.resize(image_A, size=(600+10, 600+10), interpolation=Image.BILINEAR)
		# image_B = TF.resize(image_B, size=(600+10, 600+10), interpolation=Image.BILINEAR)
		# image_C = TF.resize(image_C, size=(600+10, 600+10), interpolation=Image.BILINEAR)
		# 	# Random crop
		# i, j, h, w = transforms.RandomCrop.get_params(image_B, output_size=(256,256))
		# image_A = TF.crop(image_A, i, j, h, w)
		# image_B = TF.crop(image_B, i, j, h, w)
		# image_C = TF.crop(image_C, i, j, h, w)
		# 	# Random horizontal flipping
		# if random.random() > 0.5:
		# 	image_A = TF.hflip(image_A)
		# 	image_B = TF.hflip(image_B)
		# 	image_C = TF.hflip(image_C)

		# image_A = TF.resize(image_A, size=(512,512), interpolation=Image.BILINEAR)
		# image_B = TF.resize(image_B, size=(512,512), interpolation=Image.BILINEAR)
		# image_C = TF.resize(image_C, size=(512,512), interpolation=Image.BILINEAR)


		image_A = transforms.functional.to_tensor(image_A)
		image_B = transforms.functional.to_tensor(image_B)

		# r = np.array([255, 0, 0])
		# a = np.array([0, 255, 255])
		# g = np.array([0, 255, 0])
		# y = np.array([255, 255, 0])
		# b = np.array([0, 0, 255])
		# w = np.array([255, 255, 255])

		# image_C = np.asarray(image_C)
		# label_seg = np.zeros((image_C.shape[:2]), dtype=np.int)

		# label_seg[(image_C==r).all(axis=2)] = 0
		# label_seg[(image_C==a).all(axis=2)] = 1
		# label_seg[(image_C==g).all(axis=2)] = 2
		# label_seg[(image_C==y).all(axis=2)] = 3
		# label_seg[(image_C==b).all(axis=2)] = 4
		# label_seg[(image_C==w).all(axis=2)] = 5

		# #label_seg

		# #image_C = Image.open(self.list_label[index % len(self.list_label)]).convert('L')


		# image_C = torch.from_numpy(np.array(label_seg, dtype=np.uint8))

		# targetrgb = self.mask_to_rgb(image_C)
		# targetmask = self.mask_to_class(image_C)
		# targetmask = targetmask.long()
		# targetrgb = targetrgb.long()
		
		return image_B.type(torch.FloatTensor),image_A.type(torch.FloatTensor)

	def __len__(self):
		return len(self.merged_list)