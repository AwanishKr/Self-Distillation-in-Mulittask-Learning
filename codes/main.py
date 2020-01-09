from __future__ import print_function, division
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import math
from torch.autograd import Variable
from segnet_model import *
from load_dataset_city import *

def compute_miou(x_pred, x_output):
		_, x_pred_label = torch.max(x_pred, dim=1)
		x_output_label = x_output
		batch_size = x_pred.size(0)
		for i in range(batch_size):
			true_class = 0
			first_switch = True
			for j in range(6):
				pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda(c))
				true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor))
				mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
				union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
				intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
				if union == 0:
					continue
				if first_switch:
					class_prob = intsec / union
					first_switch = False
				else:
					class_prob = intsec / union + class_prob
				true_class += 1
			if i == 0:
				batch_avg = class_prob / true_class
			else:
				batch_avg = class_prob / true_class + batch_avg
		return batch_avg / batch_size

def compute_iou(x_pred, x_output):
	_, x_pred_label = torch.max(x_pred, dim=1)
	x_output_label = x_output.type(torch.LongTensor).cuda(c)
	batch_size = x_pred.size(0)
	for i in range(batch_size):
		if i == 0:
			pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
				torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
		else:
			pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
				torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
	
	return (pixel_acc/batch_size)

def depth_error(x_pred, x_output):
	binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).cuda(c)
	x_pred_true = x_pred.masked_select(binary_mask)
	x_output_true = x_output.masked_select(binary_mask)
	abs_err = torch.abs(x_pred_true - x_output_true)
	rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
	return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)


if __name__ == "__main__":
	batch_size, c = (8,1)
	root = "/home/biplab/datasets/city_data"
	
	train_set = ImageDataset(root=root, mode="train")
	val_set = ImageDataset(root=root, mode="test")
	
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
	
	train_len = len(train_loader)
	val_len = len(val_loader)

	print("number of tuples in training set is: ", train_len)
	print("number of tuples in validation set is: ", val_len)

	model = Segnet(3,8,1).cuda(c)

	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	# 	model = nn.DataParallel(model)
	# #model = nn.DataParallel(UNet(3,6,1))
	# model.cuda()
	
	opt = optim.Adam(model.parameters(), lr=1e-4)
	scheduler = optim.lr_scheduler.MultiStepLR(opt, [100, 150], gamma=0.5)
	total_epoch = 200
	a = torch.rand(2)
	w1 = a[0].item()
	w2 = a[1].item()
	print(w1,w2)

	learn_rate = 0.001

	for epoch in range(total_epoch):
		print("\nepoch number: ", epoch)
		train_list_abs_err = []
		train_list_rel_err = []
		train_list_iou = []
		train_list_miou = []
		train_list_loss = []

		test_list_abs_err = []
		test_list_rel_err = []
		test_list_iou = []
		test_list_miou = []
		test_list_loss = []
		
		model.train()
		
		iter_train_loader = iter(train_loader)
		for k in range(train_len):
			image, depth, label, target_rgb = next(iter_train_loader)
			train_image = Variable(image).cuda(c)
			train_depth = Variable(depth).cuda(c)
			train_label = Variable(label.type(torch.LongTensor)).cuda(c)
			output1, output2 = model(train_image)

			opt.zero_grad()

			train_loss1 = nn.functional.cross_entropy(output1, train_label)
			binary_mask = (torch.sum(output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda(c)
			train_loss2 = torch.sum(torch.abs(output2 - train_depth) * binary_mask) / torch.nonzero(binary_mask).size(0)
			train_loss = (abs(w1)/(abs(w1)+abs(w2)))*train_loss1 + (abs(w2)/(abs(w1)+abs(w2)))*train_loss2

			factor1 = abs(w2)/((abs(w1)+abs(w2))**2)
			factor2 = abs(w1)/((abs(w1)+abs(w2))**2)
			w1 = w1 - learn_rate*(train_loss2.item() - train_loss1.item())*factor1
			w2 = w2 - learn_rate*(train_loss1.item() - train_loss2.item())*factor2

			temp_train_miou = compute_miou(output1, label)
			temp_train_iou = compute_iou(output1, label)
			temp_train_abs_err, temp_train_rel_err = depth_error(output2, train_depth)


			train_loss.backward()
			opt.step()

			train_list_abs_err.append(batch_size*temp_train_abs_err.item())
			train_list_rel_err.append(batch_size*temp_train_rel_err.item())
			train_list_iou.append(batch_size*temp_train_iou.item())
			train_list_miou.append(batch_size*temp_train_miou.item())
			train_list_loss.append(batch_size*train_loss.item())
		
		t = time.time()
		scheduler.step()	
		model.eval()
		
		with torch.no_grad():
			iter_test_loader = iter(val_loader)
			for k in range(val_len):
				test_image, test_depth, test_label, target_rgb = next(iter_test_loader)
				x = Variable(test_image).cuda(c)
				y_1 = Variable(test_depth).cuda(c)
				cuda_test_label = Variable(test_label.type(torch.LongTensor)).cuda(c)
				pred1, pred2 = model(x)


				# if k == val_len -1:
					
				# 	filename_target1 = "files/multi_results/" + str(epoch) + str('_')+ str(k) + "sem_target.jpg"
				# 	filename_output2 = "files/multi_results/" + str(epoch) + str('_')+ str(k) + "dep_output.jpg"
				# 	filename_target2 = "files/multi_results/" + str(epoch) + str('_')+ str(k) + "dep_target.jpg"
				# 	filename_input = "files/multi_results/" + str(epoch) + str('_')+ str(k) + "input.jpg"

				# 	y_threshed = torch.zeros((pred1.size()[0], 3, pred1.size()[2], pred1.size()[3]))
					
				# 	for idx in range(0, output1.size()[0]):
				# 		maxindex = torch.argmax(pred1[idx], dim=0).cpu().int()
				# 		y_threshed[idx] = val_set.class_to_rgb(maxindex)
				# 	filename_output1 = "files/multi_results/"+ str(epoch) + str('_')+ str(k) + "sem_output.jpg"
				# 	torchvision.utils.save_image(y_threshed.type(torch.FloatTensor), filename_output1, normalize = True)
					
				# 	torchvision.utils.save_image(test_image, filename_input)
				# 	torchvision.utils.save_image(test_depth, filename_target2)
				# 	torchvision.utils.save_image(pred2, filename_output2)
				# 	torchvision.utils.save_image(target_rgb.type(torch.FloatTensor), filename_target1, normalize = True)
				# 	torchvision.utils.save_image(y_threshed.type(torch.FloatTensor), filename_output1, normalize = True)

				#test_loss1 = F.nll_loss(pred1, cuda_test_label, ignore_index=-1)

				test_loss1 = nn.functional.cross_entropy(pred1, cuda_test_label)
				binary_mask = (torch.sum(pred2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda(c)
				test_loss2 = torch.sum(torch.abs(pred2 - y_1) * binary_mask) / torch.nonzero(binary_mask).size(0)
				test_loss = (abs(w1)/(abs(w1)+abs(w2)))*test_loss1 + (abs(w2)/(abs(w1)+abs(w2)))*test_loss2

				temp_test_miou = compute_miou(pred1, test_label)
				temp_test_iou = compute_iou(pred1, test_label)
				temp_test_abs_err, temp_test_rel_err = depth_error(pred2, y_1)

			test_list_abs_err.append(batch_size*temp_test_abs_err.item())
			test_list_rel_err.append(batch_size*temp_test_rel_err.item())
			test_list_iou.append(batch_size*temp_test_iou.item())
			test_list_miou.append(batch_size*temp_test_miou.item())
			test_list_loss.append(batch_size*test_loss.item())

			# list_loss_test.append(batch_size*test_loss.item())




		print("evaluation time", time.time()-t)
		epoch_loss_train = sum(train_list_loss)/(batch_size*train_len)
		train_abs_err = sum(train_list_abs_err)/(batch_size*train_len)
		train_rel_err = sum(train_list_rel_err)/(batch_size*train_len)
		train_iou = sum(train_list_iou)/(batch_size*train_len)
		train_miou = sum(train_list_miou)/(batch_size*train_len)

		epoch_loss_test = sum(test_list_loss)/(batch_size*val_len)
		test_abs_err = sum(test_list_abs_err)/(batch_size*val_len)
		test_rel_err = sum(test_list_rel_err)/(batch_size*val_len)
		test_iou = sum(test_list_iou)/(batch_size*val_len)
		test_miou = sum(test_list_miou)/(batch_size*val_len)

		# file_loss.write(str(epoch_loss_train)+","+str(epoch_loss_test)+"\n")
		# file_weight.write(str(abs(w1)/(abs(w1)+abs(w2)))+","+str(abs(w2)/(abs(w1)+abs(w2)))+"\n")
		# file_acc.write(str(train_iou)+","+str(train_miou)+","+str(test_iou)+","+str(test_miou)+","+str(train_abs_err)+","+str(train_rel_err)+","+str(test_abs_err)+","+str(test_rel_err)+"\n")

		print("training loss: ", epoch_loss_train, train_abs_err, "	" ,train_rel_err, " ", train_iou, "	", train_miou, " ", "testing loss:", epoch_loss_test, test_abs_err, test_rel_err, test_iou, " ", test_miou)
		print("\nweights are: ", (abs(w1)/(abs(w1)+abs(w2))), (abs(w2)/(abs(w1)+abs(w2))))