import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import os
from PIL import Image
from data.transform import *


class CityscapesSequence(Dataset):
	def __init__(self, data_dir, gt_dir, delta=3, transform=None, target_transform=None, target_size=None):
		super(CityscapesSequence).__init__()

		self.data_dir = data_dir
		self.gt_dir = gt_dir
		self.delta = delta
		self.transform = transform
		self.target_transform = target_transform
		self.target_size = target_size
		self.timesteps = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't']

		gt_files = sorted([os.path.abspath(x) for x in Path(self.gt_dir).glob('*/*_gtFine_labelTrainIds.png')])
		#print(gt_files)
		columns=['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't', 'gt']
		self.df = pd.DataFrame(columns = columns)

		for gt_file in gt_files:
			pos = gt_file.rfind('_gtFine')
			tmp = gt_file[:pos]
			pos = tmp.rfind('_')
			seq = tmp[:pos + 1]
			index = int(tmp[pos+1:])

			d = {'gt': [gt_file]}
			for idx, i in enumerate(range(index - 18, index + 1, 3)):
				num = str(i)
				if len(num) < 6:
					num = (6 - len(num)) * '0' + num
				full_pth = (seq + num + '_leftImg8bit.png').replace('gtFine', 'leftImg8bit_sequence')
				d[columns[idx]] = [full_pth]

			tmp_df = pd.DataFrame(d)
			self.df = pd.concat([self.df, tmp_df], ignore_index = True, axis = 0)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		row = self.df.iloc[idx]
		lst = []

		for t in self.timesteps:
			ret_dict = {
			'image': row[t],
			'name': "",
			'subset': "",
		}
		if self.has_labels:
			ret_dict['labels'] = row['gt']

		lst.append(self.transforms(ret_dict))

		return custom_collate(lst)



		row = self.df.iloc[idx]

		gt = Image.open(os.path.join(self.gt_dir, row['gt']))
		if self.target_size is not None:
			gt = gt.resize(self.target_size, Image.NEAREST)
		gt = np.asarray(gt)

		imgs = [np.asarray(Image.open(row[step]).convert('RGB')) for step in self.timesteps]
		imgs = np.stack(imgs, axis=0)
		print(torch.tensor(imgs))

		target, past_feats, gt = torch.tensor(target), torch.tensor(past_feats), torch.tensor(gt, dtype=torch.long)

		if self.transform:
			past_feats = self.transform(past_feats)
		if self.target_transform:
			target = self.target_transform(target)

		return past_feats, target, gt


class CityscapesFeatureSequence(Dataset):
	def __init__(self, data_dir, gt_dir, delta=3, transform=None, target_transform=None, target_size=None):
		super(FeatureDataset).__init__()

		self.data_dir = data_dir
		self.gt_dir = gt_dir
		self.delta = delta
		self.transform = transform
		self.target_transform = target_transform
		self.target_size = target_size

		gt_files = sorted([os.path.abspath(x) for x in glob(self.gt_dir + '/*.png')])
		columns=['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't', 'gt']
		self.df = pd.DataFrame(columns = columns)

		for gt_file in gt_files:
			pos = gt_file.rfind('_gtFine')
			tmp = gt_file[:pos]
			pos = tmp.rfind('_')
			seq = tmp[:pos + 1]
			index = int(tmp[pos+1:])

			d = {'gt': [gt_file]}
			for idx, i in enumerate(range(index - 18, index + 1, 3)):
				num = str(i)
				if len(num) < 6:
					num = (6 - len(num)) * '0' + num
				d[columns[idx]] = [seq + num + '_leftImg8bit.npy']

			tmp_df = pd.DataFrame(d)
			self.df = pd.concat([self.df, tmp_df], ignore_index = True, axis = 0)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		row = self.df.iloc[idx]
		print(row)

		gt = Image.open(os.path.join(self.gt_dir, row['gt']))
		if self.target_size is not None:
			gt = gt.resize(self.target_size, Image.NEAREST)
		gt = np.asarray(gt)
		target = np.load(os.path.join(self.data_dir, row['t']))

		past_feats = np.concatenate((np.load(os.path.join(self.data_dir, row['t-' + str(self.delta + 9)])),
									 np.load(os.path.join(self.data_dir, row['t-' + str(self.delta + 6)])),
									 np.load(os.path.join(self.data_dir, row['t-' + str(self.delta + 3)])),
									 np.load(os.path.join(self.data_dir, row['t-' + str(self.delta)]))), axis=0)

		target, past_feats, gt = torch.tensor(target), torch.tensor(past_feats), torch.tensor(gt, dtype=torch.long)

		if self.transform:
			past_feats = self.transform(past_feats)
		if self.target_transform:
			target = self.target_transform(target)

		return past_feats, target, gt


def create_cityscapes_label_colormap():
	"""
	Creates a label colormap used in CITYSCAPES segmentation benchmark.
	Returns: A colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=np.uint8)
	colormap[0] = [128, 64, 128]
	colormap[1] = [244, 35, 232]
	colormap[2] = [70, 70, 70]
	colormap[3] = [102, 102, 156]
	colormap[4] = [190, 153, 153]
	colormap[5] = [153, 153, 153]
	colormap[6] = [250, 170, 30]
	colormap[7] = [220, 220, 0]
	colormap[8] = [107, 142, 35]
	colormap[9] = [152, 251, 152]
	colormap[10] = [70, 130, 180]
	colormap[11] = [220, 20, 60]
	colormap[12] = [255, 0, 0]
	colormap[13] = [0, 0, 142]
	colormap[14] = [0, 0, 70]
	colormap[15] = [0, 60, 100]
	colormap[16] = [0, 80, 100]
	colormap[17] = [0, 0, 230]
	colormap[18] = [119, 11, 32]
	return colormap



if __name__ == '__main__':

	data_dir = '/home/jakov/1TB_NVMe/Users/bubas/Data/Cityscapes/leftImg8bit_sequence/val'
	data_dir = 'C:/Users/bubas/Data/Cityscapes/leftImg8bit_sequence/val'
	gt_dir = '/home/jakov/1TB_NVMe/Users/bubas/Data/Cityscapes/gtFine/val'
	gt_dir = 'C:/Users/bubas/Data/Cityscapes/gtFine/val'
	dataset = CityscapesSequence(data_dir, gt_dir)
	#print(dataset.df.info(verbose = False))
	#print(dataset.df.iloc[0]['gt'])
	#print(dataset.df.iloc[0]['t-3'])
	dataset.__getitem__(0)
