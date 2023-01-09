import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import os
from PIL import Image

from data.transform import *
from torch.utils.data.dataloader import default_collate
from .labels import labels

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
	if label.ignoreInEval is False:
		map_to_id[label.id] = i
		i += 1
		if label.hasInstances is True:
			inst_map_to_id[label.id] = j
			j += 1

id_to_map = {id: i for i, id in map_to_id.items()}
inst_id_to_map = {id: i for i, id in inst_map_to_id.items()}


class Cityscapes(Dataset):
	class_info = class_info
	color_info = color_info
	num_classes = 19

	map_to_id = map_to_id
	id_to_map = id_to_map

	inst_map_to_id = inst_map_to_id
	inst_id_to_map = inst_id_to_map

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, images_dir='rgb', labels_dir='labels', epoch=None):
		self.root = root
		self.images_dir = self.root / images_dir / subset
		self.labels_dir = self.root / labels_dir / subset
		self.depth_dir = self.root / 'depth' / subset
		self.subset = subset
		self.has_labels = subset != 'test'
		self.open_depth = open_depth
		self.images = list(sorted(self.images_dir.glob('*/*.ppm')))
		if self.has_labels:
			self.labels = list(sorted(self.labels_dir.glob('*/*_gtFine_labelTrainIds.png')))
		self.transforms = transforms
		self.epoch = epoch

		print(f'Num images: {len(self)}')

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		ret_dict = {
			'image': self.images[item],
			'name': self.images[item].stem,
			'subset': self.subset,
		}
		if self.has_labels:
			ret_dict['labels'] = self.labels[item]
		if self.epoch is not None:
			ret_dict['epoch'] = int(self.epoch.value)
		return self.transforms(ret_dict)


class CityscapesSequence(Dataset):
	def __init__(self, data_dir, gt_dir, delta=0, transforms=None, target_transform=None, target_size=None, subset='train'):
		super(CityscapesSequence).__init__()

		self.data_dir = data_dir
		self.gt_dir = gt_dir
		self.delta = delta
		self.transforms = transforms
		self.target_transform = target_transform
		self.target_size = target_size
		self.subset = subset
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

		if self.delta == 0:
			times = self.timesteps
		if self.delta == 3:
			times = self.timesteps[2:6]
		if self.delta == 9:
			times = self.timesteps[:4]

		for t in times:
			ret_dict = {
			'image': row[t],
			'name': Path(row[t]).stem,
			'subset': self.subset,
			}
			ret_dict['labels'] = row['gt']
			trans = self.transforms(ret_dict)
			lst.append(trans)

		sequence = default_collate(lst)

		return sequence


class CityscapesFeatureSequence(Dataset):
	def __init__(self, feature_dir, gt_dir, delta=0, transforms=None, target_transform=None, subset='train'):
		super(CityscapesSequence).__init__()

		self.feature_dir = feature_dir
		self.gt_dir = gt_dir
		self.delta = delta
		self.transforms = transforms
		self.target_transform = target_transform
		self.subset = subset
		self.timesteps = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't']

		gt_files = sorted([os.path.abspath(x) for x in Path(self.gt_dir).glob('*/*_gtFine_labelTrainIds.png')])
		#print(gt_files)
		columns=['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't', 'gt']
		self.df = pd.DataFrame(columns = columns)

		for gt_file in gt_files:
			pos = gt_file.rfind('_gtFine')
			tmp = gt_file[:pos]
			pos1 = tmp.rfind('/')
			pos2 = tmp.rfind('_')
			seq = tmp[pos1 : pos2 + 1]
			index = int(tmp[pos2 + 1: ])

			d = {'gt': [gt_file]}
			for idx, i in enumerate(range(index - 18, index + 1, 3)):
				num = str(i)
				if len(num) < 6:
					num = (6 - len(num)) * '0' + num
				full_pth = (feature_dir + '/' + seq + num + '_leftImg8bit.npy')
				d[columns[idx]] = [full_pth]

			tmp_df = pd.DataFrame(d)
			self.df = pd.concat([self.df, tmp_df], ignore_index = True, axis = 0)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		row = self.df.iloc[idx]

		sem_seg_gt = Image.open(row['gt'])#.resize(self.target_size, Image.NEAREST)
		sem_seg_gt = np.asarray(sem_seg_gt)
		target_feats = np.load(row['t'])

		if self.delta == 0:
			times = self.timesteps
		if self.delta == 3:
			times = self.timesteps[2:6]
		if self.delta == 9:
			times = self.timesteps[:4]

		feat_paths = []
		for t in times:
			feat_paths.append(row[t])

		past_feats = np.concatenate([np.load(path) for path in feat_paths], axis=0)
		past_feats, target_feats, sem_seg_gt = torch.tensor(past_feats), torch.tensor(target_feats), torch.tensor(sem_seg_gt, dtype=torch.long)
		'''
		if self.transform:
			past_feats = self.transform(past_feats)
		if self.target_transform:
			past_feats = self.target_transform(past_feats)
		'''

		return past_feats, target_feats, sem_seg_gt


if __name__ == '__main__':
	print('Cityscapes dataset test')
