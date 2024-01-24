import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import os
from PIL import Image
import random

from data.transform import *
import torchvision
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


class OrsicToTensor:
    def __call__(self, pil_img):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.array(pil_img, dtype=np.float32), (2, 0, 1))))
        # return torchvision.transforms.functional.pil_to_tensor(pil_img)


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

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, 
                 images_dir='leftImg8bit', labels_dir='gtFine', epoch=None):
        self.root = root
        self.images_dir = self.root / images_dir / subset
        self.labels_dir = self.root / labels_dir / subset
        self.depth_dir = self.root / 'depth' / subset
        self.subset = subset
        self.has_labels = subset != 'test'
        self.open_depth = open_depth
        self.images = list(sorted(self.images_dir.glob('*/*.png')))
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
    def __init__(self, data_dir, gt_dir, delta=0, transforms=None, sem_labels=True, target_transform=None,
                 target_size=None, subset='train', extended=False):
        super(CityscapesSequence).__init__()

        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.delta = delta
        self.transforms = transforms
        self.target_transform = target_transform
        self.target_size = target_size
        self.subset = subset
        self.timesteps = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't']
        self.sem_labels = sem_labels

        if self.transforms is None:
            self.transforms = OrsicToTensor()

        gt_files = sorted([os.path.abspath(x) for x in Path(self.gt_dir).glob('*/*_gtFine_labelTrainIds.png')])
        # print(gt_files)
        columns = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't', 'gt']
        self.df = pd.DataFrame(columns=columns)

        iterations = 1
        if extended:
            iterations = 4

        for gt_file in gt_files:
            pos = gt_file.rfind('_gtFine')
            tmp = gt_file[:pos]
            pos = tmp.rfind('_')
            seq = tmp[:pos + 1]
            index_orig = int(tmp[pos + 1:])

            for it in range(iterations):
                index = index_orig + it * 3
                d = {'gt': [gt_file]}

                for idx, i in enumerate(range(index - 18, index + 1, 3)):
                    num = str(i)
                    if len(num) < 6:
                        num = (6 - len(num)) * '0' + num
                    full_pth = (seq + num + '_leftImg8bit.png').replace('gtFine', 'leftImg8bit_sequence')
                    d[columns[idx]] = [full_pth]

                tmp_df = pd.DataFrame(d)
                self.df = pd.concat([self.df, tmp_df], ignore_index=True, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        ret_dict = dict()

        if self.delta == 0:
            times = self.timesteps
        if self.delta == 3:
            times = self.timesteps[2:6]
            times.append('t')
        if self.delta == 9:
            times = self.timesteps[:4]
            times.append('t')

        ret_dict['times'] = times

        for t in times:
            ret_dict[t] = Image.open(row[t]).convert('RGB')

        if self.sem_labels:
            ret_dict['labels'] = Image.open(row['gt'])

        for t in times:
            ret_dict[t] = self.transforms(ret_dict[t])

        # ret_dict['t_name'] = row['t'][row['t'].rfind('/') + 1:row[t].rfind('.')]
        for t in times:
            ret_dict[t + '_name'] = row[t][row[t].rfind('/') + 1:row[t].rfind('.')]

        return ret_dict


class CityscapesFeatureSequence(Dataset):
    def __init__(self, feature_dir, gt_dir, img_dir=None, delta=0, transforms=None, target_transform=None,
                 subset='train', extended=False, sem_labels=True, flip=False, resize_target=None):
        super(CityscapesFeatureSequence).__init__()

        self.feature_dir = feature_dir
        self.gt_dir = gt_dir
        self.img_dir = img_dir
        self.delta = delta
        self.transforms = transforms
        self.target_transform = target_transform
        self.resize_target = resize_target
        self.subset = subset
        self.sem_labels = sem_labels
        self.extended = extended
        self.flip = flip
        self.flip_trans = torchvision.transforms.functional.hflip
        self.timesteps = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't']

        gt_files = sorted([os.path.abspath(x) for x in Path(self.gt_dir).glob('*/*_gtFine_labelTrainIds.png')])
        # print(gt_files)
        columns = ['t-18', 't-15', 't-12', 't-9', 't-6', 't-3', 't', 'gt']
        self.df = pd.DataFrame(columns=columns)

        iterations = 1
        if extended:
            iterations = 4

        for gt_file in gt_files:
            pos = gt_file.rfind('_gtFine')
            tmp = gt_file[:pos]
            pos1 = tmp.rfind('/')
            pos2 = tmp.rfind('_')
            seq = tmp[pos1 + 1: pos2 + 1]
            index_orig = int(tmp[pos2 + 1:])

            for it in range(iterations):
                index = index_orig + it * 3

                # if not self.sem_labels:
                #     gt_file = gt_file.replace(self.gt_dir, self.img_dir).replace('_gtFine_labelTrainIds', '_leftImg8bit')

                d = {}

                for idx, i in enumerate(range(index - 18, index + 1, 3)):
                    num = str(i)
                    if len(num) < 6:
                        num = (6 - len(num)) * '0' + num
                    full_pth = (feature_dir + '/' + seq + num + '_leftImg8bit.npy')
                    d[columns[idx]] = [full_pth]

                if not self.sem_labels:
                    num = str(index)
                    if len(num) < 6:
                        num = (6 - len(num)) * '0' + num
                    city = seq[:seq.find('_')]
                    gt_file = img_dir + '/' + city + '/' + seq + num + '_leftImg8bit.png'

                d['gt'] = gt_file

                tmp_df = pd.DataFrame(d)
                self.df = pd.concat([self.df, tmp_df], ignore_index=True, axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        target_feats = np.load(row['t'])

        do_flip = self.flip and bool(random.getrandbits(1))

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
        past_feats, target_feats = torch.from_numpy(past_feats), torch.from_numpy(target_feats)

        if do_flip:
            past_feats = self.flip_trans(past_feats)
            target_feats = self.flip_trans(target_feats)

        '''
		if self.transform:
			past_feats = self.transform(past_feats)
		if self.target_transform:
			past_feats = self.target_transform(past_feats)
		'''

        gt = Image.open(row['gt'])
        if self.resize_target is not None:
            if self.sem_labels:
                gt = gt.resize(self.resize_target, Image.NEAREST)
            else:
                gt = gt.resize(self.resize_target, Image.BICUBIC)

        if self.sem_labels:
            gt = np.asarray(gt)
            gt = torch.tensor(gt, dtype=torch.long)
        else:
            gt = T.ToTensor()(gt)

        if do_flip:
            gt = self.flip_trans(gt)

        if self.target_transform is not None:
            gt = self.target_transform(gt)

        return past_feats, target_feats, gt #, row['t']


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super(CombinedDataset).__init__()
        assert len(dataset1) == len(dataset2), 'Both dataset should have same length, got: ' + str(len(dataset1)) + ', ' + str(len(dataset1))
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]


if __name__ == '__main__':
    print('Cityscapes dataset test')

    from torchvision.transforms import Compose
    from data.transform import *

    trans_train = Compose(
        [torchvision.transforms.CenterCrop(512),
         torchvision.transforms.ToTensor(),
         ])

    dataset_train = CityscapesSequence(
        '/run/media/jakov/2TB KC3000/Users/bubas/Data/Cityscapes/leftImg8bit/train',
        '/run/media/jakov/2TB KC3000/Users/bubas/Data/Cityscapes/gtFine/train',
        delta=9, subset='train', sem_labels=False, transforms=trans_train)

    a = dataset_train.__getitem__(1)
