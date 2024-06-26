import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch import nn
import torch.nn.functional as F

from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import time
import pickle

from models.f2f import *
from models.semseg import SemsegModel
from models.resnet.resnet_pyramid_forecast import *

from data.cityscapes import *
from evaluation.evaluate import ConfusionMatrix, compute_errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

moving_IDs = range(11, 18 + 1)
timestep = 9
f2f_levels = 8

forecast_after_up_block = 2
pyramid_levels = 3

begin_time = time.strftime("%H_%M_%d_%m_%Y", time.localtime())
custom_desc = '_64x32_corr_weighted_split_dconv_offset_conv_with_corr'
current_dir = 'F2F-' + str(f2f_levels) + custom_desc + '_t' + str(timestep) + '_' + begin_time

SAVE_DIR = '/path/to/save/root/' + str(pyramid_levels) + '_levels/' + current_dir + '/'
os.mkdir(SAVE_DIR)
print('Saving in: ' + SAVE_DIR)

mean, std = None, None
mean = torch.tensor(
    np.load('/path/to/mean/numpy/array'),
    device=device)
std = torch.tensor(
    np.load('/path/to/std/numpy/array'),
    device=device)
do_normalization = mean is not None and std is not None

# f2f_model = DeformF2F_N(N=f2f_levels, in_channels=512, out_channels=128, mean=mean, std=std, split_input_dconv=True)
f2f_model = DeformF2F_N_corr(N=f2f_levels, in_channels=512, out_channels=128, mean=mean, std=std, split_input_dconv=True)

backbone = resnet18(pretrained=True,
                    pyramid_levels=pyramid_levels,
                    k_upsample=3,
                    scale=1,
                    mean=[73.15, 82.90, 72.3],
                    std=[47.67, 48.49, 47.73],
                    k_bneck=1,
                    output_stride=4,
                    efficient=True,
                    forecast_after_up_block=forecast_after_up_block)

semseg_model = SemsegModel(backbone, 19, k=1, bias=True)
semseg_model.load_state_dict(
    torch.load('/path/to/pretrained/model'), strict=False)

print('Normalize:' + str(do_normalization))

f2f_model.to(device)
semseg_model.to(device)
# print(model)
f2f_model.train()
# semseg_model.train()

# Hyperparameters
initial_learning_rate1 = 5e-4
batch_size = 12
num_epochs = 160
# num_epochs2 = 5


class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=19, print_each=20):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.step_counter = 0
        self.print_each = print_each

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        # loss = logits.mean()
        loss = self.loss(logits, labels)
        # if (self.step_counter % self.print_each) == 0:
        #    print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1
        return loss


def evaluate(f2f_model, segm_model, loader):
    f2f_model.eval()
    segm_model.eval()

    loss_fn_ce = SemsegCrossEntropy(ignore_id=255)
    conf_matrix = ConfusionMatrix(num_classes=19, ignore_label=255)

    loss_l2 = []
    loss_ce = []

    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), desc='Testing') as progress:
            for batch_idx, batch in progress:
                past_feats, target_feats, sem_seg_gt = batch
                past_feats = past_feats.to(device)

                target_feats = target_feats.to(device)
                sem_seg_gt = sem_seg_gt.to(device)

                pred_feats, normalized_pred = f2f_model.forward(past_feats, additional=True)

                loss = f2f_model.loss(normalized_pred, target_feats)
                loss_l2.append(loss.cpu().item())

                logits = segm_model.forward_decoder_no_skip(pred_feats, image_size=sem_seg_gt.shape[-2:])
                probs = logits.softmax(dim=1)
                preds = torch.argmax(logits, 1)
                loss_ce.append(loss_fn_ce(probs, sem_seg_gt).cpu().item())
                
                conf_matrix.update(preds.cpu().numpy().flatten(), sem_seg_gt.cpu().numpy().flatten())

        print("--- TEST ---")
        # pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_matrix.get_matrix(), Cityscapes.class_info, verbose=True)
        mse = np.mean(np.array(loss_l2))
        ce = np.mean(np.array(loss_ce))
        print("L2 (MSE) loss: ", mse)
        print("Cross entropy loss: ", ce)
        miou, per_class_iou = conf_matrix.get_metrics(verbose=False)
        miou_mo = conf_matrix.get_subset_miou(moving_IDs)
        print("mIoU: ", round(miou * 100, 2), '%')
        print("mIoU-MO: ", round(miou_mo * 100, 2), '%')

    f2f_model.train()
    segm_model.train()
    return miou, miou_mo, mse, ce


if __name__ == '__main__':
    dataset_train = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        delta=timestep, subset='train', sem_labels=True, extended=False, flip=False)
    dataset_val = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        delta=timestep, subset='val', sem_labels=True, extended=False, flip=False)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=2,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=None, num_workers=2,
                            persistent_workers=True, prefetch_factor=8)

    metrics_dict = {
        'val_miou': [],
        'val_miou_mo': [],
        'val_mse': [],
        'val_ce': [],
        'train_miou': [],
        'train_miou_mo': [],
        'train_mse': [],
        'train_ce': []
    }

    optimizer = torch.optim.Adam(f2f_model.parameters(), lr=initial_learning_rate1)
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(semseg_model.parameters()), lr=learning_rate1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-7)

    loss_fn_ce = SemsegCrossEntropy(ignore_id=255)
    conf_matrix = ConfusionMatrix(num_classes=19, ignore_label=255)

    max_val_miou = 0
    max_val_miou_mo = 0

    for epoch in range(1, num_epochs + 1):
        loss_l2 = []
        loss_ce = []

        # print('LR:', scheduler.get_last_lr()[0])

        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f'Training (epoch={epoch}/{num_epochs})') as epoch_progress:
            for batch_idx, batch in epoch_progress:
                past_feats, target_feats, sem_seg_gt = batch
                past_feats = past_feats.to(device)
                target_feats = target_feats.to(device)
                # sem_seg_gt = sem_seg_gt.to(device)
                # print(past_feats.shape, target_feats.shape)

                optimizer.zero_grad()

                pred_feats, normalized_pred = f2f_model.forward(past_feats, additional=True)

                loss = f2f_model.loss(normalized_pred, target_feats)
                loss_l2.append(loss.cpu().item())

                # logits = semseg_model.forward_decoder_no_skip(pred_feats, image_size=sem_seg_gt.shape[1:])
                # preds = torch.argmax(logits, 1)
                # semseg_loss = loss_fn_ce(logits, sem_seg_gt)
                # loss = loss + semseg_loss
                # loss_ce.append(semseg_loss.cpu().item())

                # conf_matrix.update(preds.cpu().numpy().flatten(), sem_seg_gt.cpu().numpy().flatten())

                loss.backward()
                optimizer.step()
                scheduler.step()

        # scheduler.step()

        print("--- TRAIN ---")
        train_mse = np.mean(np.array(loss_l2))
        # train_ce = np.mean(np.array(loss_ce))
        print("L2 (MSE) loss: ", train_mse)
        # print("Cross entropy loss: ", train_ce)
        # train_miou, train_per_class_iou = conf_matrix.get_metrics(verbose=False)
        # train_miou_mo = conf_matrix.get_subset_miou(moving_IDs)
        # print("mIoU: ",  round(train_miou * 100, 2), '%')
        # print("mIoU-MO: ", round(train_miou_mo * 100, 2), '%')

        # conf_matrix.reset()

        val_miou, val_miou_mo, val_mse, val_ce = evaluate(f2f_model, semseg_model, val_loader)
        metrics_dict['val_miou'].append(val_miou)
        metrics_dict['val_miou_mo'].append(val_miou_mo)
        metrics_dict['val_mse'].append(val_mse)
        metrics_dict['val_ce'].append(val_ce)
        metrics_dict['train_mse'].append(train_mse)
        # metrics_dict['train_ce'].append(train_ce)
        # metrics_dict['train_miou'].append(train_miou)
        # metrics_dict['train_miou_mo'].append(train_miou_mo)

        if val_miou > max_val_miou:
            max_val_miou = val_miou
            max_val_miou_mo = val_miou_mo
            f2f_model.eval()
            torch.save(f2f_model.state_dict(), SAVE_DIR + 'f2f_model.pth')
            f2f_model.train()
            # torch.save(semseg_model.state_dict(), SAVE_DIR + 'semseg_model.pth')
            print('Saving')

        with open(SAVE_DIR + 'metrics.pickle', 'wb') as f:
            pickle.dump(metrics_dict, f)
        print()

    os.rename(SAVE_DIR, SAVE_DIR.replace(current_dir, current_dir + '_' + str(round(max_val_miou * 100, 2)) + '_' + str(
        round(max_val_miou_mo * 100, 2))))
