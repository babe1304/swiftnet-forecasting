import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.semseg import SemsegModel
from models.resnet.resnet_pyramid_forecast import *

from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import time
import pickle

from data.cityscapes.cityscapes import CombinedDataset
from evaluation.evaluate import ConfusionMatrix, compute_errors
from models.f2f import *
from models.f2mf import *
from data.cityscapes import *

from external_packages.taming.models.vqgan import VQModel
import yaml
from omegaconf import OmegaConf


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

moving_IDs = range(11, 18 + 1)
timestep = 9
f2f_levels = 3

forecast_after_up_block = 2
pyramid_levels = 2

begin_time = time.strftime("%H_%M_%d_%m_%Y", time.localtime())
custom_desc = '_2_levels_2_skips_RGB_multimodal_shared_512ch'
current_dir = 'F2MF_multimodal' + custom_desc + '_t' + str(timestep) + '_' + begin_time

SAVE_DIR = '/path/to/save/root/' + current_dir + '/'
os.mkdir(SAVE_DIR)
print('Saving in: ' + SAVE_DIR)

mean_rgb, std_rgb = None, None
mean_sem = torch.tensor(
    np.load('/path/to/mean/numpy/array'),
    device=device)
std_sem = torch.tensor(
    np.load('/path/to/std/numpy/array'),
    device=device)
mean_rgb = torch.tensor(
    np.load('/path/to/mean/numpy/array'),
    device=device)
std_rgb = torch.tensor(
    np.load('/path/to/std/numpy/array'),
    device=device)
do_normalization = mean_rgb is not None and std_rgb is not None

# model = DeformF2F_N(N=f2f_levels, in_channels=1024, out_channels=256, mean=mean, std=std, split_input_dconv=False)
# model = DeformF2F_N_corr(N=f2f_levels, in_channels=512, out_channels=128, mean=None, std=None)
model = F2MF_multihead(mean_sem=mean_sem, std_sem=std_sem, mean_rgb=mean_rgb, std_rgb=std_rgb)

config16384 = load_config("external_packages/taming/logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
reconstruction_model = load_vqgan(config16384, ckpt_path="external_packages/taming/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt")

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

segm_model = SemsegModel(backbone, 19, k=1, bias=True)
segm_model.load_state_dict(
    torch.load('/path/to/pretrained/model'), strict=False)

print('Normalize:' + str(do_normalization))

model.to(device)
reconstruction_model.to(device)
segm_model.to(device)
# print(model)
model.train()
reconstruction_model.eval()

# Hyperparameters
# initial_learning_rate1 = 5e-4
# initial_learning_rate1 = 1e-4
initial_learning_rate1 = 2e-4

batch_size = 12
num_epochs = 80
print(initial_learning_rate1)
# num_epochs2 = 5


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


class Scale(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return 2. * image - 1.


def evaluate(model, loader, epoch=None):
    model.eval()

    conf_matrix = ConfusionMatrix(num_classes=19, ignore_label=255)
    loss_l2_feats = []
    loss_l2_sem_feats = []
    loss_l2_rgb_feats = []
    loss_l2_rgb = []
    loss_ce = []

    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), desc='Testing') as progress:
            for batch_idx, batch in progress:
                batch_sem, batch_rgb = batch

                past_feats_sem, target_feats_sem, sem_seg_gt = batch_sem
                past_feats_rgb, target_feats_rgb, target_img_rgb = batch_rgb

                past_feats_sem = past_feats_sem.to(device)
                target_feats_sem = target_feats_sem.to(device)
                past_feats_rgb = past_feats_rgb.to(device)
                target_feats_rgb = target_feats_rgb.to(device)
                sem_seg_gt = sem_seg_gt.squeeze(dim=1).to(device)
                target_img_rgb = target_img_rgb.to(device)

                pred_feats_sem, pred_feats_rgb, additional = model.forward(past_feats_sem, past_feats_rgb, additional_dict=True)

                loss, loss_sem, loss_rgb = model.loss(additional, target_feats_sem, target_feats_rgb, additional_ret=True)
                loss_l2_feats.append(loss.cpu().item())
                loss_l2_sem_feats.append(loss_sem.cpu().item())
                loss_l2_rgb_feats.append(loss_rgb.cpu().item())

                # imgrec = reconstruction_model.decode(pred_feats_rgb)
                imgrec = reconstruction_model.decode_with_quant(pred_feats_rgb)
                # imgrec = reconstruction_model.decode_with_quant_v2(pred_feats_rgb)

                loss_l2_rgb.append(F.mse_loss(imgrec, target_img_rgb).cpu().item())

                logits = segm_model.forward_decoder_no_skip(pred_feats_sem, image_size=sem_seg_gt.shape[-2:])
                probs = logits.softmax(dim=1)
                preds = torch.argmax(logits, 1)
                loss_ce.append(F.cross_entropy(probs, sem_seg_gt, ignore_index=255).cpu().item())

                conf_matrix.update(preds.cpu().numpy().flatten(), sem_seg_gt.cpu().numpy().flatten())

                if batch_idx == 0:
                    pil_img = custom_to_pil(imgrec.squeeze())
                    # pil_img.show()

                    # print('\n[Features]       Min: ', torch.min(pred_feats.detach()).cpu().item(), ' Max: ',
                    #       torch.max(pred_feats.detach()).cpu().item(), ' Mean: ', torch.mean(pred_feats.detach()).cpu().item())
                    # print('[Norm Features]  Min: ', torch.min(normalized_pred.detach()).cpu().item(), ' Max: ',
                    #       torch.max(normalized_pred.detach()).cpu().item(), ' Mean: ', torch.mean(normalized_pred.detach()).cpu().item())
                    # print('[Targ Features]  Min: ', torch.min(target_feats.detach()).cpu().item(), ' Max: ',
                    #       torch.max(target_feats.detach()).cpu().item(), ' Mean: ', torch.mean(target_feats.detach()).cpu().item())
                    # print('[Reconstruction] Min: ', torch.min(imgrec.detach()).cpu().item(), ' Max: ',
                    #       torch.max(imgrec.detach()).cpu().item(), ' Mean: ', torch.mean(imgrec.detach()).cpu().item())

                    if not os.path.exists(SAVE_DIR + 'samples'):
                        os.mkdir(SAVE_DIR + 'samples')

                    pil_img.save(SAVE_DIR + 'samples/epoch_' + str(epoch) + '.png')
                    # pil_img.show()
                #
                #     import time
                #     time.sleep(20)
                #     # print(pred_feats)

        print("--- TEST ---")
        mse_feat = np.mean(np.array(loss_l2_feats))
        mse_feat_sem = np.mean(np.array(loss_l2_sem_feats))
        mse_feat_rgb = np.mean(np.array(loss_l2_rgb_feats))
        mse_rgb = np.mean(np.array(loss_l2_rgb))
        ce = np.mean(np.array(loss_ce))
        print("L2 (MSE) feature loss: ", mse_feat)
        print("L2 (MSE) semantic feature loss: ", mse_feat_sem)
        print("L2 (MSE) reconstruction feature loss: ", mse_feat_rgb)
        print("L2 (MSE) reconstruction loss: ", mse_rgb)
        print("Cross entropy loss: ", ce)
        miou, per_class_iou = conf_matrix.get_metrics(verbose=False)
        miou_mo = conf_matrix.get_subset_miou(moving_IDs)
        print("mIoU: ", round(miou * 100, 2), '%')
        print("mIoU-MO: ", round(miou_mo * 100, 2), '%')

    model.train()
    return mse_feat, mse_feat_sem, mse_feat_rgb, mse_rgb, ce, miou, miou_mo


if __name__ == '__main__':

    pil_size = (1024, 512)
    # size = (512, 1024)
    rgb_target_transform = T.Compose(
        [
         # T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
         Scale()
         ])

    # sem_target_transform = T.Compose(
    #     [
    #      T.Resize(size, interpolation=T.InterpolationMode.NEAREST)
    #      ])

    dataset_train_rgb = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        '/path/to/RGB/sequences',
        delta=timestep, subset='train', sem_labels=False, extended=False)

    dataset_val_rgb = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        '/path/to/RGB/sequences',
        delta=timestep, subset='val', sem_labels=False, extended=False, target_transform=rgb_target_transform, resize_target=pil_size)

    dataset_train_sem = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        delta=timestep, subset='train', sem_labels=True, extended=False, flip=False)

    dataset_val_sem = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        delta=timestep, subset='val', sem_labels=True, extended=False, flip=False, resize_target=pil_size)

    dataset_train = CombinedDataset(dataset_train_sem, dataset_train_rgb)
    dataset_val = CombinedDataset(dataset_val_sem, dataset_val_rgb)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=1,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=None, num_workers=1,
                            persistent_workers=True, prefetch_factor=4)

    metrics_dict = {
        'val_mse_feat': [],
        'val_mse_feat_sem': [],
        'val_mse_feat_rgb': [],
        'val_mse_rgb': [],
        'val_ce': [],
        'val_miou': [],
        'val_miou_mo': [],
        'train_mse_feat': [],
        'train_mse_feat_sem': [],
        'train_mse_feat_rgb': [],
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate1)
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(semseg_model.parameters()), lr=learning_rate1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=5e-9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94, verbose=True)

    min_val_mse_feat = np.inf

    for epoch in range(1, num_epochs + 1):
        loss_mse_feat = []
        loss_mse_feat_sem = []
        loss_mse_feat_rgb = []

        # print('LR:', scheduler.get_last_lr()[0])

        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f'Training (epoch={epoch}/{num_epochs})') as epoch_progress:
            for batch_idx, batch in epoch_progress:

                batch_sem, batch_rgb = batch

                past_feats_sem, target_feats_sem, sem_seg_gt = batch_sem
                past_feats_rgb, target_feats_rgb, target_img_rgb = batch_rgb

                past_feats_sem = past_feats_sem.to(device)
                target_feats_sem = target_feats_sem.to(device)

                past_feats_rgb = past_feats_rgb.to(device)
                target_feats_rgb = target_feats_rgb.to(device)

                # sem_seg_gt = sem_seg_gt.to(device)
                # print(past_feats.shape, target_feats.shape)

                optimizer.zero_grad()

                pred_feats_sem, pred_feats_rgb, additional = model.forward(past_feats_sem, past_feats_rgb, additional_dict=True)

                loss, loss_sem, loss_rgb = model.loss(additional, target_feats_sem, target_feats_rgb, additional_ret=True)
                loss_mse_feat.append(loss.cpu().item())
                loss_mse_feat_sem.append(loss_sem.cpu().item())
                loss_mse_feat_rgb.append(loss_rgb.cpu().item())

                # logits = semseg_model.forward_decoder_no_skip(pred_feats, image_size=sem_seg_gt.shape[1:])
                # preds = torch.argmax(logits, 1)
                # semseg_loss = loss_fn_ce(logits, sem_seg_gt)
                # loss = loss + semseg_loss
                # loss_ce.append(semseg_loss.cpu().item())

                # conf_matrix.update(preds.cpu().numpy().flatten(), sem_seg_gt.cpu().numpy().flatten())

                loss.backward()
                optimizer.step()
                # scheduler.step()

        scheduler.step()

        print("--- TRAIN ---")
        train_mse_feat = np.mean(np.array(loss_mse_feat))
        train_mse_feat_sem = np.mean(np.array(loss_mse_feat_sem))
        train_mse_feat_rgb = np.mean(np.array(loss_mse_feat_rgb))
        # train_ce = np.mean(np.array(loss_ce))
        print("L2 (MSE) feature loss: ", train_mse_feat)
        print("L2 (MSE) semantic feature loss: ", train_mse_feat_sem)
        print("L2 (MSE) reconstruction feature loss: ", train_mse_feat_rgb)
        # print("Cross entropy loss: ", train_ce)
        # train_miou, train_per_class_iou = conf_matrix.get_metrics(verbose=False)
        # train_miou_mo = conf_matrix.get_subset_miou(moving_IDs)
        # print("mIoU: ",  round(train_miou * 100, 2), '%')
        # print("mIoU-MO: ", round(train_miou_mo * 100, 2), '%')

        # conf_matrix.reset()

        val_mse_feat, val_mse_feat_sem, val_mse_feat_rgb, val_mse_rgb, val_ce, val_miou, val_miou_mo = evaluate(model, val_loader, epoch=epoch)
        metrics_dict['val_mse_feat'].append(val_mse_feat)
        metrics_dict['val_mse_feat_sem'].append(val_mse_feat_sem)
        metrics_dict['val_mse_feat_rgb'].append(val_mse_feat_rgb)
        metrics_dict['val_mse_rgb'].append(val_mse_rgb)
        metrics_dict['val_ce'].append(val_ce)
        metrics_dict['val_miou'].append(val_miou)
        metrics_dict['val_miou_mo'].append(val_miou_mo)
        metrics_dict['train_mse_feat'].append(train_mse_feat)
        metrics_dict['train_mse_feat_sem'].append(train_mse_feat_sem)
        metrics_dict['train_mse_feat_rgb'].append(train_mse_feat_rgb)

        if val_mse_feat < min_val_mse_feat:
            min_val_mse_feat = val_mse_feat
            model.eval()
            torch.save(model.state_dict(), SAVE_DIR + 'model.pth')
            model.train()
            # torch.save(semseg_model.state_dict(), SAVE_DIR + 'semseg_model.pth')
            print('Saving')

        with open(SAVE_DIR + 'metrics.pickle', 'wb') as f:
            pickle.dump(metrics_dict, f)
        print()

    os.rename(SAVE_DIR, SAVE_DIR.replace(current_dir, current_dir + '_' + str(round(min_val_mse_feat, 3)) + '_' + str(
        round(min_val_mse_feat, 3))))
