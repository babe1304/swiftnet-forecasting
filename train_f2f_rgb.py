import torch
import torchvision.transforms as T
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
f2f_levels = 8

forecast_after_up_block = 2
pyramid_levels = 3

begin_time = time.strftime("%H_%M_%d_%m_%Y", time.localtime())
custom_desc = '_RGB_no_quant_expLR_conv1x1_only_last_t_no_norm_delete'
current_dir = 'F2F-' + str(f2f_levels) + custom_desc + '_t' + str(timestep) + '_' + begin_time

SAVE_DIR = '/path/to/save/root/' + str(
    pyramid_levels) + '_levels/' + current_dir + '/'
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

f2f_model = DeformF2F_N(N=f2f_levels, in_channels=1024, out_channels=256, mean=mean, std=std, split_input_dconv=False)
# f2f_model = DeformF2F_N_corr(N=f2f_levels, in_channels=512, out_channels=128, mean=None, std=None)

config16384 = load_config("external_packages/taming/logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
reconstruction_model = load_vqgan(config16384, ckpt_path="external_packages/taming/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt")

print('Normalize:' + str(do_normalization))

f2f_model.to(device)
reconstruction_model.to(device)
# print(model)
f2f_model.train()
reconstruction_model.eval()

# Hyperparameters
# initial_learning_rate1 = 5e-4
# initial_learning_rate1 = 1e-4
initial_learning_rate1 = 2e-4

batch_size = 12
num_epochs = 160
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


def evaluate(f2f_model, segm_model, loader, epoch=None):
    f2f_model.eval()

    loss_l2 = []
    loss_l2_rgb = []
    # loss_ce = []

    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), desc='Testing') as progress:
            for batch_idx, batch in progress:
                past_feats, target_feats, target_rgb = batch

                past_feats = past_feats.to(device)
                target_feats = target_feats.to(device)
                target_rgb = target_rgb.to(device)

                pred_feats, normalized_pred = f2f_model.forward(past_feats, additional=True)

                loss = f2f_model.loss(normalized_pred, target_feats)
                loss_l2.append(loss.cpu().item())

                # imgrec = reconstruction_model.decode(pred_feats)
                imgrec = reconstruction_model.decode_with_quant(pred_feats)
                # imgrec = reconstruction_model.decode_with_quant_v2(pred_feats)

                loss_l2_rgb.append(F.mse_loss(imgrec, target_rgb).cpu().item())

                if batch_idx == 0:
                    pil_img = custom_to_pil(imgrec.squeeze())
                    # pil_img.show()

                    print('\n[Features]       Min: ', torch.min(pred_feats.detach()).cpu().item(), ' Max: ',
                          torch.max(pred_feats.detach()).cpu().item(), ' Mean: ', torch.mean(pred_feats.detach()).cpu().item())
                    print('[Norm Features]  Min: ', torch.min(normalized_pred.detach()).cpu().item(), ' Max: ',
                          torch.max(normalized_pred.detach()).cpu().item(), ' Mean: ', torch.mean(normalized_pred.detach()).cpu().item())
                    print('[Targ Features]  Min: ', torch.min(target_feats.detach()).cpu().item(), ' Max: ',
                          torch.max(target_feats.detach()).cpu().item(), ' Mean: ', torch.mean(target_feats.detach()).cpu().item())
                    print('[Reconstruction] Min: ', torch.min(imgrec.detach()).cpu().item(), ' Max: ',
                          torch.max(imgrec.detach()).cpu().item(), ' Mean: ', torch.mean(imgrec.detach()).cpu().item())

                    if not os.path.exists(SAVE_DIR + 'samples'):
                        os.mkdir(SAVE_DIR + 'samples')

                    pil_img.save(SAVE_DIR + 'samples/epoch_' + str(epoch) + '.png')
                    # pil_img.show()
                #
                #     import time
                #     time.sleep(20)
                #     # print(pred_feats)


                # logits = segm_model.forward_decoder_no_skip(pred_feats, image_size=sem_seg_gt.shape[-2:])
                # preds = torch.argmax(logits, 1)
                # loss_ce.append(loss_fn_ce(logits, sem_seg_gt).cpu().item())
                #
                # conf_matrix.update(preds.cpu().numpy().flatten(), sem_seg_gt.cpu().numpy().flatten())

        print("--- TEST ---")
        # pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_matrix.get_matrix(), Cityscapes.class_info, verbose=True)
        mse = np.mean(np.array(loss_l2))
        mse_rgb = np.mean(np.array(loss_l2_rgb))
        print("L2 (MSE) loss: ", mse)
        print("L2 (MSE) reconstruction loss: ", mse_rgb)

    f2f_model.train()
    return mse, mse_rgb


if __name__ == '__main__':

    size = (512, 1024)
    val_target_transform = T.Compose(
        [T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
         T.ToTensor(),
         Scale()
         ])

    dataset_train = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        '/path/to/RGB/sequences',
        delta=9, subset='train', sem_labels=False, extended=False)

    dataset_val = CityscapesFeatureSequence(
        '/path/to/saved/features',
        '/path/to/ground/truths',
        '/path/to/RGB/sequences',
        delta=9, subset='val', sem_labels=False, extended=False, target_transform=val_target_transform)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=None, num_workers=1,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=None, num_workers=1,
                            persistent_workers=True, prefetch_factor=4)

    metrics_dict = {
        'val_miou': [],
        'val_miou_mo': [],
        'val_mse': [],
        'val_mse_rgb': [],
        'val_ce': [],
        'train_miou': [],
        'train_miou_mo': [],
        'train_mse': [],
        'train_ce': []
    }

    optimizer = torch.optim.Adam(f2f_model.parameters(), lr=initial_learning_rate1)
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(semseg_model.parameters()), lr=learning_rate1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=5e-9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94, verbose=True)

    min_val_mse = np.inf

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
                # scheduler.step()

        scheduler.step()

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

        val_mse, val_mse_rgb = evaluate(f2f_model, None, val_loader, epoch=epoch)
        # metrics_dict['val_miou'].append(val_miou)
        # metrics_dict['val_miou_mo'].append(val_miou_mo)
        metrics_dict['val_mse'].append(val_mse)
        metrics_dict['val_mse_rgb'].append(val_mse_rgb)
        # metrics_dict['val_ce'].append(val_ce)
        metrics_dict['train_mse'].append(train_mse)
        # metrics_dict['train_ce'].append(train_ce)
        # metrics_dict['train_miou'].append(train_miou)
        # metrics_dict['train_miou_mo'].append(train_miou_mo)

        if val_mse < min_val_mse:
            min_val_mse = val_mse
            f2f_model.eval()
            torch.save(f2f_model.state_dict(), SAVE_DIR + 'f2f_model.pth')
            f2f_model.train()
            # torch.save(semseg_model.state_dict(), SAVE_DIR + 'semseg_model.pth')
            print('Saving')

        with open(SAVE_DIR + 'metrics.pickle', 'wb') as f:
            pickle.dump(metrics_dict, f)
        print()

    os.rename(SAVE_DIR, SAVE_DIR.replace(current_dir, current_dir + '_' + str(round(min_val_mse, 3)) + '_' + str(
        round(min_val_mse, 3))))
