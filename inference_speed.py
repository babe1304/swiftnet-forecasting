import torch
from tqdm import tqdm, trange
import numpy as np

from models.f2f import *
from models.f2mf import *
from models.semseg import SemsegModel
from models.resnet.resnet_pyramid_forecast import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestep = 9
forecast_after_up_block = 3
pyramid_levels = 3

mean, std = None, None
mean = torch.tensor(
    np.load('/path/to/mean/numpy/array'),
    device=device)
std = torch.tensor(
    np.load('/path/to/std/numpy/array'),
    device=device)
do_normalization = mean is not None and std is not None

# f2f_model = DeformF2F_N(N=5, in_channels=512, out_channels=128, mean=mean, std=std, split_input_dconv=False)

# f2f_model = DeformF2F_N_corr(N=3, in_channels=512, out_channels=128, mean=mean, std=std, split_input_dconv=True)

f2mf_model = F2MF(in_channels=512, out_channels=128, mean=mean, std=std, patch_size=9)

# f2mf_model.load_state_dict(torch.load('/path/to/pretrained/model'), strict=False)
#
# backbone = resnet18(pretrained=True,
#                     pyramid_levels=pyramid_levels,
#                     k_upsample=3,
#                     scale=1,
#                     mean=[73.15, 82.90, 72.3],
#                     std=[47.67, 48.49, 47.73],
#                     k_bneck=1,
#                     output_stride=4,
#                     efficient=True,
#                     forecast_after_up_block=forecast_after_up_block)
#
# semseg_model = SemsegModel(backbone, 19, k=1, bias=True)
# semseg_model.load_state_dict(
#     torch.load('/path/to/pretrained/model'), strict=False)

print('Normalize:' + str(do_normalization))

dummy_input_feat_l = torch.randn((1, 512, 64, 32)).to(device)
dummy_input_feat_h = torch.randn((1, 512, 128, 64)).to(device)
dummy_input_img = torch.randn((1, 3, 1024, 2048)).to(device)
target_size = dummy_input_img.shape[-2:]
batch = {'image': dummy_input_img}

dummy_input = dummy_input_feat_h
model = f2mf_model

model.to(device)
model.eval()

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))

# GPU-WARM-UP
for _ in trange(100, desc='Warmup'):
    _ = model(dummy_input)
    # _ = model.forward_split(batch, image_size=target_size)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        # _ = model.forward_split(batch, image_size=target_size)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print('Mean inference time per sample: ' + str(mean_syn) + ' ms')

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of parameters: ' + str(round(pytorch_total_params / 1e6, 2)) + 'M')
print('Total number of trainable parameters: ' + str(round(pytorch_total_trainable_params / 1e6, 2)) + 'M')
