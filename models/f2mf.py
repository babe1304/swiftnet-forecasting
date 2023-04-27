import torch
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    # mask = torch.ones(x.size()).cuda()
    # mask = F.grid_sample(mask, vgrid, align_corners=True)
    #
    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1
    #
    # return output * mask
    return output


def normalize(x, mean, std):
    return (x - mean) / std


def normalize_concatenated(x, mean, std):
    base_c = mean.shape[1]
    for i in range(4):
        x[:, i * base_c:(i + 1) * base_c, :, :] -= mean
        x[:, i * base_c:(i + 1) * base_c, :, :] /= std
    return x


def unnormalize(x, mean, std):
    return (x * std) + mean


class BN_ReLU_DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, offset_kernel_size=None, custom_offset_op=None):
        super(BN_ReLU_DConv, self).__init__()

        if offset_kernel_size is None:
            offset_kernel_size = kernel_size

        self.bn = torch.nn.BatchNorm2d(in_channels)

        if custom_offset_op is not None:
            self.offset_conv = custom_offset_op
        else:
            self.offset_conv = torch.nn.Conv2d(in_channels=in_channels,
                                               out_channels=2 * kernel_size * kernel_size,
                                               kernel_size=offset_kernel_size,
                                               stride=1,
                                               padding=int((offset_kernel_size - 1) / 2),
                                               bias=True)

        self.dconv = DeformConv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=int((kernel_size - 1) / 2),
                                  bias=True)

    def forward(self, x):
        x = torch.relu(self.bn(x))
        offsets = self.offset_conv(x)
        x = self.dconv(x, offsets)
        return x


class BN_ReLU_Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BN_ReLU_Conv, self).__init__()

        self.bn = torch.nn.BatchNorm2d(in_channels)

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=int((kernel_size - 1) / 2),
                                    bias=False)

    def forward(self, x):
        x = torch.relu(self.bn(x))
        x = self.conv(x)
        return x


class CorrelationModule(torch.nn.Module):
    def __init__(self, in_channels_one, out_channels_one=128, patch_size=9, kernel_size=1):
        super(CorrelationModule, self).__init__()

        self.in_channels_one = in_channels_one
        self.out_channels_one = out_channels_one
        self.patch_size = patch_size

        self.bn_relu_conv = BN_ReLU_Conv(in_channels_one, out_channels_one, 3)
        self.corr_layer = SpatialCorrelationSampler(
            kernel_size=kernel_size,
            patch_size=patch_size,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=2)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = [x[:, self.in_channels_one * i: self.in_channels_one * (i + 1), :, :] for i in range(4)]  # split by T
        conv_feats = [self.bn_relu_conv(x) for x in feats]
        norm_feats = [x / x.norm(dim=1, keepdim=True) for x in conv_feats]

        corr_list = []
        for i in range(4 - 1):
            corr_list.append(self.corr_layer(norm_feats[i], norm_feats[i + 1]).reshape(B, self.patch_size ** 2, H, W))

        corr_feats = torch.cat(corr_list, dim=1)

        return corr_feats


class F2MF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mean=None, std=None):
        super(F2MF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean, self.std = None, None

        self.do_normalization = mean is not None and std is not None
        if self.do_normalization:
            self.mean = mean.reshape((1, mean.shape[0], 1, 1))
            self.std = std.reshape((1, std.shape[0], 1, 1))

        patch_size = 9
        input_frames = 4
        self.correlation = CorrelationModule(in_channels_one=out_channels, out_channels_one=128,
                                             patch_size=patch_size)
        self.fusion = BN_ReLU_DConv(in_channels, out_channels, 1)
        in_channels_with_corr = out_channels + patch_size ** 2 * (input_frames - 1)

        # shared dconv
        shared_dconv_layers = []
        in_dconv = BN_ReLU_DConv(in_channels_with_corr, out_channels * 2, 3)
        shared_dconv_layers.append(in_dconv)

        for i in range(5):
            shared_dconv_layers.append(BN_ReLU_DConv(out_channels * 2, out_channels * 2, 3))

        self.shared_dconv = torch.nn.Sequential(*shared_dconv_layers)

        # F2M
        self.f2m_head = BN_ReLU_DConv(out_channels * 2, input_frames * 2, 3)

        # F2F
        self.f2f_head = BN_ReLU_DConv(out_channels * 2, out_channels, 3)

        # weights
        self.weight_head = BN_ReLU_DConv(out_channels * 2, input_frames + 1, 3)

        print(self.correlation, self.fusion, self.shared_dconv, self.f2m_head, self.f2f_head, self.weight_head)

    def forward(self, input, additional_dict=False):
        if self.do_normalization:
            input = normalize_concatenated(input, self.mean, self.std)

        B, C, H, W = input.shape

        fused = self.fusion(input)
        corr = self.correlation(input)

        combined = torch.cat((fused, corr), dim=1)
        x = self.shared_dconv(combined)

        flow = self.f2m_head(x)
        predicted = self.f2f_head(x)
        weights = self.weight_head(x)
        probs = F.softmax(weights, dim=1)

        warped = warp(input.reshape(B * 4, self.out_channels, H, W), flow.reshape(B * 4, 2, H, W))
        warped = warped.reshape(B, self.in_channels, H, W)

        x = probs[:, 0:1, :, :] * predicted
        for i in range(4):
            x += probs[:, i + 1:i + 2, :, :] * warped[:, i * self.out_channels: (i + 1) * self.out_channels, :, :]

        if self.do_normalization:
            unnormalized_x = unnormalize(x, self.mean, self.std)
        else:
            unnormalized_x = x

        if additional_dict:
            additional = dict()
            additional['predicted'] = predicted
            additional['weights'] = weights
            additional['warped'] = warped
            additional['output'] = x
            return unnormalized_x, additional

        return unnormalized_x

    def loss(self, additional, target, coef=(1.0, 1.0, 1.0)):
        if self.do_normalization:
            target = normalize(target, self.mean, self.std)

        # compute f2m prediction
        f2m_weights = additional['weights'][:, 1:, :, :]
        probs = F.softmax(f2m_weights, dim=1)
        f2m_out = torch.zeros_like(additional['output'], device=additional['output'].device)
        for i in range(4):
            f2m_out += probs[:, i:i + 1, :, :] * additional['warped'][:,
                                                 i * self.out_channels: (i + 1) * self.out_channels, :, :]

        loss_output = F.mse_loss(additional['output'], target)
        loss_f2f = F.mse_loss(additional['predicted'], target)
        loss_f2m = F.mse_loss(f2m_out, target)

        return coef[0] * loss_output + coef[1] * loss_f2f + coef[2] * loss_f2m


if __name__ == '__main__':

    # just for testing
    x = torch.cat(
        [torch.ones(2, 4, 3, 3), torch.ones(2, 4, 3, 3) * 2, torch.ones(2, 4, 3, 3) * 3, torch.ones(2, 4, 3, 3) * 4],
        dim=1)
    w = torch.ones(2, 4, 3, 3)
    w[:, 0, :, :] *= 0.1
    w[:, 1, :, :] *= 0.2
    w[:, 2, :, :] *= 0.3
    w[:, 3, :, :] *= 0.4
    print(x[:, 0, :, :] * w[:, 0, :, :])

    F2MF(512, 128).cuda().forward(torch.randn(12, 512, 32, 64).cuda())

    print(list(F2MF(512, 128).named_parameters()))

    exit()

    import numpy as np

    p1 = '/run/media/jakov/2TB KC3000/Users/bubas/Data/Cityscapes/features/2_levels/1_skip_32x64/val/frankfurt_000000_001001_leftImg8bit.npy'
    p2 = '/run/media/jakov/2TB KC3000/Users/bubas/Data/Cityscapes/features/2_levels/1_skip_32x64/val/frankfurt_000000_001007_leftImg8bit.npy'

    x1 = torch.from_numpy(np.load(p1)).unsqueeze(0).cuda()
    x2 = torch.from_numpy(np.load(p2)).unsqueeze(0).cuda()
    # x1 = normalize(x1, x1.mean(), x1.std())
    # x2 = normalize(x2, x2.mean(), x2.std())

    corr_layer = SpatialCorrelationSampler(
        kernel_size=1,
        patch_size=9,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=2)

    x = torch.randn(12, 512, 32, 64)
    B, C, H, W = x.shape
    feats = [x[:, 128 * i: 128 * (i + 1), :, :] for i in range(4)]  # split by T
    # conv_feats = [self.bn_relu_conv(x) for x in feats]
    # norm_feats = [x / x.norm(dim=1, keepdim=True) for x in feats]

    c_feats = torch.cat(feats, dim=1)
    norm_feats = c_feats / c_feats.norm(dim=1, keepdim=True)
    a = norm_feats.reshape(B * 4, 128, H, W)
    b = norm_feats.reshape(B, C, H, W)

    x = torch.randn(12, 512, 32, 64).cuda()
    flow = torch.randn(12, 2, 32, 64).cuda()

    a = torch.cat([warp(x[:, 128 * i: 128 * (i + 1), :, :], flow) for i in range(4)], dim=1)
    b = warp(x.reshape(B * 4, 128, H, W), torch.cat([flow, flow, flow, flow], dim=1).reshape(B * 4, 2, H, W)).reshape(B,
                                                                                                                      512,
                                                                                                                      H,
                                                                                                                      W)
    print(a.shape)
    print(b.shape)
    print(torch.allclose(a, b))

    c = warp(x[:, 0: 128, :, :], flow)
    d = warp(x.reshape(B * 4, 128, H, W), torch.cat([flow, flow, flow, flow], dim=0))[:12, :, :, :]
    print(c.shape)
    print(d.shape)
    print(torch.allclose(c, d))

    print(a[3][0][0])
    print(b[3][0][0])

    [print(x.shape) for x in feats]
    exit()

    print(x1.shape)
    xs = torch.cat((x1, x2, x1, x2), dim=1)
    print(xs.shape)
    # corr1 = CorrelationModule(in_channels_one=128).cuda()
    # out1 = corr1.forward(xs)
    # print(out1.shape)

    corr1 = Correlation(pad_size=4 + int(3 // 2), kernel_size=3,
                        max_displacement=4, stride1=1, stride2=1, corr_multiply=1)

    correlation_sampler = SpatialCorrelationSampler(
        kernel_size=1,
        patch_size=9,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=2)

    from tqdm import tqdm

    # for i in tqdm(range(10000)):
    #     out1 = corr1(x1, x2)
    #     out1[0,0,0,0] = 1
    #
    # for i in tqdm(range(10000)):
    #     out2 = correlation_sampler(x1, x2)
    #     out2[0, 0, 0, 0] = 1

    for i in tqdm(range(100)):
        out1 = corr1(x1, x2)

    for i in tqdm(range(100)):
        out2 = correlation_sampler(x1, x2)

    print(out1.shape)
    print(out2.shape)

    # print(x2.shape)

    # myCorr(x1, x2)

    # kernel_size = 1
    # max_displacement = 4
    # corrLayer = Correlation(pad_size=max_displacement + int(kernel_size // 2), kernel_size=kernel_size,
    #                         max_displacement=max_displacement, stride1=1, stride2=1, corr_multiply=1)

    # out1 = corrLayer.forward(x1, x2)
    #
    # print(out1.shape)

    # a = torch.randn(2, 3, 20, 30)
    # b = torch.randn(2, 3, 20, 30)
    # test_layer = CorrelationLayer(padding=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=2)
    #
    # print(test_layer(x1, x2))

    out = out2.squeeze().cpu().reshape(81, 32, 64).numpy()

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(9, 9)
    for i in range(9):
        for j in range(9):
            x = out[i * j + j, :, :]
            axs[i, j].imshow(x)

    plt.show()
