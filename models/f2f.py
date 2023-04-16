import torch
from torchvision.ops import DeformConv2d


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


class Split_BN_ReLU_DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Split_BN_ReLU_DConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = torch.nn.BatchNorm2d(in_channels)

        self.offset_convs = torch.nn.ModuleList()
        self.split_dconvs = torch.nn.ModuleList()
        for i in range(int(in_channels / out_channels)):
            offset_conv = torch.nn.Conv2d(in_channels=in_channels,
                                          out_channels=2 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=int((kernel_size - 1) / 2),
                                          bias=True)

            dconv = DeformConv2d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=int((kernel_size - 1) / 2),
                                 bias=True)

            self.offset_convs.append(offset_conv)
            self.split_dconvs.append(dconv)

    def forward(self, x):
        x = torch.relu(self.bn(x))

        feats = []
        out = None
        # print(x.shape)
        for i in range(int(self.in_channels / self.out_channels)):
            offsets = self.offset_convs[i](x)
            x_i = x[:, i * self.out_channels:(i + 1) * self.out_channels, :, :]
            # print(i * self.out_channels,(i+1) * self.out_channels)
            # print(x_i.shape)
            x_i = self.split_dconvs[i](x_i, offsets)
            # print(x_i.shape)
            # feats.append(x_i)
            if out is None:
                out = x_i
            else:
                out = out + x_i
            # if out is None:
            #     out = x_i
            # else:
            #     out = torch.cat((out, x_i), dim=1)

        # x = torch.cat(feats, dim=1)
        # print(x.shape)
        return out


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


class DeformF2F_N(torch.nn.Module):
    def __init__(self, N, in_channels, out_channels, mean=None, std=None, split_input_dconv=False):
        super(DeformF2F_N, self).__init__()
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean, self.std = None, None

        self.do_normalization = mean is not None and std is not None
        if self.do_normalization:
            self.mean = mean.reshape((1, mean.shape[0], 1, 1))
            self.std = std.reshape((1, std.shape[0], 1, 1))

        layers = []

        if split_input_dconv:
            layers.append(Split_BN_ReLU_DConv(in_channels, out_channels, 1))
            # layers.append(BN_ReLU_Conv(in_channels, out_channels*2, 1))
            # layers.append(BN_ReLU_Conv(out_channels*2, out_channels, 1))
        else:
            layers.append(BN_ReLU_DConv(in_channels, out_channels, 1))
            # layers.append(BN_ReLU_DConv(in_channels, out_channels, 1, offset_kernel_size=3))
            # layers.append(BN_ReLU_DConv(in_channels, out_channels, 3))

            # layers.append(BN_ReLU_DConv(out_channels, out_channels, 3))
            # layers.append(BN_ReLU_DConv(out_channels, out_channels*2, 3))
            # layers.append(BN_ReLU_DConv(out_channels*2, out_channels*2, 3))
            # layers.append(BN_ReLU_DConv(out_channels*2, out_channels*2, 3))
            # layers.append(BN_ReLU_DConv(out_channels*2, out_channels*2, 3))
            # layers.append(BN_ReLU_DConv(out_channels*2, out_channels, 3))
            # layers.append(BN_ReLU_DConv(out_channels, out_channels, 3))


        for i in range(N - 1):
            layers.append(BN_ReLU_DConv(out_channels, out_channels, 3))
            layers.append(BN_ReLU_DConv(out_channels, out_channels, 3, offset_kernel_size=3))
            layers.append(BN_ReLU_DConv(out_channels, out_channels, 3))

        self.model = torch.nn.Sequential(*layers)
        print(self.model)

    def forward(self, x):
        if not self.do_normalization:
            return self.model(x)

        x = normalize_concatenated(x, self.mean, self.std)
        x = self.model(x)
        # x = unnormalize(x, self.mean, self.std)
        return x

    def normalize(self, x):
        return normalize(x, self.mean, self.std)

    def normalize_concatenated(self, x):
        return normalize_concatenated(x, self.mean, self.std)

    def unnormalize(self, x):
        return unnormalize(x, self.mean, self.std)


class BN_ReLU_DConvCeption(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BN_ReLU_DConvCeption, self).__init__()

        self.bn = torch.nn.BatchNorm2d(in_channels)

        self.offset_conv = torch.nn.Conv2d(in_channels=in_channels,
                                           out_channels=2 * kernel_size * kernel_size,
                                           kernel_size=kernel_size,
                                           stride=1,
                                           padding=int((kernel_size - 1) / 2),
                                           bias=True)

        self.offset_dconv = DeformConv2d(in_channels=in_channels,
                                         out_channels=2 * kernel_size * kernel_size,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding=int((kernel_size - 1) / 2),
                                         bias=True)

        self.dconv = DeformConv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=int((kernel_size - 1) / 2),
                                  bias=True)

    def forward(self, x):
        x = torch.relu(self.bn(x))
        conv_offsets = self.offset_conv(x)
        dconv_offsets = self.offset_dconv(x, conv_offsets)
        x = self.dconv(x, dconv_offsets)
        return x


class DeformCeptionF2F_N(torch.nn.Module):
    def __init__(self, N, in_channels, out_channels, mean=None, std=None):
        super(DeformCeptionF2F_N, self).__init__()
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean, self.std = None, None

        self.do_normalization = mean is not None and std is not None
        if self.do_normalization:
            self.mean = mean.reshape((1, mean.shape[0], 1, 1))
            self.std = std.reshape((1, std.shape[0], 1, 1))

        layers = []
        layers.append(BN_ReLU_DConvCeption(in_channels, out_channels, 1))
        for i in range(N - 1):
            layers.append(BN_ReLU_DConvCeption(out_channels, out_channels, 3))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        if not self.do_normalization:
            return self.model(x)

        x = normalize_concatenated(x, self.mean, self.std)
        x = self.model(x)
        # x = unnormalize(x, self.mean, self.std)
        return x

    def normalize(self, x):
        return normalize(x, self.mean, self.std)

    def normalize_concatenated(self, x):
        return normalize_concatenated(x, self.mean, self.std)

    def unnormalize(self, x):
        return unnormalize(x, self.mean, self.std)


if __name__ == '__main__':
    x = torch.rand((1, 512, 32, 16))
    layer1 = BN_ReLU_DConv(512, 128, 1)
    layer2 = BN_ReLU_DConv(128, 128, 3)

    print(x.shape)
    x = layer1(x)
    print(x.shape)
    x = layer2(x)
    print(x.shape)

    x = torch.rand((4, 512, 32, 16))
    model = DeformF2F_N(8, 512, 128)

    print(x.shape)
    x = model(x)
    print(x.shape)

    print('Split DConv test:')
    x = torch.rand((1, 512, 32, 16))
    print(x.shape)
    split_dconv = Split_BN_ReLU_DConv(512, 128, 1)
    x = split_dconv(x)
    print(x.shape)

    print('Kernel size test:')
    x = torch.rand((1, 512, 32, 16))
    print(x.shape)
    dconv = BN_ReLU_DConv(512, 128, 1, offset_kernel_size=5)
    dconv(x)
    print(x.shape)


'''
class DeformF2F_5(torch.nn.Module):
    def __init__(self):
        super(DeformF2F_5, self).__init__()

        self.dconv1 = DeformConv2d(in_channels=128*4, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dconv2 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv3 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv4 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv5 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=128*4, out_channels=2*1*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        offset = self.conv1(x)
        x = self.dconv1(x, offset=offset)
        x = torch.relu(self.bn1(x))

        offset = self.conv2(x)
        x = self.dconv2(x, offset=offset)
        x = torch.relu(self.bn2(x))

        offset = self.conv3(x)
        x = self.dconv3(x, offset=offset)
        x = torch.relu(self.bn3(x))

        offset = self.conv4(x)
        x = self.dconv4(x, offset=offset)
        x = torch.relu(self.bn4(x))

        offset = self.conv5(x)
        x = self.dconv5(x, offset=offset)
        x = torch.relu(self.bn5(x))

        return x


class DeformF2F_8(torch.nn.Module):
    def __init__(self):
        super(DeformF2F_8, self).__init__()

        self.dconv1 = DeformConv2d(in_channels=128*4, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dconv2 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv3 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv4 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv5 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv6 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv7 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv8 = DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=128*4, out_channels=2*1*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = torch.nn.Conv2d(in_channels=128, out_channels=2*3*3, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1 = torch.nn.BatchNorm2d(128)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(128)
        self.bn8 = torch.nn.BatchNorm2d(128)

    def forward(self, x):
        offset = self.conv1(x)
        x = self.dconv1(x, offset=offset)
        x = torch.relu(self.bn1(x))

        offset = self.conv2(x)
        x = self.dconv2(x, offset=offset)
        x = torch.relu(self.bn2(x))

        offset = self.conv3(x)
        x = self.dconv3(x, offset=offset)
        x = torch.relu(self.bn3(x))

        offset = self.conv4(x)
        x = self.dconv4(x, offset=offset)
        x = torch.relu(self.bn4(x))

        offset = self.conv5(x)
        x = self.dconv5(x, offset=offset)
        x = torch.relu(self.bn5(x))

        offset = self.conv6(x)
        x = self.dconv6(x, offset=offset)
        x = torch.relu(self.bn6(x))

        offset = self.conv7(x)
        x = self.dconv7(x, offset=offset)
        x = torch.relu(self.bn7(x))

        offset = self.conv8(x)
        x = self.dconv8(x, offset=offset)
        x = torch.relu(self.bn8(x))

        return x
'''

'''
class F2F(torch.nn.Module):
    def __init__(self):
        super(F2F, self).__init__()

        self.dconv1 = DeformableConv2d(in_channels=128*4, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dconv2 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv3 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv4 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv5 = DeformableConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.relu(self.dconv1(x))
        x = torch.relu(self.dconv2(x))
        x = torch.relu(self.dconv3(x))
        x = torch.relu(self.dconv4(x))
        x = torch.relu(self.dconv5(x))
        return x
'''
'''
from deform_conv_v2 import DeformConv2d

class F2F(torch.nn.Module):
    def __init__(self):
        super(F2F, self).__init__()

        self.dconv1 = DeformConv2d(128*4, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dconv2 = DeformConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv3 = DeformConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv4 = DeformConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv5 = DeformConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.relu(self.dconv1(x))
        x = torch.relu(self.dconv2(x))
        x = torch.relu(self.dconv3(x))
        x = torch.relu(self.dconv4(x))
        x = torch.relu(self.dconv5(x))
        return x
'''
