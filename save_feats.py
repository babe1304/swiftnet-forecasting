import torchvision.transforms as T
from data.transform import *
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os
from data.cityscapes import *
from models.semseg import SemsegModel
from models.resnet.resnet_pyramid_forecast import *
from PIL import Image
from data.transform import *
from os.path import exists

color_info = Cityscapes.color_info
to_color = ColorizeLabels(color_info)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

moving_IDs = range(11, 18 + 1)
timestep = 9

forecast_after_up_block = 2
pyramid_levels = 2

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
model = SemsegModel(backbone, 19, k=1, bias=True)

model.load_state_dict(
    torch.load('/path/to/pretrained/model'),
    strict=False)

save_dir = '/path/to/save/dir'

subset = 'val'
batch_size = 1
ostride = 4
# target_size_feats = (2048 // ostride, 1024 // ostride)
target_size = (1024, 512)
target_size_feats = (1024 // ostride, 512 // ostride)


class OrsicToTensor:
    def __call__(self, pil_img):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.array(pil_img, dtype=np.float32), (2, 0, 1))))
        # return torchvision.transforms.functional.pil_to_tensor(pil_img)


trans_val = T.Compose(
    [T.Resize((512, 1024), interpolation=T.InterpolationMode.BICUBIC),
     OrsicToTensor(),
     ])

# trans_val = T.Compose(
#     [Open(),
#      Resize(target_size),
#      SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
#      Tensor(),
#      ]
# )

# transforms = T.Compose(
#         [T.ToTensor()])

dataset_train = CityscapesSequence(
    '/path/to/images/' + subset,
    '/path/to/ground/truths/' + subset,
    delta=0, subset=subset, sem_labels=False, transforms=trans_val, extended=False)

loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=2,
                    persistent_workers=True, prefetch_factor=2)

if __name__ == '__main__':

    model.to(device)
    model.eval()

    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=len(loader)):

            for t in batch['times']:
                t = t[0]

                # if exists(save_dir + '/' + subset + '/' + batch[t + '_name'][0] + '.npy'):
                #     # print('Duplicate ' + t)
                #     continue

                # Image.fromarray(np.transpose(batch[t][0].numpy(), (1,2,0)), 'RGB').show()
                # exit()

                img = batch[t]
                img_size = batch[t].shape[-2:]
                # print(img_size)

                mock_batch = dict()
                mock_batch['image'] = img
                # mock_batch['target_size'] = img_size
                # mock_batch['target_size_feats'] = target_size_feats

                features, additional = model.forward_encoder(mock_batch, img_size)
                features_cp = torch.clone(features)
                features = features.detach().cpu().numpy()

                for i in range(batch_size):
                    name = batch[t + '_name'][i]
                    feat_map = features[i]

                    # print(feat_map.shape)

                    if os.path.exists(save_dir + '/' + subset + '/' + batch[t + '_name'][0] + '.npy'):
                        existing = np.load(save_dir + '/' + subset + '/' + name + '.npy')
                        # print(np.allclose(feat_map, existing))
                        # print(feat_map.shape, existing.shape)

                    else:
                        # print('New feats')
                        np.save(save_dir + '/' + subset + '/' + name, feat_map)

            # batch['image'] = batch['image'].squeeze()
            # batch['original_labels'] = batch['original_labels'].squeeze().numpy().astype(np.uint32)
            # img_size = batch['original_labels'].shape[-2:]
            # # print(img_size)
            # # print(batch['name'])
            # # print(batch['image'].shape)
            #
            # features, additional = model.forward_encoder(batch, img_size)
            #
            # # print(batch['name'])
            #
            # for i in range(len(batch['name'])):
            #     name = batch['name'][i][0]
            #     subset = batch['subset'][i][0]
            #     feat_map = features[i].cpu().numpy().squeeze()
            #     # print(feat_map.shape)
            #
            #     if os.path.exists(save_dir + '/' + subset + '/' + name + '.npy'):
            #         print(np.allclose(feat_map, np.load(save_dir + '/' + subset + '/' + name + '.npy')))
            #     else:
            #         # np.save(save_dir + '/' + subset + '/' + name, feat_map)
            #         pass

            # print(np.min(feat_map), np.max(feat_map), np.mean(feat_map), np.std(feat_map))
            # print(name, feat_map.shape)

            # logits = model.forward_decoder_no_skip(features, img_size)
            # pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

            # rgb_preds = conf.to_color(pred).squeeze()
            # print(rgb_preds.shape)
            # Image.fromarray(rgb_preds, 'RGB').show()
            # input('Press Enter to continue.')

    model.train()
