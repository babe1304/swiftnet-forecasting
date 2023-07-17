from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from os.path import exists
import torch
from PIL import Image
from data.cityscapes import Cityscapes, CityscapesSequence
import yaml
from omegaconf import OmegaConf
from external_packages.taming.models.vqgan import VQModel


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


def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


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
        image /= 255.
        return 2. * image - 1.


class OrsicToTensor:
    def __call__(self, pil_img):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.array(pil_img, dtype=np.float32), (2, 0, 1))))
        # return torchvision.transforms.functional.pil_to_tensor(pil_img)


if __name__ == '__main__':

    DEVICE = torch.device("cuda")

    save_dir = '/path/to/save/dir'

    config16384 = load_config("external_packages/taming/logs/vqgan_imagenet_f16_16384/configs/model.yaml",
                              display=False)
    model = load_vqgan(config16384,
                       ckpt_path="external_packages/taming/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(
        DEVICE)
    model.eval()

    subset = 'train'
    batch_size = 1
    size = (512, 1024)

    transforms = T.Compose(
        [T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
         OrsicToTensor(),
         # T.ToTensor(),
         Scale()
         ])

    dataset_train = CityscapesSequence(
        '/path/to/images/' + subset,
        '/path/to/ground/truths/' + subset,
        delta=0, subset=subset, sem_labels=False, transforms=transforms, extended=True)

    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=4,
                              persistent_workers=True, prefetch_factor=2)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=len(loader)):

            for t in batch['times']:
                t = t[0]

                if exists(save_dir + '/' + subset + '/' + batch[t + '_name'][0] + '.npy'):
                    # print('Duplicate ' + t)
                    continue

                img = batch[t].to(DEVICE)
                # z, _, [_, _, indices] = model.encode(img)
                # z = model.encode_no_quant(img)
                z = model.encode_no_quant_v2(img)
                z = z.detach().cpu().numpy()

                for i in range(batch_size):
                    name = batch[t + '_name'][i]
                    feat_map = z[i]

                    # print(feat_map.shape)
                    full_name = save_dir + '/' + subset + '/' + name + '.npy'
                    np.save(full_name, feat_map)

                    # if exists(full_name):
                    #     existing = np.load(full_name)
                    #     print(np.allclose(feat_map, existing))
                    #     # print(feat_map.shape, existing.shape)

                # imgrec = model.decode(z)
                # print(imgrec.shape)
                # custom_to_pil(imgrec.squeeze()).show()
                # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")



