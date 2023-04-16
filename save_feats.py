import argparse
from pathlib import Path
import importlib.util
from evaluation import evaluate_semseg
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from data.cityscapes import Cityscapes, CityscapesSequence


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')

if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    save_dir = '/run/media/jakov/2TB KC3000/Users/bubas/Data/Cityscapes/features/3_levels/new/3_skips_64x128'

    model = conf.model.cuda()
    model.eval()

    with torch.no_grad():
        for loader, name in conf.eval_loaders:
            for step, batch in tqdm(enumerate(loader), total=len(loader)):

                batch['image'] = batch['image'].squeeze()
                batch['original_labels'] = batch['original_labels'].squeeze().numpy().astype(np.uint32)
                img_size = batch['original_labels'].shape[-2:]
                # print(img_size)
                # print(batch['name'])
                # print(batch['image'].shape)

                features, additional = model.forward_encoder(batch, img_size)

                # print(batch['name'])

                for i in range(len(batch['name'])):
                    name = batch['name'][i][0]
                    subset = batch['subset'][i][0]
                    feat_map = features[i].cpu().numpy().squeeze()
                    # print(feat_map.shape)
                    np.save(save_dir + '/' + subset + '/' + name, feat_map)

                    # print(np.min(feat_map), np.max(feat_map), np.mean(feat_map), np.std(feat_map))
                    # print(name, feat_map.shape)

                # logits = model.forward_decoder_no_skip(features, img_size)
                # pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

                # rgb_preds = conf.to_color(pred).squeeze()
                # print(rgb_preds.shape)
                # Image.fromarray(rgb_preds[0], 'RGB').show()
                # input('Press Enter to continue.')

    model.train()
