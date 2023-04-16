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

    save_dir = '/home/jakov/1TB_NVMe/Users/bubas/Data/Cityscapes/features_pyr_forecast'

    model = conf.model#.cuda()
    model.eval()

    with torch.no_grad():
        for loader, name in conf.eval_loaders:
            for step, batch in tqdm(enumerate(loader), total=len(loader)):

                past_feats, target_feats, sem_seg_gt = batch

                # batch['image'] = batch['image']#.squeeze()
                # batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
                # img_size = batch['original_labels'].shape[1:3]
                #print(img_size)
                #print(batch['image'].shape)
                #print(batch['name'])

                # features, additional = model.forward_encoder(target_feats)

                #print(batch['name'])
                #print(features.shape)

                logits = model.forward_decoder_no_skip(target_feats, (1024, 2048))
                pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

                rgb_preds = conf.to_color(pred).squeeze()
                Image.fromarray(rgb_preds, 'RGB').show()
                input('Press Enter to continue.')

    model.train()
