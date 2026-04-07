import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from deepmist.datasets import build_dataset, DataLoaderX
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml, make_dir


def set_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Tester(object):
    def __init__(self, args):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.val_dataset, _ = build_dataset(self.cfg['dataset'], mode='val')
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['test']['num_workers'], pin_memory=True)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

        # visualize
        self.feat_vis_dir = os.path.join(self.cfg['test']['exp_root'], 'feat_vis_dir')
        make_dir(self.feat_vis_dir)
        self.z_4_max_dir = os.path.join(self.feat_vis_dir, 'z_4_max')
        self.z_4_mean_dir = os.path.join(self.feat_vis_dir, 'z_4_mean')
        self.z_3_max_dir = os.path.join(self.feat_vis_dir, 'z_3_max')
        self.z_3_mean_dir = os.path.join(self.feat_vis_dir, 'z_3_mean')
        self.z_2_max_dir = os.path.join(self.feat_vis_dir, 'z_2_max')
        self.z_2_mean_dir = os.path.join(self.feat_vis_dir, 'z_2_mean')
        self.z_1_max_dir = os.path.join(self.feat_vis_dir, 'z_1_max')
        self.z_1_mean_dir = os.path.join(self.feat_vis_dir, 'z_1_mean')
        make_dir(self.z_4_max_dir)
        make_dir(self.z_4_mean_dir)
        make_dir(self.z_3_max_dir)
        make_dir(self.z_3_mean_dir)
        make_dir(self.z_2_max_dir)
        make_dir(self.z_2_mean_dir)
        make_dir(self.z_1_max_dir)
        make_dir(self.z_1_mean_dir)

    def test(self):
        print(f"[testset]\n")
        self.model.eval()

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(self.val_loader)):
                frames, mask, h, w, name = data

                if name[0] != '117/0117.png':
                    continue

                frames, mask = frames.to(self.device), mask.to(self.device)
                preds, z_4, z_3, z_2, z_1 = run_model(self.model, self.model_name, False, False, frames)

                size = (384, 384)

                # z_4
                # max
                z_4_max = z_4.cpu().detach().numpy()
                z_4_max = np.max(z_4_max, axis=1)
                z_4_max = np.squeeze(z_4_max)
                z_4_max = cv2.resize(z_4_max, size)

                #############
                max_value = z_4_max.max()
                min_value = z_4_max.min()
                print(f"{name[0]}的特征图z4的最大值: {max_value}")
                print(f"{name[0]}的特征图z4的最小值: {min_value}")
                # break
                #############

                # z_4_max = (((z_4_max - z_4_max.min()) / (z_4_max.max() - z_4_max.min()) + 1e-8) * 255).astype(np.uint8)
                # z_4_max = cv2.applyColorMap(z_4_max, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_4_max_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_4_max_dir, name[0]), z_4_max)
                # # mean
                # z_4_mean = z_4.cpu().detach().numpy()
                # z_4_mean = np.mean(z_4_mean, axis=1)
                # z_4_mean = np.squeeze(z_4_mean)
                # z_4_mean = cv2.resize(z_4_mean, size)
                # z_4_mean = (((z_4_mean - z_4_mean.min()) / (z_4_mean.max() - z_4_mean.min()) + 1e-8) * 255).astype(
                #     np.uint8)
                # z_4_mean = cv2.applyColorMap(z_4_mean, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_4_mean_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_4_mean_dir, name[0]), z_4_mean)

                # z_3
                # max
                z_3_max = z_3.cpu().detach().numpy()
                z_3_max = np.max(z_3_max, axis=1)
                z_3_max = np.squeeze(z_3_max)
                z_3_max = cv2.resize(z_3_max, size)

                #############
                max_value = z_3_max.max()
                min_value = z_3_max.min()
                print(f"{name[0]}的特征图z3的最大值: {max_value}")
                print(f"{name[0]}的特征图z3的最小值: {min_value}")
                # break
                #############

                # z_3_max = (((z_3_max - z_3_max.min()) / (z_3_max.max() - z_3_max.min()) + 1e-8) * 255).astype(np.uint8)
                # z_3_max = cv2.applyColorMap(z_3_max, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_3_max_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_3_max_dir, name[0]), z_3_max)
                # # mean
                # z_3_mean = z_3.cpu().detach().numpy()
                # z_3_mean = np.mean(z_3_mean, axis=1)
                # z_3_mean = np.squeeze(z_3_mean)
                # z_3_mean = cv2.resize(z_3_mean, size)
                # z_3_mean = (((z_3_mean - z_3_mean.min()) / (z_3_mean.max() - z_3_mean.min()) + 1e-8) * 255).astype(
                #     np.uint8)
                # z_3_mean = cv2.applyColorMap(z_3_mean, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_3_mean_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_3_mean_dir, name[0]), z_3_mean)

                # z_2
                # max
                z_2_max = z_2.cpu().detach().numpy()
                z_2_max = np.max(z_2_max, axis=1)
                z_2_max = np.squeeze(z_2_max)
                z_2_max = cv2.resize(z_2_max, size)

                #############
                max_value = z_2_max.max()
                min_value = z_2_max.min()
                print(f"{name[0]}的特征图z2的最大值: {max_value}")
                print(f"{name[0]}的特征图z2的最小值: {min_value}")
                # break
                #############

                # z_2_max = (((z_2_max - z_2_max.min()) / (z_2_max.max() - z_2_max.min()) + 1e-8) * 255).astype(np.uint8)
                # z_2_max = cv2.applyColorMap(z_2_max, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_2_max_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_2_max_dir, name[0]), z_2_max)
                # # mean
                # z_2_mean = z_2.cpu().detach().numpy()
                # z_2_mean = np.mean(z_2_mean, axis=1)
                # z_2_mean = np.squeeze(z_2_mean)
                # z_2_mean = cv2.resize(z_2_mean, size)
                # z_2_mean = (((z_2_mean - z_2_mean.min()) / (z_2_mean.max() - z_2_mean.min()) + 1e-8) * 255).astype(
                #     np.uint8)
                # z_2_mean = cv2.applyColorMap(z_2_mean, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_2_mean_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_2_mean_dir, name[0]), z_2_mean)

                # z_1
                # max
                z_1_max = z_1.cpu().detach().numpy()
                z_1_max = np.max(z_1_max, axis=1)
                z_1_max = np.squeeze(z_1_max)
                z_1_max = cv2.resize(z_1_max, size)

                #############
                max_value = z_1_max.max()
                min_value = z_1_max.min()
                print(f"{name[0]}的特征图z1的最大值: {max_value}")
                print(f"{name[0]}的特征图z1的最小值: {min_value}")
                break
                #############

                # z_1_max = (((z_1_max - z_1_max.min()) / (z_1_max.max() - z_1_max.min()) + 1e-8) * 255).astype(np.uint8)
                # z_1_max = cv2.applyColorMap(z_1_max, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_1_max_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_1_max_dir, name[0]), z_1_max)
                # # mean
                # z_1_mean = z_1.cpu().detach().numpy()
                # z_1_mean = np.mean(z_1_mean, axis=1)
                # z_1_mean = np.squeeze(z_1_mean)
                # z_1_mean = cv2.resize(z_1_mean, size)
                # z_1_mean = (((z_1_mean - z_1_mean.min()) / (z_1_mean.max() - z_1_mean.min()) + 1e-8) * 255).astype(
                #     np.uint8)
                # z_1_mean = cv2.applyColorMap(z_1_mean, cv2.COLORMAP_JET)
                # save_dir = os.path.join(self.z_1_mean_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # cv2.imwrite(os.path.join(self.z_1_mean_dir, name[0]), z_1_mean)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    tester = Tester(args)
    tester.test()


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str,
                        default='./configs/multiframe/RFR/vis_feat/test_ResUNet_RFR_vis_feat_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='1', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)
