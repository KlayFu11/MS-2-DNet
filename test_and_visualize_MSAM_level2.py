import argparse
import os

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import auc
from tqdm import tqdm

from deepmist.datasets import build_dataset
from torch.utils.data import DataLoader
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml, make_dir
from deepmist.utils.feature_map_visualize import draw_single_feature_map


def mist_collate_fn(batch):
    """自定义collate函数，处理MISTDataset返回的额外字段"""
    # batch_size=1时，batch是包含1个样本tuple的列表，需要特殊处理
    if len(batch) == 1 and isinstance(batch[0], tuple):
        return batch[0]

    # 正常情况（batch_size > 1）
    batch_frames = [item[0] for item in batch]
    batch_masks = [item[1] for item in batch]
    batch_img_h = [item[2] for item in batch]
    batch_img_w = [item[3] for item in batch]
    batch_names = [item[4] for item in batch]
    batch_frame_names = [item[5] for item in batch]

    frames = torch.stack(batch_frames, dim=0)
    masks = torch.stack(batch_masks, dim=0)
    img_h = torch.tensor(batch_img_h)
    img_w = torch.tensor(batch_img_w)

    return frames, masks, img_h, img_w, batch_names, batch_frame_names


class Trainer(object):
    def __init__(self, args):
        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.val_dataset = build_dataset(self.cfg['dataset'], mode='val')[0]  # 取第一个val_dataset
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                      num_workers=0, pin_memory=True,
                                      collate_fn=mist_collate_fn)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

        # visualize
        self.LSC_visualize_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/MSAM_CSHD_visual_level2'
        self.use_dataparallel = args.DataParallel
        make_dir(self.LSC_visualize_dir)

    def _register_hooks(self):
        """注册hook捕获:
        1. self.LSC[1] 的完整输出（5张）
        2. self.MFA 的输出 (observation[1])（1张）
        """
        self._lsc1_outputs = []   # LSC[1] 的完整输出（5张）
        self._mfa_outputs = []    # MFA 的输出 observation[1]（1张）

        # hook: 捕获 LSC[1] 的完整输出
        def lsc1_hook(module, input, output):
            self._lsc1_outputs.append(output.detach().clone())

        # hook: 捕获 MFA 的输出（observation[1]）
        def mfa_hook(module, input, output):
            # output 是一个列表，observation[1] 是第二个元素
            self._mfa_outputs.append(output[1].detach().clone())

        # 获取 LSC[1] 和 MFA 模块
        # DataParallel 包装后需要先取 .module 再访问属性
        if self.use_dataparallel:
            lsc1_block = self.model.module.LSC[1]
            mfa_module = self.model.module.MFA
        else:
            lsc1_block = self.model.LSC[1]
            mfa_module = self.model.MFA

        # 注册 hook
        self._lsc1_hook_handle = lsc1_block.register_forward_hook(lsc1_hook)
        self._mfa_hook_handle = mfa_module.register_forward_hook(mfa_hook)

    def _remove_hooks(self):
        """移除已注册的hook"""
        if hasattr(self, '_lsc1_hook_handle'):
            self._lsc1_hook_handle.remove()
        if hasattr(self, '_mfa_hook_handle'):
            self._mfa_hook_handle.remove()

    def test(self):
        print(f"[testset]\n")
        self.model.eval()

        # 注册hook
        self._register_hooks()

        try:
            with torch.no_grad():
                for iter_idx, data in enumerate(tqdm(self.val_loader)):
                    frames, mask, _, _, _, frame_names = data
                    frames, mask = frames.to(self.device), mask.to(self.device)
                    # Add batch dimension: [C, T, H, W] -> [B, C, T, H, W]
                    frames = frames.unsqueeze(0)
                    mask = mask.unsqueeze(0)

                    # 清空hook捕获的数据
                    self._lsc1_outputs.clear()
                    self._mfa_outputs.clear()

                    # forward
                    result = run_model(self.model, self.model_name, False, False, frames)
                    if isinstance(result, tuple):
                        preds, *rest = result
                        feature_maps = rest[0] if rest else None
                    else:
                        preds = result
                        feature_maps = None
                    pred = preds[0]

                    # update metrics
                    self.mIoU_metric.update(pred, mask)
                    self.nIoU_metric.update(pred, mask)
                    self.PdFa_metric.update(pred, mask)
                    self.ROC_metric.update(pred, mask)

                    # LSC特征可视化
                    # frame_names: 列表，因为batch_size=1，collate返回的就是列表
                    # 例如 ['3/0000.png', '3/0001.png', '3/0002.png', '3/0003.png', '3/0004.png']
                    # 参考帧是最后一帧 frame_names[-1] = '3/0004.png'
                    ref_frame_name = frame_names[-1]
                    seq_id = int(ref_frame_name.split('/')[0])
                    ref_frame_id = ref_frame_name.split('/')[1].split('.')[0]  # '0024'

                    # 文件夹名：序列号3位 + 参考帧号，如 '0030024'
                    folder_name = f'{seq_id:03d}{ref_frame_id}'

                    # 5帧的帧号列表
                    frame_ids = [fn.split('/')[1].split('.')[0] for fn in frame_names]

                    # LSC[0]输出命名：当前帧号 + 参考帧号
                    # 如 ['00200024', '00210024', '00220024', '00230024', '00240024']
                    lsc_feat_names = [f'{fid}{ref_frame_id}' for fid in frame_ids]

                    # observation[0] 命名：序列号 + 参考帧号，如 '0030024'
                    obs_feat_name = f'{seq_id:03d}{ref_frame_id}'

                    # 创建文件夹
                    seq_dir = os.path.join(self.LSC_visualize_dir, folder_name)
                    make_dir(seq_dir)

                    # 保存5张 LSC[1] 的完整输出
                    for i, feat in enumerate(self._lsc1_outputs):
                        # 从frame_names中获取实际帧号
                        frame_name = frame_names[i]
                        img_path = os.path.join(self.cfg['dataset']['root'], 'image', frame_name)
                        save_path = os.path.join(seq_dir, lsc_feat_names[i] + '.png')
                        draw_single_feature_map(feat, img_path, save_path)

                    # 保存1张 observation[1]（MFA 输出）
                    if len(self._mfa_outputs) > 0:
                        obs_feat = self._mfa_outputs[0]
                        # 使用参考帧的图像作为背景
                        ref_img_path = os.path.join(self.cfg['dataset']['root'], 'image', ref_frame_name)
                        save_path = os.path.join(seq_dir, obs_feat_name + '.png')
                        draw_single_feature_map(obs_feat, ref_img_path, save_path)

        finally:
            # 确保退出时移除hook
            self._remove_hooks()

        # get metrics
        # print('FPS=%.3f' % ((iter_idx + 1) / (time.time() - start_time)))
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        precision_05 = precision[(self.bins + 1) // 2 + 1]
        recall_05 = recall[(self.bins + 1) // 2 + 1]
        f_score_05 = f_score[(self.bins + 1) // 2 + 1]

        # log the test info
        message = f'\n[mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}][PD:{PD:.5f}][FA:{FA:.5e}][AUC:{AUC:.5f}]' \
                  f'[precision:{precision_05:.5f}][recall:{recall_05:.5f}][f_score:{f_score_05:.5f}]'
        print(message)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = Trainer(args)
    trainer.test()


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str, default='/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/configs/multiframe/LS_STDNet/test_LS_STDNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='1', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)
