import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
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
        self.visualize_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/CSHD_SIZE_visual_allSSRD1'
        self.use_dataparallel = args.DataParallel
        make_dir(self.visualize_dir)

    def _register_hooks(self):
        """注册hook捕获 z_1, d_2, d_3, pred_z_1"""
        self._z1_outputs = []
        self._d2_outputs = []
        self._d3_outputs = []
        self._pred_outputs = []

        def z1_hook(module, input, output):
            self._z1_outputs.append(output.detach().clone())

        def d2_hook(module, input, output):
            self._d2_outputs.append(output.detach().clone())

        def d3_hook(module, input, output):
            self._d3_outputs.append(output.detach().clone())

        def pred_hook(module, input, output):
            self._pred_outputs.append(output.detach().clone())

        # 获取 decoder 模块
        if self.use_dataparallel:
            decoder_module = self.model.module.decoder
        else:
            decoder_module = self.model.decoder

        # 注册 hook
        self._z1_hook_handle = decoder_module.updecoder_1.register_forward_hook(z1_hook)
        self._d2_hook_handle = decoder_module.downdecoder_1.register_forward_hook(d2_hook)
        self._d3_hook_handle = decoder_module.downdecoder_2.register_forward_hook(d3_hook)
        self._pred_hook_handle = decoder_module.head_1.register_forward_hook(pred_hook)

    def _remove_hooks(self):
        """移除已注册的hook"""
        for attr in ['_z1_hook_handle', '_d2_hook_handle', '_d3_hook_handle', '_pred_hook_handle']:
            if hasattr(self, attr):
                getattr(self, attr).remove()

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
                    self._z1_outputs.clear()
                    self._d2_outputs.clear()
                    self._d3_outputs.clear()
                    self._pred_outputs.clear()

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

                    # 解析路径，获取序列号和参考帧号
                    # 例如 frame_names[-1] = '12/0006.png'
                    ref_frame_name = frame_names[-1]
                    seq_id = int(ref_frame_name.split('/')[0])
                    ref_frame_id = ref_frame_name.split('/')[1].split('.')[0]  # '0006'

                    # 创建文件夹：序列号3位 + 参考帧号4位，如 '0120006'
                    folder_name = f'{seq_id:03d}{ref_frame_id}'
                    seq_dir = os.path.join(self.visualize_dir, folder_name)
                    make_dir(seq_dir)

                    # 保存 z_1
                    if len(self._z1_outputs) > 0:
                        z1_feat = self._z1_outputs[0]
                        ref_img_path = os.path.join(self.cfg['dataset']['root'], 'image', ref_frame_name)
                        save_path = os.path.join(seq_dir, f'{ref_frame_id}_level1.png')
                        draw_single_feature_map(z1_feat, ref_img_path, save_path)

                    # 保存 d_2
                    if len(self._d2_outputs) > 0:
                        d2_feat = self._d2_outputs[0]
                        ref_img_path = os.path.join(self.cfg['dataset']['root'], 'image', ref_frame_name)
                        save_path = os.path.join(seq_dir, f'{ref_frame_id}_level2.png')
                        draw_single_feature_map(d2_feat, ref_img_path, save_path)

                    # 保存 d_3
                    if len(self._d3_outputs) > 0:
                        d3_feat = self._d3_outputs[0]
                        ref_img_path = os.path.join(self.cfg['dataset']['root'], 'image', ref_frame_name)
                        save_path = os.path.join(seq_dir, f'{ref_frame_id}_level3.png')
                        draw_single_feature_map(d3_feat, ref_img_path, save_path)

                    # 保存 pred_z_1 (mask)，经过sigmoid处理
                    if len(self._pred_outputs) > 0:
                        pred_feat = self._pred_outputs[0]
                        pred_sigmoid = torch.sigmoid(pred_feat)
                        final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                        mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                        save_path = os.path.join(seq_dir, 'mask.png')
                        mask_pred.save(save_path)

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


# left_y_data = [
#     94.45,  # 3~5 对应的PD值
#     83.89,  # 5~7 对应的PD值
#     96.25,  # 7~9 对应的PD值
#     92.07,  # 9~11 对应的PD值
#     74.07   # 11~13 对应的PD值
# ]

# # 右Y轴数据 (柱状图) - Fa (10^-6)
# # 请在这里填入5个数值，范围建议在0-6之间
# right_y_data = [
#     1.92,  # 3~5 对应的Fa值
#     8.28,  # 5~7 对应的Fa值
#     1.57,  # 7~9 对应的Fa值
#     5.86,  # 9~11 对应的Fa值
#     13.40   # 11~13 对应的Fa值
# ]

# left_y_data = [
#     95.01,  # 3~5 对应的PD值
#     87.65,  # 5~7 对应的PD值
#     97.38,  # 7~9 对应的PD值
#     93.21,  # 9~11 对应的PD值
#     63.7   # 11~13 对应的PD值
# ]

# # 右Y轴数据 (柱状图) - Fa (10^-6)
# # 请在这里填入5个数值，范围建议在0-6之间
# right_y_data = [
#     1.13,  # 3~5 对应的Fa值
#     4.15,  # 5~7 对应的Fa值
#     0.84,  # 7~9 对应的Fa值
#     5.26,  # 9~11 对应的Fa值
#     0.14   # 11~13 对应的Fa值
# ]