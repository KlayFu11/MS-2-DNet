import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os

# 定义ResBlock类
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# 定义make_layer函数
def make_layer(block, in_channels, out_channels, num_blocks=1):
    layers = []
    # 第一个block可能需要下采样或改变通道数
    layers.append(block(in_channels, out_channels, stride=1))
    # 后续block保持相同的通道数和空间大小
    for _ in range(num_blocks - 1):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)

# 定义ResNet类
class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, nb_filter):
        super(ResNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder_1 = make_layer(block, in_channels, nb_filter[0])
        self.encoder_2 = make_layer(block, nb_filter[0], nb_filter[1], num_blocks=num_blocks[0])
        self.encoder_3 = make_layer(block, nb_filter[1], nb_filter[2], num_blocks=num_blocks[1])
        self.encoder_4 = make_layer(block, nb_filter[2], nb_filter[3], num_blocks=num_blocks[2])

    def forward(self, x):
        feat_1 = self.encoder_1(x)  # [1, 8, 384, 384]
        feat_2 = self.encoder_2(self.pool(feat_1))  # [1, 16, 192, 192]
        feat_3 = self.encoder_3(self.pool(feat_2))  # [1, 32, 96, 96]
        feat_4 = self.encoder_4(self.pool(feat_3))  # [1, 64, 48, 48]
        return feat_1, feat_2, feat_3, feat_4

# 创建模型实例
in_channels = 3
nb_filter = [8, 16, 32, 64]
num_blocks = [2, 2, 2]
model = ResNet(in_channels, ResBlock, num_blocks, nb_filter)

# 读取并预处理实际图片
image_path = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912/image/98/0000.png'

# 定义图片预处理转换
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),  # 调整大小为384x384
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
])

# 读取图片
image = Image.open(image_path).convert('RGB')  # 确保是RGB格式

# 预处理图片
input_tensor = preprocess(image).unsqueeze(0)  # 添加batch维度，形状变为(1, 3, 384, 384)
print(f"输入图片张量尺寸: {input_tensor.shape}")

# 前向传播获取特征图
model.eval()
with torch.no_grad():
    feat_1, feat_2, feat_3, feat_4 = model(input_tensor)

# 显式显示所有特征图张量的尺寸
print("\n所有特征图张量尺寸:")
print(f"feat_1 (encoder_1): {feat_1.shape}")
print(f"feat_2 (encoder_2): {feat_2.shape}")
print(f"feat_3 (encoder_3): {feat_3.shape}")
print(f"feat_4 (encoder_4): {feat_4.shape}")

# 创建保存目录
save_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/visual_result'
os.makedirs(save_dir, exist_ok=True)

# 保存原始特征图张量函数
def save_feature_map_tensor(feat_map, save_path):
    """
    保存原始特征图张量
    feat_map: 特征图，形状为 (B, C, H, W)
    save_path: 保存路径，支持.pt或.npy格式
    """
    # 显示要保存的张量尺寸
    print(f"\n正在保存张量尺寸: {feat_map.shape}")
    
    if save_path.endswith('.pt'):
        torch.save(feat_map, save_path)
        print(f"已保存特征图张量为.pt文件: {save_path}")
        # 显示文件大小
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"  文件大小: {file_size:.2f} MB")
    elif save_path.endswith('.npy'):
        np.save(save_path, feat_map.cpu().numpy())
        print(f"已保存特征图张量为.npy文件: {save_path}")
        # 显示文件大小
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"  文件大小: {file_size:.2f} MB")
    else:
        raise ValueError(f"不支持的保存格式: {save_path}，仅支持.pt或.npy格式")

# 简化的可视化函数 - 无标注纯图像
def visualize_feature_map_simple(feat_map, save_path=None):
    """
    可视化特征图的所有通道，仅显示纯图像，无任何标注
    feat_map: 特征图，形状为 (B, C, H, W)
    save_path: 保存路径，None表示不保存
    """
    # 选择第一个样本
    feat = feat_map[0]
    
    # 对于多通道特征图，使用mean方法组合所有通道
    combined = feat.mean(dim=0).cpu().numpy()
    print(f"\n可视化图像信息:")
    print(f"  原始特征图通道数: {feat.shape[0]}")
    print(f"  合并后图像尺寸: {combined.shape}")
    
    plt.figure(figsize=(combined.shape[1]/100, combined.shape[0]/100), dpi=100)
    plt.imshow(combined, cmap='viridis')
    plt.axis('off')  # 关闭坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除所有边距
    
    # 保存图片（如果需要）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"已保存纯图像: {save_path}")
        # 显示PNG文件大小
        if os.path.exists(save_path):
            png_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"  PNG文件大小: {png_size:.4f} MB")
    
    plt.close()  # 关闭图片以释放内存

# 处理最后的特征图（encoder_4的64通道特征图）
final_feat = feat_4
print(f"\n最终处理的特征图 (encoder_4): {final_feat.shape}")

# 保存完整通道的原始特征图张量
save_feature_map_tensor(final_feat, os.path.join(save_dir, '81_encoder_4_features.pt'))  # 保存为PyTorch张量
save_feature_map_tensor(final_feat, os.path.join(save_dir, '81_encoder_4_features.npy'))  # 保存为NumPy数组

# 生成无标注的纯图像
visualize_feature_map_simple(final_feat, os.path.join(save_dir, '81_encoder_4_visualization.png'))

print("\n" + "="*60)
print("PNG格式特征图大小压缩原因解释:")
print("="*60)
print("1. 维度压缩: 原始特征图是64通道的4D张量 (1,64,48,48)，")
print("   保存为PNG时需要将多通道合并为单通道或3通道图像，丢失了维度信息。")
print("2. 数据类型转换: 原始张量通常是float32类型(4字节/像素/通道)，")
print("   而PNG使用8位或16位整数(1-2字节/像素/通道)，数据精度降低。")
print("3. 图像压缩算法: PNG使用无损压缩算法(Deflate)对图像数据进行压缩，")
print("   可以显著减小文件大小而不损失视觉质量。")
print("4. 信息损失: 多通道合并过程(如mean操作)会丢失通道间的差异信息，")
print("   进一步减小了数据量。")
print("\n如果需要完整保存特征图信息，请使用.pt或.npy格式！")
print("="*60)

print("处理完成！已保存：")
print("1. 完整通道的原始特征图张量 (.pt 和 .npy 格式)")
print("2. 无标注的纯可视化图像")