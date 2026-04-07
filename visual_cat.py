import torch
import matplotlib.pyplot as plt
import os

def cat_tensor_files(tensor_paths, cat_dimension=1, save_path=None):
    """
    读取.pt文件并在指定维度拼接
    tensor_paths: .pt文件路径列表
    cat_dimension: 拼接维度，默认通道维度(dim=1)
    save_path: 保存路径，None表示不保存
    """
    # 检查文件是否存在
    for path in tensor_paths:
        if not os.path.exists(path):
            print(f"错误: 文件 {path} 不存在")
            return None
    
    # 读取所有张量
    tensors = []
    for path in tensor_paths:
        tensor = torch.load(path)
        tensors.append(tensor)
        print(f"读取张量 {path}，形状: {tensor.shape}")
    
    # 检查所有张量在非拼接维度上的形状是否一致
    base_shape = list(tensors[0].shape)
    for i, tensor in enumerate(tensors):
        tensor_shape = list(tensor.shape)
        for dim in range(len(tensor_shape)):
            if dim != cat_dimension and tensor_shape[dim] != base_shape[dim]:
                print(f"错误: 张量 {i+1} 在维度 {dim} 上的形状与第一个张量不一致")
                return None
    
    # 在指定维度拼接
    concatenated_tensor = torch.cat(tensors, dim=cat_dimension)
    print(f"拼接后的张量形状: {concatenated_tensor.shape}")
    
    # 可视化拼接后的特征图
    visualize_feature_map(concatenated_tensor, save_path=save_path)
    
    return concatenated_tensor

def visualize_feature_map(feat_map, save_path=None):
    """
    可视化特征图
    feat_map: 特征图，形状为 (B, C, H, W)
    save_path: 保存路径，None表示不保存
    """
    # 选择第一个样本
    feat = feat_map[0]
    
    print(f"可视化特征图信息:")
    print(f"  通道数: {feat.shape[0]}")
    print(f"  空间尺寸: {feat.shape[1:]}")
    
    # 使用不同方法组合所有通道进行可视化
    combination_methods = ['mean', 'sum', 'max']
    
    for method in combination_methods:
        if method == 'mean':
            combined = feat.mean(dim=0).cpu().numpy()
        elif method == 'sum':
            combined = feat.sum(dim=0).cpu().numpy()
        elif method == 'max':
            combined = feat.max(dim=0)[0].cpu().numpy()
        
        plt.figure(figsize=(feat.shape[2]/100, feat.shape[1]/100), dpi=100)
        plt.imshow(combined, cmap='viridis')
        plt.axis('off')  # 关闭坐标轴
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除所有边距
        
        # 保存图片（如果需要）
        if save_path:
            # 为不同组合方法创建不同的文件名
            base_name, ext = os.path.splitext(save_path)
            method_save_path = f"{base_name}_{method}{ext}"
            plt.savefig(method_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"已保存 {method} 组合可视化: {method_save_path}")
        
        plt.close()  # 关闭图片以释放内存

def cat_images_as_tensors(image_paths, cat_dimension=0, save_path=None):
    # 定义图片预处理转换（转为张量）
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    ])
    
    # 读取并转换所有图片为张量
    tensors = []
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images.append(img)
        tensor = preprocess(img)  # 形状: (C, H, W)
        tensors.append(tensor)
    
    # 检查图片大小
    for i, img in enumerate(images):
        print(f"图片 {i+1} 大小: {img.size}")
    
    # 确保所有图片大小相同
    for i in range(1, len(images)):
        if images[i].size != images[0].size:
            print(f"警告: 图片 {i+1} 大小与第一张不同 ({images[i].size} vs {images[0].size})")
            print("将调整图片大小以匹配第一张图片")
            # 调整大小
            resized_img = images[i].resize(images[0].size)
            images[i] = resized_img
            tensors[i] = preprocess(resized_img)
    
    # 添加batch维度
    tensors = [tensor.unsqueeze(0) for tensor in tensors]  # 形状变为: (1, C, H, W)
    
    # 在指定维度拼接
    concatenated_tensor = torch.cat(tensors, dim=cat_dimension)
    
    print(f"拼接后的张量形状: {concatenated_tensor.shape}")
    
    # 可视化拼接后的结果 - 显示所有图片
    num_images = concatenated_tensor.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(15, 10))
    
    for i in range(num_images):
        img_tensor = concatenated_tensor[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1}")
    
    plt.tight_layout()
    
    # 保存结果
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存拼接结果至: {save_path}")
    
    plt.show()
    
    return concatenated_tensor

if __name__ == "__main__":
    # 张量文件路径
    tensor_paths = [
        '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/visual_result/54_encoder_4_features.pt',
        '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/visual_result/16_encoder_4_features.pt'
    ]
    
    # 在通道维度(dim=1)拼接
    save_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/visual_result/cat_channels.png'
    concatenated_tensor = cat_tensor_files(tensor_paths, cat_dimension=1, save_path=save_path)
    
    # 显式输出张量信息
    if concatenated_tensor is not None:
        print("\n张量详细信息:")
        print(f"形状: {concatenated_tensor.shape}")
        print(f"数据类型: {concatenated_tensor.dtype}")
        print(f"最小值: {concatenated_tensor.min().item()}")
        print(f"最大值: {concatenated_tensor.max().item()}")
        print(f"平均值: {concatenated_tensor.mean().item()}")
        print(f"标准差: {concatenated_tensor.std().item()}")