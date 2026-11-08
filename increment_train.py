import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from dataset import HyperSpectralDataset
from Image_search import EnhancedHybridMatcher
from image_generate import VQDiT, LatentMeanFlow

import time
from tqdm import tqdm
from generate_train import train_vqdit_hyperspectral, fine_tune_vqdit_hyperspectral
import random
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import scipy.io as sio

def generate_patches_with_types(train_loader, image, image_pad, device, half,
                                local_radius=40, local_topk=10, superpixel_topk=3,
                                global_topk=10, min_dist=15, spectral_threshold=0.5,
                                context_threshold=0.6):
    """
    根据每个 batch 的中心点生成局部/全局 patch，并构建新的 TensorDataset。

    参数:
    - train_loader: 训练数据加载器，提供输入数据、标签和坐标。
    - image: 原始图像数据。
    - image_pad: 填充后的图像数据，用于处理边缘情况。
    - device: 设备信息，用于确定计算是在 CPU 还是 GPU 上进行。
    - half: patch 半边长，用于确定 patch 大小。
    - local_radius: 局部区域的半径。
    - local_topk: 从局部区域选择的 top-k 候选。
    - superpixel_topk: 从超像素区域选择的 top-k 候选。
    - global_topk: 从全局区域选择的 top-k 候选。
    - min_dist: 候选点之间的最小距离。
    - spectral_threshold: 谱相似性的阈值。
    - context_threshold: 上下文相似性的阈值。

    返回:
    - TensorDataset: 包含生成的 patches、扩展的标签和坐标信息的数据集。
    """
    patches = []
    labels_extended = []
    xs_list = []
    ys_list = []
    types_list = []  # 用于记录点类型：0=原始中心点，1=候选点

    # 计算总批次数
    total_batches = len(train_loader)

    # 创建进度条
    pbar = tqdm(total=total_batches, desc="🔍🔍 生成匹配点", unit="batch")

    for batch_idx, (inputs, labels, ys, xs) in enumerate(train_loader):
        y, x = ys[0].item(), xs[0].item()

        # 1. 首先添加原始中心点本身
        patches.append(inputs)  # 原始输入点
        labels_extended.append(labels)
        ys_list.append(torch.tensor([y]))
        xs_list.append(torch.tensor([x]))
        types_list.append(torch.tensor([0]))  # 0表示原始中心点

        # 创建匹配器
        matcher = EnhancedHybridMatcher(image, (y, x), device=device, normalize=True)

        results = matcher.hybrid_selection(
            local_radius=local_radius,
            local_topk=local_topk,
            superpixel_topk=superpixel_topk,
            global_topk=global_topk,
            min_dist=min_dist,
            spectral_threshold=spectral_threshold,
            context_threshold=context_threshold,
        )

        # 获取局部和全局坐标
        local_coords = results['local'].astype(int) if 'local' in results else np.array([])
        global_coords = results['global'].astype(int) if 'global' in results else np.array([])

        # 合并所有候选点
        all_coords = np.vstack([local_coords, global_coords]) if len(local_coords) > 0 or len(
            global_coords) > 0 else np.array([])

        if len(all_coords) > 0:
            # 遍历所有候选点
            for coord in all_coords:
                ny, nx = coord

                # 提取patch
                top = ny
                bottom = ny + half * 2 + 1
                left = nx
                right = nx + half * 2 + 1

                if (0 <= top < image_pad.shape[1] and
                        0 <= bottom <= image_pad.shape[1] and
                        0 <= left < image_pad.shape[2] and
                        0 <= right <= image_pad.shape[2]):
                    patch = image_pad[:, top:bottom, left:right]
                    patch_tensor = patch.float().unsqueeze(0)

                    # 添加到数据集
                    patches.append(patch_tensor)
                    labels_extended.append(labels)  # 使用相同的标签
                    ys_list.append(torch.tensor([y]))  # 保持原始中心点坐标
                    xs_list.append(torch.tensor([x]))  # 保持原始中心点坐标
                    types_list.append(torch.tensor([1]))  # 1表示候选点

        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            "当前批次": f"{batch_idx + 1}/{total_batches}",
            "生成点数": len(patches),
            "候选点数": len(all_coords)
        })

    # 关闭进度条
    pbar.close()

    # 转换为Tensor
    patches_tensor = torch.cat(patches, dim=0) if patches else torch.tensor([])
    labels_tensor = torch.cat(labels_extended, dim=0) if labels_extended else torch.tensor([])
    ys_tensor = torch.cat(ys_list, dim=0) if ys_list else torch.tensor([])
    xs_tensor = torch.cat(xs_list, dim=0) if xs_list else torch.tensor([])
    types_tensor = torch.cat(types_list, dim=0) if types_list else torch.tensor([])

    # 打印最终统计信息
    print(f"✅ 生成完成: 总批次数 = {total_batches}, 成功生成点数 = {len(patches)}")

    return TensorDataset(patches_tensor, labels_tensor, ys_tensor, xs_tensor)

def print_class_distribution(test_loader):
    """
    打印测试数据集中每个类别的样本数量

    参数:
    - test_loader: 测试数据加载器

    返回:
    - class_counts: 字典，键为类别ID，值为样本数量
    """
    class_counts = {}

    for batch_idx, (images, patches, labels, ys, xs) in enumerate(test_loader):
        # 将标签转换为numpy数组，并确保在CPU上
        labels_np = labels.cpu().numpy()  # ✅ 修正：先移动到CPU再转为numpy

        # 统计当前batch中的类别分布
        unique, counts = np.unique(labels_np, return_counts=True)

        # 更新总统计
        for cls, cnt in zip(unique, counts):
            if cls in class_counts:
                class_counts[cls] += cnt
            else:
                class_counts[cls] = cnt

    # 打印结果
    print("\n测试数据集类别分布统计:")
    print("{:<10} {:<15}".format('类别ID', '样本数量'))
    print("-" * 25)
    for cls in sorted(class_counts.keys()):
        print("{:<10} {:<15}".format(cls, class_counts[cls]))

    # 打印总数
    total_samples = sum(class_counts.values())
    print("-" * 25)
    print("{:<10} {:<15}".format('总计', total_samples))
    print("类别数量:", len(class_counts))

    return class_counts


def evaluate_classifier(model, test_loader, device, stage, round_idx=None):
    """
    评估分类器并保存分类结果（保留二维结构）
    参数:
    - model: 分类模型
    - test_loader: 测试数据加载器（包含坐标信息）
    - device: 计算设备
    - stage: 阶段名称 ('base' 或 'incremental')
    - round_idx: 增量轮次索引 (可选)
    """
    model.eval()

    # 初始化存储结构
    all_true = []
    all_pred = []
    all_coords = []  # 存储坐标信息 (y, x)

    print_class_distribution(test_loader)

    with torch.no_grad():
        # 测试集现在包含5个元素：images, patches, labels, ys, xs
        for images, patches, labels, ys, xs in test_loader:
            images, patches, labels = images.to(device), patches.to(device), labels.to(device)
            logits = model(images, patches)
            _, predicted = torch.max(logits, 1)

            # 收集真实标签、预测结果和坐标
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

            # 收集坐标信息
            for i in range(len(ys)):
                all_coords.append((ys[i].item(), xs[i].item()))

    # 计算评估指标
    oa = accuracy_score(all_true, all_pred)

    # 计算每个类别的准确率
    unique_classes = np.unique(all_true)
    class_accuracies = []
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_mask = np.array(all_true) == cls
        # 计算当前类别的准确率
        if np.sum(cls_mask) > 0:  # 确保有样本
            cls_true = np.array(all_true)[cls_mask]
            cls_pred = np.array(all_pred)[cls_mask]
            cls_acc = accuracy_score(cls_true, cls_pred)
            class_accuracies.append(cls_acc)
        else:
            class_accuracies.append(0.0)  # 如果没有样本，准确率设为0

    aa = np.mean(class_accuracies)
    kappa = cohen_kappa_score(all_true, all_pred)

    print(f"📊📊 评估结果 ({stage}阶段{' 轮次' + str(round_idx) if round_idx is not None else ''}):")
    print(f"  整体准确率 (OA): {oa:.4f}")
    print(f"  平均准确率 (AA): {aa:.4f}")
    print(f"  Kappa系数: {kappa:.4f}")

    # 打印每个类别的准确率
    print("  类别准确率:")
    for i, cls in enumerate(unique_classes):
        print(f"    类别 {cls}: {class_accuracies[i]:.4f}")

    # 保存分类结果为.mat文件（包含坐标信息）
    os.makedirs('mat_results', exist_ok=True)
    filename = f'mat_results/classification_results_{stage}'
    if round_idx is not None:
        filename += f'_round{round_idx}'
    filename += '.mat'

    # 创建二维标签映射
    max_y = max(coord[0] for coord in all_coords) + 1
    max_x = max(coord[1] for coord in all_coords) + 1

    true_map = np.zeros((max_y, max_x), dtype=np.int16)
    pred_map = np.zeros((max_y, max_x), dtype=np.int16)

    # 填充二维映射
    for (y, x), true_label, pred_label in zip(all_coords, all_true, all_pred):
        true_map[y, x] = true_label + 1
        pred_map[y, x] = pred_label + 1

    # 保存为MAT文件
    sio.savemat(filename, {
        'true_map': true_map,
        'pred_map': pred_map,
        'oa': oa,
        'aa': aa,
        'kappa': kappa
    })
    print(f"✅ 分类结果已保存为MAT文件: {filename} (包含二维映射)")

    return oa, aa, kappa, filename

def collect_replay_samples(classifier, dataset, classes, samples_per_class, args, stage="base", save_path=None):
    """
    收集每个类别的高置信度样本用于旧类回放，并保存到本地文件

    参数:
        classifier: 训练好的分类模型
        dataset: 原始数据集
        classes: 需要收集的类别列表
        samples_per_class: 每个类别收集的样本数量
        args: 包含设备等信息的参数对象
        stage: 阶段标识 ('base' 或 'incremental_roundX')
        save_path: 保存回放数据集的路径 (可选)

    返回:
        TensorDataset: 包含高置信度样本的数据集
        str: 保存的文件路径
    """
    # 创建固定的数据加载器（shuffle=False）
    fixed_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 关键：固定顺序
        num_workers=args.num_workers
    )

    # 初始化存储结构
    high_confidence_samples = defaultdict(list)

    classifier.eval()
    with torch.no_grad():
        # 遍历整个数据集
        for x1, x2, labels in fixed_loader:
            x1, x2, labels = x1.to(args.device), x2.to(args.device), labels.to(args.device)

            # 获取模型预测和置信度
            logits = classifier(x1, x2)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            # 遍历batch中的每个样本
            for i in range(x1.size(0)):
                true_label = labels[i].item()

                # 只处理指定类别的样本
                if true_label in classes:
                    confidence = confidences[i].item()
                    sample_data = {
                        'x1': x1[i].cpu().clone(),
                        'x2': x2[i].cpu().clone(),
                        'label': labels[i].cpu().clone(),
                        'confidence': confidence
                    }

                    # 添加到当前类别的样本列表
                    high_confidence_samples[true_label].append(sample_data)

    # 准备回放数据集
    replay_x1 = []
    replay_x2 = []
    replay_labels = []

    for cls_id, samples in high_confidence_samples.items():
        # 按置信度降序排序
        samples.sort(key=lambda x: x['confidence'], reverse=True)

        # 保留前k个样本
        top_samples = samples[:samples_per_class]

        for sample in top_samples:
            replay_x1.append(sample['x1'])
            replay_x2.append(sample['x2'])
            replay_labels.append(sample['label'])

    # 转换为张量
    if len(replay_x1) > 0:  # 检查列表是否非空
        replay_x1 = torch.stack(replay_x1)
        replay_x2 = torch.stack(replay_x2)
        replay_labels = torch.stack(replay_labels)
        replay_dataset = TensorDataset(replay_x1, replay_x2, replay_labels)
    else:
        replay_dataset = None

    num_samples = len(replay_x1)  # 直接使用列表长度获取样本数量
    print(f"✅ 收集了{num_samples}个回放样本 ({samples_per_class}个/类, {len(classes)}个类)")

    # 自动生成保存路径（如果未提供）
    if save_path is None:
        cache_dir = 'replay_cache'
        os.makedirs(cache_dir, exist_ok=True)
        min_class = min(classes)
        max_class = max(classes)
        save_path = os.path.join(
            cache_dir,
            f"replay_{stage}_classes{min_class}to{max_class}_"
            f"samples{samples_per_class}.pth"
        )

    # 保存到本地文件
    if replay_dataset is not None:
        torch.save(replay_dataset, save_path)
        print(f"💾💾 回放数据集已保存至: {save_path}")
    else:
        print("⚠️ 未生成回放数据集，跳过保存")
        save_path = None

    return replay_dataset, save_path


def incremental_prepare_classification_testset(t1, classes, vqdit_model, replace_ratio, args, t_incremental, round_idx, latent_meanflow=None):
    """
    准备分类测试集t9 - 增加缓存支持并加入 sample_number 和 replace_ratio 到文件名
    """
    # 检查是否使用缓存
    if args.use_cached_t9:
        # 构造带 sample_number 和 replace_ratio 的缓存路径
        cache_dir = 'datasets/cache'
        os.makedirs(cache_dir, exist_ok=True)

        save_path = os.path.join(
            cache_dir,
            f't9_{args.data_name}_patch{args.patch_size}'
            f'_round_idx{round_idx}_rr{replace_ratio:.2f}.pth'
        )

        if os.path.exists(save_path):
            print(f"💾 加载缓存的 t9 数据集: {save_path}")
            return torch.load(save_path)
        else:
            print(f"⚠️ 缓存文件 {save_path} 不存在，将重新生成 t9 数据集")

    # 将所有操作放在CPU上
    device = torch.device('cpu')
    vqdit_model = vqdit_model.to(device)
    if latent_meanflow is not None:
        latent_meanflow.device = device

    # 将t1中的数据移到CPU
    t1_cpu = []
    for i in range(len(t1)):
        t1_cpu.append((
            t1[i][0].cpu(),
            t1[i][1].cpu(),
            t1[i][2].cpu(),
            t1[i][3].cpu(),
            t1[i][4].cpu()
        ))

    indices = [i for i, (_, _, label, _, _) in enumerate(t1_cpu) if label.item() in classes]
    images = []
    patches = []
    labels = []
    ys_list = []
    xs_list = []

    for idx in indices:
        image, patch, label, y, x = t1_cpu[idx]
        ys_list.append(y)
        xs_list.append(x)

        if torch.rand(1).item() < replace_ratio:
            with torch.no_grad():
                generated_patch = latent_meanflow.increment_sample(
                    model=vqdit_model,
                    n_per_class=1,
                    classes=[label.item()],
                    device=device
                )
                generated_patch = vqdit_model.decode_continuous(generated_patch)
            patches.append(generated_patch.squeeze(0))
        else:
            patches.append(patch)
        images.append(image)
        labels.append(label)

    unique_classes = sorted(set([lbl.item() for lbl in labels]))
    combined_samples = []

    # 波段选择 - 提前选择波段以减少计算量
    in_channels = vqdit_model.in_channels
    band_indices = []
    for band in [100, 60, 30]:
        if band < in_channels:
            band_indices.append(band)
        else:
            band_indices.append(in_channels - 1)

    def enhance_image(img_tensor):
        """图像增强函数：亮度、对比度和尺寸"""
        # 只处理选定的波段
        img_tensor = img_tensor[band_indices]

        # 线性拉伸增强对比度
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        if max_val > min_val:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        else:
            img_tensor = torch.clamp(img_tensor, 0, 1)  # 如果所有值相同，确保在[0,1]范围

        # 伽马校正增加亮度 (gamma < 1 使图像变亮)
        img_tensor = torch.clamp(img_tensor ** 0.7, 0, 1)

        # 放大图像 (4倍)
        scale_factor = 4
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=True
        ).squeeze(0)

        # 对每个通道独立应用高斯模糊
        num_channels = img_tensor.size(0)
        blurred_channels = []

        # 创建高斯模糊核
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        kernel = kernel.view(1, 1, 3, 3).to(img_tensor.device)

        for c in range(num_channels):
            # 对单个通道进行卷积
            channel = img_tensor[c:c + 1].unsqueeze(0)  # [1, 1, H, W]
            blurred = F.conv2d(
                channel,
                kernel,
                padding=1
            )
            blurred_channels.append(blurred.squeeze(0))

        img_tensor = torch.cat(blurred_channels, dim=0)

        return img_tensor

    sample_number = 50  # 每类采样数量
    for class_id in unique_classes:
        class_indices = [i for i, lbl in enumerate(labels) if lbl.item() == class_id]
        selected_indices = class_indices if len(class_indices) < sample_number else random.sample(class_indices, sample_number)

        # 应用图像增强 - 只增强选定波段的图像
        class_images = []
        class_patches = []
        for idx in selected_indices:
            # 只处理选定的波段
            class_images.append(enhance_image(images[idx]))
            class_patches.append(enhance_image(patches[idx]))

        combined_samples.extend(class_images)
        combined_samples.extend(class_patches)

    combined_samples = torch.stack(combined_samples)

    # 高质量保存
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    samples_per_class = len(selected_indices)

    # 创建高分辨率图像
    grid = make_grid(combined_samples, nrow=samples_per_class, padding=10, pad_value=0.8)

    # 转换为PIL图像进行质量调整
    pil_img = TF.to_pil_image(grid)

    # 保存为高质量PNG
    save_path_img = f"images/classification_testset_{timestamp}.png"
    pil_img.save(save_path_img, "PNG", quality=95, dpi=(300, 300))
    print(f"📸 保存高质量图像: {save_path_img}")

    # 返回测试集
    images_tensor = torch.stack(images).to(args.device)
    patches_tensor = torch.stack(patches).to(args.device)
    labels_tensor = torch.tensor(labels, device=args.device).squeeze()
    ys_tensor = torch.tensor(ys_list, device=args.device)
    xs_tensor = torch.tensor(xs_list, device=args.device)

    t9_dataset = TensorDataset(images_tensor, patches_tensor, labels_tensor, ys_tensor, xs_tensor)

    if t_incremental is not None:
        # 确保t_incremental有5个元素
        if len(t_incremental.tensors) < 5:
            print(f"⚠️ t_incremental 缺少坐标信息，元素数量: {len(t_incremental.tensors)}")
            # 添加默认坐标
            default_ys = torch.zeros(len(t_incremental), dtype=torch.long)
            default_xs = torch.zeros(len(t_incremental), dtype=torch.long)
            t_incremental = TensorDataset(
                t_incremental.tensors[0],
                t_incremental.tensors[1],
                t_incremental.tensors[2],
                default_ys,
                default_xs
            )

        # 合并现有的t_incremental和新的t9_dataset
        combined_images = torch.cat([t_incremental.tensors[0], t9_dataset.tensors[0]], dim=0)
        combined_patches = torch.cat([t_incremental.tensors[1], t9_dataset.tensors[1]], dim=0)
        combined_labels = torch.cat([t_incremental.tensors[2], t9_dataset.tensors[2]], dim=0)
        combined_ys = torch.cat([t_incremental.tensors[3], t9_dataset.tensors[3]], dim=0)
        combined_xs = torch.cat([t_incremental.tensors[4], t9_dataset.tensors[4]], dim=0)

        t_incremental = TensorDataset(
            combined_images,
            combined_patches,
            combined_labels,
            combined_ys,
            combined_xs
        )
    else:
        t_incremental = t9_dataset

    return t9_dataset, t_incremental


def count_class_distribution(dataset, dataset_name, round_idx=None):
    """
    统计数据集中的类别分布并打印结果

    参数:
    - dataset: TensorDataset 对象
    - dataset_name: 数据集名称 (用于打印)
    - round_idx: 增量轮次 (可选)
    """
    # 确保数据集不为空
    if len(dataset) == 0:
        print(f"⚠️ {dataset_name} 是空数据集")
        return {}

    # 获取标签张量
    labels_tensor = dataset.tensors[2]

    # 如果标签是多维的，展平
    if labels_tensor.dim() > 1:
        labels_tensor = labels_tensor.squeeze()

    # 统计类别分布
    class_counts = {}
    for label in labels_tensor:
        class_id = label.item()
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # 打印结果
    round_prefix = f"round {round_idx} - " if round_idx is not None else ""
    print(f"📊 {round_prefix}{dataset_name} 类别分布:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}: {count} 个样本")

    # 打印总数
    total_samples = sum(class_counts.values())
    print(f"  总计: {total_samples} 个样本 ({len(class_counts)} 个类别)")

    return class_counts

def prepare_incremental_classification_trainset(train_dataset, full_image, pad_image, vqdit_model, args, latent_meanflow):
    """
    准备分类训练集t10，并支持缓存机制。
    参数:
        num_local_samples: 每个x1对应的局部搜索样本数量
        num_global_samples: 每个x1对应的全局搜索样本数量
        num_generated_samples: 每个x1对应的生成样本数量
    """

    # 构造缓存路径
    cache_dir = 'datasets/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # 从args中获取三个控制参数
    num_local_samples = args.incremental_num_local_samples_t10
    num_global_samples = args.incremental_num_global_samples_t10
    num_generated_samples = args.incremental_num_generated_samples_t10

    # 构造唯一的缓存文件名，包含关键参数
    save_path = os.path.join(
        cache_dir,
        f't10_{args.data_name}_patch{args.patch_size}'
        f'_local{num_local_samples}'
        f'_global{num_global_samples}'
        f'_gen{num_generated_samples}.pth'
    )

    # 如果启用缓存且文件存在，直接加载
    if args.use_cached_t10 and os.path.exists(save_path):
        print(f"💾 加载缓存的 t10 数据集: {save_path}")
        return torch.load(save_path)
    elif args.use_cached_t10 and not os.path.exists(save_path):
        print(f"⚠️ 缓存文件 {save_path} 不存在，将重新生成 t10 数据集")

    half = args.patch_size // 2
    device = args.device

    # 将pad_image移动到目标设备（如果不在）
    pad_image = pad_image.to(device)

    patches_x1 = []
    patches_x2 = []
    labels = []

    # 添加进度条
    total_points = len(train_dataset)
    pbar = tqdm(total=total_points, desc="🔧 生成分类训练集t10", unit="点")
    generated_samples_count = 0

    # 遍历t2中的每个训练点
    for i in range(len(train_dataset)):
        # 获取当前训练点的信息，并移动到目标设备
        patch_x1, label, y, x = train_dataset[i]
        patch_x1 = patch_x1.to(device)
        if torch.is_tensor(label):
            label = label.to(device)
        else:
            label = torch.tensor(label, device=device)

        # 创建匹配器
        matcher = EnhancedHybridMatcher(full_image, (y, x), device=device, normalize=True)

        # 执行混合搜索（局部和全局）
        results = matcher.hybrid_selection(
            local_radius=args.local_radius,
            local_topk=num_local_samples,  # 确保足够抽取
            superpixel_topk=2,
            global_topk=num_global_samples,  # 确保足够抽取
            min_dist=args.min_dist,
            spectral_threshold=args.spectral_threshold,
            context_threshold=args.context_threshold
        )

        # 获取候选点坐标
        local_coords = results['local']  # [n, 2]
        global_coords = results['global']  # [n, 2]

        # 1. 添加局部搜索样本
        if len(local_coords) > 0:
            selected_indices = np.random.choice(
                len(local_coords),
                size=min(num_local_samples, len(local_coords)),
                replace=False
            )
            for idx in selected_indices:
                coord = local_coords[idx]
                y_coord = int(round(coord[0]))
                x_coord = int(round(coord[1]))
                # 提取patch并移动到设备
                patch = pad_image[:, y_coord:y_coord + 2 * half + 1, x_coord:x_coord + 2 * half + 1].to(device)
                patches_x1.append(patch_x1)  # 已经在设备上
                patches_x2.append(patch)  # 已经移动到设备
                labels.append(label)  # 已经在设备上
                generated_samples_count += 1

        # 2. 添加全局搜索样本（类似处理）
        if len(global_coords) > 0:
            selected_indices = np.random.choice(
                len(global_coords),
                size=min(num_global_samples, len(global_coords)),
                replace=False
            )
            for idx in selected_indices:
                coord = global_coords[idx]
                y_coord = int(round(coord[0]))
                x_coord = int(round(coord[1]))
                patch = pad_image[:, y_coord:y_coord + 2 * half + 1, x_coord:x_coord + 2 * half + 1].to(device)
                patches_x1.append(patch_x1)
                patches_x2.append(patch)
                labels.append(label)
                generated_samples_count += 1

        # 3. 添加生成样本
        for _ in range(num_generated_samples):
            with torch.no_grad():
                generated_z = latent_meanflow.increment_sample(
                    model=vqdit_model,
                    n_per_class=1,
                    classes=[label.item()],  # 注意：label是张量，用.item()取标量
                    device=device
                )
                generated_patch = vqdit_model.decode_continuous(generated_z)
            patches_x1.append(patch_x1)
            patches_x2.append(generated_patch.squeeze(0))  # 已经在设备上
            labels.append(label)
            generated_samples_count += 1

        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            "当前点": f"{i + 1}/{total_points}",
            "已生成样本": generated_samples_count
        })

    # 关闭进度条
    pbar.close()

    # 转换为张量
    patches_x1 = torch.stack(patches_x1)
    patches_x2 = torch.stack(patches_x2)
    labels = torch.tensor(labels, device=device)

    print(f"✅ t10数据集生成完成: 总点数 = {total_points}, 总样本数 = {len(patches_x1)}")

    # 保存到缓存路径
    if hasattr(args, 'use_cached_t10') and args.use_cached_t10:
        t10_dataset = TensorDataset(patches_x1, patches_x2, labels)
        torch.save(t10_dataset, save_path)
        print(f"💾 t10数据集已保存至: {save_path}")

    return TensorDataset(patches_x1.cpu(), patches_x2.cpu(), labels.cpu())


def print_trainable_params(model, model_name="Model"):
    """打印模型的可训练参数信息"""
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    print(f"\n{' Trainable Parameters Analysis ':=^80}")
    print(f"{model_name} Parameter Status:")

    # 递归检查参数状态
    def print_module(module, prefix=""):
        nonlocal total_params, trainable_params, frozen_params
        for name, param in module.named_parameters(recurse=False):
            param_name = f"{prefix}{name}"
            param_count = param.numel()

            if param.requires_grad:
                status = "TRAINABLE"
                trainable_params += param_count
            else:
                status = "FROZEN  "
                frozen_params += param_count

            total_params += param_count
            print(f"  {param_name:<50} | {status} | {param_count:,} parameters")

        # 递归处理子模块
        for name, child in module.named_children():
            child_prefix = f"{prefix}{name}."
            print_module(child, child_prefix)

    # 从根模块开始递归
    print_module(model, "")

    # 打印统计信息
    print("\n" + "-" * 80)
    print(f"  {'Total Parameters':<50} | {total_params:,}")
    print(f"  {'Trainable Parameters':<50} | {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
    print(f"  {'Frozen Parameters':<50} | {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)")
    print("=" * 80 + "\n")


def train_incremental_classifier(model, train_dataset, test_dataset, new_classes, args):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 添加新类别
    model.add_new_classes(new_classes)

    # 如果有指定分类模型路径，则加载模型
    if hasattr(args, 'incremental_classifier_path') and args.incremental_classifier_path:
        classifier_path = args.incremental_classifier_path
        if os.path.exists(classifier_path):
            print(f"🔍 尝试加载增量分类模型: {classifier_path}")
            try:
                # 加载模型状态
                state_dict = torch.load(classifier_path, map_location=args.device)

                # 处理可能的参数不匹配问题
                model_state = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items()
                                   if k in model_state and v.size() == model_state[k].size()}

                # 加载匹配的参数
                model_state.update(pretrained_dict)
                model.load_state_dict(model_state)

                # 打印加载信息
                loaded_params = len(pretrained_dict)
                total_params = len(model_state)
                print(f"✅ 成功加载增量分类模型: {loaded_params}/{total_params} 参数匹配 ({loaded_params / total_params * 100:.1f}%)")
                print("⏭ 跳过训练阶段，直接返回模型")
                return model, None
            except Exception as e:
                print(f"⚠️ 加载增量分类模型失败: {str(e)}")
                print("🆕 将使用传入的分类模型")
        else:
            print(f"⚠️ 分类模型文件不存在: {classifier_path}")

    # 冻结部分参数
    model.freeze_for_incremental_learning()

    # 创建优化器（仅优化可训练参数）
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = torch.optim.Adam(trainable_params, lr=args.classifier_lr)
    best_acc = 0.0
    best_model_state = None  # 保存最佳模型状态

    print_trainable_params(model, "Initial Classifier")

    # 初始评估
    model.eval()
    overall_acc, class_acc = evaluate(model, test_loader, args.device)
    print(f"*Epoch {0}/{args.classifier_epochs}: Acc={overall_acc:.2f}%")
    print("Class-wise Accuracy:")
    for cls, acc in class_acc.items():
        print(f"  Class {cls}: {acc:.2f}%")

    for epoch in range(args.classifier_epochs):
        model.train()
        total_loss = 0.0
        loss_details = defaultdict(float)  # 用于收集详细损失

        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            x1, x2, labels = x1.to(args.device), x2.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            logits, loss_value, losses_dict = model(x1, x2, labels)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

            # 收集详细损失
            for key, value in losses_dict.items():
                loss_details[key] += value

        # 计算平均详细损失
        avg_losses = {k: v / len(train_loader) for k, v in loss_details.items()}
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.classifier_epochs:
            # 在测试集上评估
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, patches, labels, _, _ in test_loader:
                    images, patches, labels = images.to(args.device), patches.to(args.device), labels.to(args.device)
                    logits = model(images, patches)
                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                overall_acc, class_acc = evaluate(model, test_loader, args.device)

                print(f"Epoch {epoch + 1}/{args.classifier_epochs}: "
                      f"Total Loss={total_loss / len(train_loader):.4f}, "
                      f"{loss_str}, "
                      f"Acc={overall_acc:.2f}%")

                # 打印每个类别的准确率
                print("Class-wise Accuracy:")
                for cls, acc in class_acc.items():
                    print(f"  Class {cls}: {acc:.2f}%")

            accuracy = correct / total

            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                best_model_path = f"checkpoints/base_classifier_{timestamp}_{accuracy:.4f}.pth"
                # 直接保存到文件，不保存引用
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ 增量分类模型已保存至: {best_model_path}")

    # 训练结束后加载最佳模型状态
    if best_model_path and os.path.exists(best_model_path):
        print(f"🔁 从文件加载最佳模型: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

    # 在训练结束后计算回放数据集
    replay_dataset = compute_replay_dataset(model, train_dataset, new_classes, args)

    return model, replay_dataset


def compute_replay_dataset(model, train_dataset, new_classes, args):
    """计算当前增量轮次的回放数据集"""
    print("🧠 计算当前轮次回放数据...")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model.eval()

    # 存储回放数据
    replay_x1 = []
    replay_x2 = []
    replay_labels = []

    # 为每个新类别选择代表性样本
    samples_per_class = args.replay_samples_per_class

    for class_id in new_classes:
        # 获取该类别的所有样本
        class_indices = [i for i, (_, _, label) in enumerate(train_dataset)
                         if label.item() == class_id]

        if not class_indices:
            print(f"⚠️ 类别 {class_id} 无训练样本，跳过回放数据生成")
            continue

        # 随机选择样本（或使用更复杂的方法）
        selected_indices = np.random.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False
        )

        # 添加选中的样本
        for idx in selected_indices:
            x1, x2, label = train_dataset[idx]
            replay_x1.append(x1)
            replay_x2.append(x2)
            replay_labels.append(label)

    # 转换为张量
    if replay_x1:
        replay_x1 = torch.stack(replay_x1)
        replay_x2 = torch.stack(replay_x2)
        replay_labels = torch.tensor(replay_labels)

        replay_dataset = TensorDataset(replay_x1, replay_x2, replay_labels)
        print(f"✅ 生成回放数据: {len(replay_dataset)} 个样本")
        return replay_dataset
    else:
        print("⚠️ 未生成任何回放数据")
        return None

def visualize_features(model, test_loader, epoch, device):
    """
    可视化特征分布
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        epoch: 当前epoch
        device: 使用的设备
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, patches, labels in test_loader:
            images, patches, labels = images.to(device), patches.to(device), labels.to(device)

            # 获取特征和预测结果
            if model.training:
                _, _, _, features = model(images, patches, return_features=True)
            else:
                _, features = model(images, patches, return_features=True)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    # 合并所有特征和标签
    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_features)

    # 创建可视化
    plt.figure(figsize=(12, 10))

    # 获取类别数量
    num_classes = len(np.unique(all_labels))
    cmap = plt.get_cmap('tab20', num_classes)

    # 为每个类别绘制点
    for cls in range(num_classes):
        idx = np.where(all_labels == cls)
        plt.scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            color=cmap(cls),
            label=f'Class {cls}',
            alpha=0.6,
            s=10
        )

    plt.title(f'Feature Space Visualization (Epoch {epoch})')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)

    # 保存图像
    os.makedirs('feature_visualizations', exist_ok=True)
    save_path = f'feature_visualizations/features_epoch_{epoch}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ 特征可视化已保存至: {save_path}")
    return save_path


def visualize_decision_boundary(model, test_loader, epoch, device, n_samples=500):
    """
    可视化决策边界（使用PCA降维）
    """
    model.eval()
    features = []
    labels = []

    # 采样部分数据点
    with torch.no_grad():
        for i, (images, patches, lbls) in enumerate(test_loader):
            if i * images.size(0) >= n_samples:
                break

            images, patches, lbls = images.to(device), patches.to(device), lbls.to(device)

            # 获取特征
            if model.training:
                _, _, _, feats = model(images, patches, return_features=True)
            else:
                _, feats = model(images, patches, return_features=True)

            features.append(feats.cpu())
            labels.append(lbls.cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # 使用PCA降维到2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(features)

    # 创建网格
    x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
    y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 创建网格点特征
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 将网格点逆变换回原始空间
    grid_features = pca.inverse_transform(grid_points)
    grid_tensor = torch.tensor(grid_features, dtype=torch.float32).to(device)

    # 预测网格点类别
    model.eval()
    with torch.no_grad():
        logits = model.classifier.linear(grid_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    # 绘制决策边界
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.4, cmap=plt.cm.tab20)

    # 绘制原始数据点
    num_classes = len(np.unique(labels))
    cmap = plt.get_cmap('tab20', num_classes)

    for cls in range(num_classes):
        idx = np.where(labels == cls)
        plt.scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            color=cmap(cls),
            label=f'Class {cls}',
            edgecolor='k',
            s=30
        )

    plt.title(f'Decision Boundary (Epoch {epoch})')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 保存图像
    os.makedirs('decision_boundaries', exist_ok=True)
    save_path = f'decision_boundaries/boundary_epoch_{epoch}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ 决策边界可视化已保存至: {save_path}")
    return save_path

def evaluate(model, test_loader, device):
    """评估函数，自动获取模型输出类别数并统计准确率"""
    model.eval()
    num_classes = model.classifier.linear.out_features  # 获取当前模型输出类别数

    # 初始化统计变量
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, patches, labels,_,_ in test_loader:
            images, patches, labels = images.to(device), patches.to(device), labels.to(device)
            logits = model(images, patches)
            _, predicted = torch.max(logits, 1)

            # 统计整体准确率
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            total_correct += batch_correct
            total_samples += batch_total

            # 统计每个类别的准确率
            for i in range(batch_total):
                label = labels[i].item()
                # 确保标签在有效范围内
                if 0 <= label < num_classes:
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1

    # 计算整体准确率
    overall_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    # 计算每个类别的准确率
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0.0

    return overall_accuracy, class_accuracy


# -----------------------------
# 增量训练函数
# -----------------------------

def incremental_training(args, vqdit_model, classifier_model, t1, full_dataset, round_idx, t_incremental,
                         best_replay_dataset, base_replay_path):
    """增量阶段训练"""
    print(f"🚀 增量阶段第{round_idx + 1}轮")

    # 缓存目录
    cache_dir = 'datasets/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 准备新增类别
    start_class = args.base_classes + round_idx * args.new_classes_per_round
    end_class = start_class + args.new_classes_per_round
    new_classes = list(range(start_class, end_class))

    # ===== t6数据集 (增量训练数据) =====
    # 构造唯一缓存文件名
    t6_cache_name = (
        f"t6_{args.data_name}_round{round_idx}_"
        f"classes{start_class}to{end_class - 1}_"
        f"patch{args.patch_size}_"
        f"local{args.num_samples_local_class}_"
        f"global{args.num_samples_global_class}.pth"
    )
    t6_cache_path = os.path.join(cache_dir, t6_cache_name)

    new_train_dataset = HyperSpectralDataset(
        data_name=args.data_name,
        mode='train',
        classes=list(range(start_class + 1, end_class + 1)),
        num_train_samples_per_class=args.new_samples_per_class,
        patch_size=args.patch_size,
        normalize=args.normalize,
        method=args.norm_method
    )

    # 创建数据加载器
    new_train_loader = DataLoader(
        new_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 检查t6缓存
    if os.path.exists(t6_cache_path):
        print(f"💾 加载缓存的增量训练数据集t6: {t6_cache_path}")
        t6 = torch.load(t6_cache_path)
    else:
        print("🛠 生成增量训练数据集t6...")
        # 准备增量训练数据 特别注意classes要+1！！！

        # 生成增强训练集t6
        t6 = generate_patches_with_types(
            train_loader=new_train_loader,
            image=full_dataset.full_image,
            image_pad=full_dataset.pad_image,
            device=args.device,
            half=args.patch_size // 2,
            local_radius=args.local_radius,
            local_topk=args.num_samples_local_class,
            # local_topk=20,
            superpixel_topk=3,
            global_topk=args.num_samples_global_class,
            # global_topk=50,
            min_dist=args.min_dist,
            spectral_threshold=args.spectral_threshold,
            context_threshold=args.context_threshold
        )
        torch.save(t6, t6_cache_path)
        print(f"💾 保存增量训练数据集t6到: {t6_cache_path}")

    # ===== t7数据集 (所有已见类别) =====
    # 构造唯一缓存文件名
    all_classes = list(range(0, args.base_classes)) + new_classes
    t7_cache_name = (
        f"t7_{args.data_name}_round{round_idx}_"
        f"classes{min(all_classes)}to{max(all_classes)}_"
        f"patch{args.patch_size}.pth"
    )
    t7_cache_path = os.path.join(cache_dir, t7_cache_name)

    # 检查t7缓存
    if os.path.exists(t7_cache_path):
        print(f"💾 加载缓存的已见类别数据集t7: {t7_cache_path}")
        t7 = torch.load(t7_cache_path)
    else:
        print("🛠 准备已见类别数据集t7...")
        # 准备测试数据t7（所有已见类别）
        t7_indices = [i for i, (_, _, label, _, _) in enumerate(t1) if label.item() in all_classes]
        t7 = TensorDataset(
            t1.tensors[0][t7_indices],
            t1.tensors[2][t7_indices].squeeze(),
            t1.tensors[3][t7_indices].squeeze(),
            t1.tensors[4][t7_indices].squeeze()
        )
        torch.save(t7, t7_cache_path)
        print(f"💾 保存已见类别数据集t7到: {t7_cache_path}")

    # ===== t8数据集 (新类别) =====
    # 构造唯一缓存文件名
    t8_cache_name = (
        f"t8_{args.data_name}_round{round_idx}_"
        f"classes{min(new_classes)}to{max(new_classes)}_"
        f"patch{args.patch_size}.pth"
    )
    t8_cache_path = os.path.join(cache_dir, t8_cache_name)

    # 检查t8缓存
    if os.path.exists(t8_cache_path):
        print(f"💾 加载缓存的新类别数据集t8: {t8_cache_path}")
        t8 = torch.load(t8_cache_path)
    else:
        print("🛠 准备新类别数据集t8...")
        # 准备测试数据t8（仅新类别）
        t8_indices = [i for i, (_, _, label, _, _) in enumerate(t1) if label.item() in new_classes]
        t8 = TensorDataset(
            t1.tensors[0][t8_indices],
            t1.tensors[2][t8_indices].squeeze(),
            t1.tensors[3][t8_indices].squeeze(),
            t1.tensors[4][t8_indices].squeeze()
        )
        torch.save(t8, t8_cache_path)
        print(f"💾 保存新类别数据集t8到: {t8_cache_path}")

    # 在调用fine_tune前创建latent_meanflow
    latent_meanflow = LatentMeanFlow(
        latent_dim=args.latent_dim,
        image_size=args.patch_size,
        num_classes=len(new_classes),  # 使用新增类别数量
        normalizer=['minmax', None, None],
        flow_ratio=0,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0,
        cfg_scale=2
    )

    train_t6_loader = DataLoader(
        t6,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_t7_loader = DataLoader(
        t7,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_t8_loader = DataLoader(
        t8,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # 4. 微调VQDiT模型
    print("🔧 微调VQDiT模型...")
    incremental_model, latent_meanflow = fine_tune_vqdit_hyperspectral(vqdit_model, latent_meanflow,
                                                                       train_t6_loader, test_t8_loader, train_t6_loader,
                                                                       test_t8_loader, args, new_classes, round_idx)

    # 分别创建 t9_dataset 和 t_incremental 的缓存路径
    t9_dataset_cache_path = os.path.join(
        cache_dir,
        f't9_dataset_{args.data_name}_round{round_idx}_'
        f'classes{min(new_classes)}to{max(new_classes)}_'
        f'patch{args.patch_size}_rr{args.incremental_replace_ratio:.2f}.pth'
    )

    t_incremental_cache_path = os.path.join(
        cache_dir,
        f't_incremental_{args.data_name}_round{round_idx}_'
        f'classes{min(new_classes)}to{max(new_classes)}_'
        f'patch{args.patch_size}.pth'
    )

    # 检查缓存
    t9_cached = os.path.exists(t9_dataset_cache_path) and args.use_cached_t9
    t_inc_cached = os.path.exists(t_incremental_cache_path) and args.use_cached_t9

    if t9_cached and t_inc_cached:
        print(f"💾 加载缓存的分类测试集t9: {t9_dataset_cache_path}")
        print(f"💾 加载缓存的增量测试集t_incremental: {t_incremental_cache_path}")
        t9_dataset = torch.load(t9_dataset_cache_path)
        t_incremental = torch.load(t_incremental_cache_path)

        # 打印加载结果
        print(f"✅ t9_dataset大小: {len(t9_dataset)}")
        print(f"✅ t_incremental大小: {len(t_incremental)}")

        # 统计数据集类别分布
        count_class_distribution(t9_dataset, "缓存的t9_dataset", round_idx)
        count_class_distribution(t_incremental, "缓存的t_incremental", round_idx)
    else:
        if args.use_cached_t9:
            print(f"⚠️ 部分缓存不存在，将重新生成")
            if not t9_cached:
                print(f"  - t9_dataset缓存不存在: {t9_dataset_cache_path}")
            if not t_inc_cached:
                print(f"  - t_incremental缓存不存在: {t_incremental_cache_path}")

        print("📊 准备分类测试集t9...")
        t9_dataset, t_incremental = incremental_prepare_classification_testset(
            t1,
            new_classes,
            incremental_model,
            args.incremental_replace_ratio,
            args,
            t_incremental,  # 传入当前的t_incremental
            round_idx,
            latent_meanflow
        )

        if args.use_cached_t9:
            # 分别保存t9_dataset和t_incremental
            torch.save(t9_dataset, t9_dataset_cache_path)
            torch.save(t_incremental, t_incremental_cache_path)
            print(f"💾 保存分类测试集t9到: {t9_dataset_cache_path}")
            print(f"💾 保存增量测试集t_incremental到: {t_incremental_cache_path}")

            # 打印保存结果
            print(f"✅ t9_dataset大小: {len(t9_dataset)}")
            print(f"✅ t_incremental大小: {len(t_incremental)}")

    # 6. 准备分类训练集t10
    # 构造唯一缓存文件名
    t10_cache_name = (
        f"t10_{args.data_name}_round{round_idx}_"
        f"classes{min(new_classes)}to{max(new_classes)}_"
        f"patch{args.patch_size}_"
        f"local{args.incremental_num_local_samples_t10}_"
        f"global{args.incremental_num_global_samples_t10}_"
        f"gen{args.incremental_num_generated_samples_t10}.pth"
    )
    t10_cache_path = os.path.join(cache_dir, t10_cache_name)

    # 检查t10缓存
    if os.path.exists(t10_cache_path) and args.use_cached_t10:
        print(f"💾 加载缓存的分类训练集t10: {t10_cache_path}")
        t10 = torch.load(t10_cache_path)
    else:
        print("📚 准备分类训练集t10...")
        t10 = prepare_incremental_classification_trainset(
            train_dataset=new_train_dataset,
            full_image=full_dataset.full_image,
            pad_image=full_dataset.pad_image,
            vqdit_model=vqdit_model,
            args=args,
            latent_meanflow=latent_meanflow
        )
        if args.use_cached_t10:
            torch.save(t10, t10_cache_path)
            print(f"💾 保存分类训练集t10到: {t10_cache_path}")

    # ===== 将基础阶段的回放数据加入到训练集 =====
    if os.path.exists(base_replay_path):
        print(f"💾💾 加载基础阶段回放数据: {base_replay_path}")
        base_replay_dataset = torch.load(base_replay_path)

        # 合并到当前训练集
        t10_x1, t10_x2, t10_labels = t10.tensors
        replay_x1, replay_x2, replay_labels = base_replay_dataset.tensors

        # 确保设备一致
        t10_x1 = t10_x1.to(args.device)
        t10_x2 = t10_x2.to(args.device)
        t10_labels = t10_labels.to(args.device)
        replay_x1 = replay_x1.to(args.device)
        replay_x2 = replay_x2.to(args.device)
        replay_labels = replay_labels.to(args.device)

        combined_x1 = torch.cat([t10_x1, replay_x1], dim=0)
        combined_x2 = torch.cat([t10_x2, replay_x2], dim=0)
        combined_labels = torch.cat([t10_labels, replay_labels], dim=0)

        t10 = TensorDataset(combined_x1, combined_x2, combined_labels)
        print(f"✅ 合并后训练集大小: {len(t10)} (新增类: {len(t10_x1)}, 基础回放: {len(replay_x1)})")
    else:
        print(f"⚠️ 基础回放数据不存在: {base_replay_path}")

    # 获取所有旧类别（基础阶段已学习的类别）
    old_classes = list(range(0, args.base_classes + round_idx * args.new_classes_per_round))

    # 标记这些类别为旧类
    classifier_model.prompt_module.prepare_for_incremental_learning(old_classes)

    # 7. 微调分类模型
    print("🔧🔧🔧🔧 微调分类模型...")
    classifier_model, replay_dataset = train_incremental_classifier(  # 修改这里：接收两个返回值
        classifier_model,
        t10,  # 使用包含回放数据的训练集
        t_incremental,
        new_classes,
        args
    )

    # 8. 评估增量分类模型
    print("📊📊📊📊 评估增量分类模型...")
    # 创建测试数据加载器
    incremental_test_loader = DataLoader(
        t_incremental,  # 包含坐标信息
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    oa, aa, kappa, mat_file = evaluate_classifier(
        classifier_model,  # 现在是一个模型对象，不是元组
        incremental_test_loader,
        args.device,
        'incremental',
        round_idx + 1
    )
    # 8. 收集当前增量轮次的回放数据
    print("📦📦 收集增量阶段回放数据...")
    incremental_replay, incremental_replay_path = collect_replay_samples(
        classifier_model,
        t10,  # 使用包含回放数据的训练集
        new_classes,  # 只收集新增类别的样本
        args.replay_samples_per_class,
        args,
        stage=f"incremental_round{round_idx}"
    )

    # 9. 合并到最佳回放数据集
    if best_replay_dataset is not None and incremental_replay is not None:
        # 解包张量
        best_x1, best_x2, best_labels = best_replay_dataset.tensors
        inc_x1, inc_x2, inc_labels = incremental_replay.tensors

        # 合并数据
        combined_x1 = torch.cat([best_x1, inc_x1], dim=0)
        combined_x2 = torch.cat([best_x2, inc_x2], dim=0)
        combined_labels = torch.cat([best_labels, inc_labels], dim=0)

        best_replay_dataset = TensorDataset(combined_x1, combined_x2, combined_labels)
        print(f"✅ 合并后回放数据集大小: {len(best_replay_dataset)} (新增 {len(inc_labels)} 个样本)")
    elif incremental_replay is not None:
        best_replay_dataset = incremental_replay
        print(f"✅ 初始化回放数据集大小: {len(best_replay_dataset)}")

    print(f"✅ 增量阶段第{round_idx + 1}轮完成")
    return vqdit_model, classifier_model, t_incremental, best_replay_dataset

