import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from dataset import HyperSpectralDataset
from Image_search import EnhancedHybridMatcher
from Class_Increment import DualPromptNetwork, print_model_params

import time
from tqdm import tqdm
from generate_train import train_vqdit_hyperspectral, fine_tune_vqdit_hyperspectral
import random
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import scipy.io as sio

def generate_t1_dataset(full_loader, image, image_pad, device, half, local_radius=40):
    """
    优化版：基于类别原型的全局匹配 + 局部/全局随机选择
    """
    class_coords = {}  # {class_id: [(y1, x1), (y2, x2), ...]}
    all_targets = []
    all_lab = []

    # 第一次遍历：收集所有点的位置和类别
    for batch_idx, (_, labels, ys, xs) in enumerate(full_loader):
        y, x = ys[0].item(), xs[0].item()
        label = labels[0].item()
        all_targets.append((y, x, label))
        all_lab.append(label)

        if label not in class_coords:
            class_coords[label] = []
        class_coords[label].append((y, x))

    # 2. 为每个类别计算原型（平均光谱）
    prototypes = {}
    for class_id, coords in class_coords.items():
        spectra = []
        for (y, x) in coords:
            # 修复：使用正确的索引方式 [C, H, W]
            if isinstance(image, np.ndarray):
                # 如果是numpy数组，形状应该是 [H, W, C]，需要转置
                if image.shape[0] != 204:  # 假设204是通道数
                    # 如果第一个维度不是通道，需要调整
                    spectrum = torch.from_numpy(image[y, x, :]).float()
                else:
                    # 如果已经是 [C, H, W] 格式
                    spectrum = torch.from_numpy(image[:, y, x]).float()
            else:
                # 如果是tensor，形状是 [C, H, W]
                spectrum = image[:, y, x].float()  # 正确索引：[所有通道, y坐标, x坐标]
            spectra.append(spectrum)
        stacked_spectra = torch.stack(spectra)
        prototypes[class_id] = torch.mean(stacked_spectra, dim=0)

    # 3. 为每个类别进行全局匹配
    class_matches = {}
    matcher = EnhancedHybridMatcher(image, (0, 0), device=device, normalize=True)

    print("🚀🚀 执行全局匹配...")
    for class_id, prototype in tqdm(prototypes.items(), desc="🌐🌐 类别全局匹配"):
        # 确保原型在正确的设备上
        matcher.set_target_spectrum(prototype.to(device))

        # 执行全局匹配
        results = matcher.global_superpixel_search(
            topk_per_superpixel=50,
            global_topk=1000,
            min_dist=3,
            spectral_threshold=0.5,
            context_threshold=0.6
        )
        class_matches[class_id] = results.cpu().numpy()

    # 4. 为每个点生成匹配
    patches_x1 = []
    patches_x2 = []
    labels_list = []
    ys_list = []
    xs_list = []

    pbar = tqdm(total=len(all_targets), desc="⚡⚡ 生成匹配点")

    for y, x, label in all_targets:
        # 随机选择匹配模式
        use_local = np.random.rand() < 0.5

        if use_local:
            # 局部匹配
            candidates = []
            for cy, cx in class_coords[label]:
                dist = np.sqrt((cy - y) ** 2 + (cx - x) ** 2)
                if dist <= local_radius and (cy != y or cx != x):
                    candidates.append((cy, cx))

            if candidates:
                ny, nx = candidates[np.random.randint(0, len(candidates))]
            else:
                use_local = False
        else:
            # 全局匹配
            candidates = class_matches[label]
            if len(candidates) > 0:
                ny, nx = candidates[np.random.randint(0, len(candidates))]
            else:
                use_local = True

        # 回退逻辑
        if not use_local:
            candidates = []
            for cy, cx in class_coords[label]:
                dist = np.sqrt((cy - y) ** 2 + (cx - x) ** 2)
                if dist <= local_radius and (cy != y or cx != x):
                    candidates.append((cy, cx))
            if candidates:
                ny, nx = candidates[np.random.randint(0, len(candidates))]
            else:
                ny, nx = y, x

        # 提取patch - 修复索引方式
        top = ny
        bottom = ny + half * 2 + 1
        left = nx
        right = nx + half * 2 + 1

        # 检查边界
        H, W = image_pad.shape[1], image_pad.shape[2]  # [C, H, W] 格式
        if (0 <= y < H and 0 <= x < W and
            0 <= top < H and 0 <= left < W and
            y + half*2 + 1 <= H and x + half*2 + 1 <= W and
            top + half*2 + 1 <= H and left + half*2 + 1 <= W):

            # 提取原始点patch
            x1 = image_pad[:, y:y + half*2 + 1, x:x + half*2 + 1].float().unsqueeze(0)
            # 提取匹配点patch
            x2 = image_pad[:, top:top + half*2 + 1, left:left + half*2 + 1].float().unsqueeze(0)

            patches_x1.append(x1)
            patches_x2.append(x2)
            labels_list.append(torch.tensor([label]))
            ys_list.append(torch.tensor([y]))
            xs_list.append(torch.tensor([x]))

        pbar.update(1)

    pbar.close()

    # 合并张量
    if patches_x1:
        x1s = torch.cat(patches_x1, dim=0)
        x2s = torch.cat(patches_x2, dim=0)
        labels = torch.stack(labels_list)
        ys = torch.stack(ys_list)
        xs = torch.stack(xs_list)
    else:
        x1s = torch.tensor([])
        x2s = torch.tensor([])
        labels = torch.tensor([])
        ys = torch.tensor([])
        xs = torch.tensor([])

    return TensorDataset(x1s, x2s, labels, ys, xs)


# -----------------------------
# 数据集准备函数
# -----------------------------

def prepare_dataset(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """准备阶段：构建基础数据集t1"""
    print("🚀🚀 准备阶段：构建基础数据集t1")

    # 创建完整数据集对象 (始终需要)
    full_dataset = HyperSpectralDataset(
        data_name=args.data_name,
        mode='full',
        patch_size=args.patch_size,
        normalize=args.normalize,
        method=args.norm_method
    )

    # 缓存路径
    cache_dir = 'datasets/cache'
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, f't1_{args.data_name}_patch{args.patch_size}_lr{args.local_radius}.pth')
    print(f"💾💾 数据集缓存路径: {save_path}")

    # 检查缓存
    if os.path.exists(save_path):
        print(f"💾💾 加载已有的 t1 数据集: {save_path}")
        t1 = torch.load(save_path)
        return t1, full_dataset

    # 创建数据加载器
    full_loader = DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 生成增强数据集t1
    print("🛠🛠🛠️ 生成t1数据集（使用固定优化参数）...")
    t1 = generate_t1_dataset(
        full_loader=full_loader,
        image=full_dataset.full_image,
        image_pad=full_dataset.pad_image,
        device=args.device,
        half=args.patch_size // 2,
        local_radius=args.local_radius
    )

    print(f"✅ 准备阶段完成：t1数据集大小 {len(t1)}")
    torch.save(t1, save_path)
    print(f"💾💾 数据集已缓存至: {save_path}")

    return t1, full_dataset


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


def evaluate_classifier(model, test_loader, device, stage, round_idx=None, initial_class_acc=None):
    """
    评估分类器并保存分类结果（保留二维结构），集成正确的APD计算
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
    class_accuracies = {}
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_mask = np.array(all_true) == cls
        # 计算当前类别的准确率
        if np.sum(cls_mask) > 0:  # 确保有样本
            cls_true = np.array(all_true)[cls_mask]
            cls_pred = np.array(all_pred)[cls_mask]
            cls_acc = accuracy_score(cls_true, cls_pred)
            class_accuracies[cls] = cls_acc
        else:
            class_accuracies[cls] = 0.0  # 如果没有样本，准确率设为0

    aa = np.mean(list(class_accuracies.values()))
    kappa = cohen_kappa_score(all_true, all_pred)

    # 新增：计算APD
    apd_score = 0.0
    if initial_class_acc is not None:
        # 计算每个类别的准确率下降
        accuracy_drops = []
        for cls in unique_classes:
            if cls in initial_class_acc:
                # 当前类别的准确率下降
                drop = initial_class_acc[cls] - class_accuracies.get(cls, 0.0)
                accuracy_drops.append(drop)

        # 计算平均准确率下降
        if accuracy_drops:
            apd_score = np.mean(accuracy_drops)
        print(f"✅ APD计算成功: {apd_score:.4f}")
    else:
        print("⚠️ 无初始准确率数据，跳过APD计算")

    print(f"📊📊📊📊 评估结果 ({stage}阶段{' 轮次' + str(round_idx) if round_idx is not None else ''}):")
    print(f"  整体准确率 (OA): {oa:.4f}")
    print(f"  平均准确率 (AA): {aa:.4f}")
    print(f"  Kappa系数: {kappa:.4f}")
    print(f"  APD分数: {apd_score:.4f}")  # 新增APD输出

    # 打印每个类别的准确率
    print("  类别准确率:")
    for cls in sorted(class_accuracies.keys()):
        print(f"    类别 {cls}: {class_accuracies[cls]:.4f}")

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
        'kappa': kappa,
        'apd': apd_score,  # 新增APD保存
        'class_accuracies': class_accuracies  # 保存当前阶段的类别准确率
    })
    print(f"✅ 分类结果已保存为MAT文件: {filename} (包含二维映射和APD指标)")

    # 返回当前阶段的类别准确率，供后续计算APD使用
    return oa, aa, kappa, apd_score, class_accuracies, filename

def prepare_classification_testset(t1, classes, vqdit_model, replace_ratio, args, latent_meanflow=None):
    """
    准备分类测试集t4 - 增加缓存支持并加入 sample_number 和 replace_ratio 到文件名
    """
    # 检查是否使用缓存
    if args.use_cached_t4:
        # 构造带 sample_number 和 replace_ratio 的缓存路径
        cache_dir = 'datasets/cache'
        os.makedirs(cache_dir, exist_ok=True)

        save_path = os.path.join(
            cache_dir,
            f't4_{args.data_name}_patch{args.patch_size}'
            f'base_classes_{args.base_classes}'
            f'_rr{replace_ratio:.2f}.pth'
        )

        if os.path.exists(save_path):
            print(f"💾 加载缓存的 t4 数据集: {save_path}")
            return torch.load(save_path, map_location='cpu')
        else:
            print(f"⚠️ 缓存文件 {save_path} 不存在，将重新生成 t4 数据集")

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
    # 添加坐标收集
    ys_list = []
    xs_list = []

    for idx in indices:
        image, patch, label, y, x = t1_cpu[idx]
        ys_list.append(y)
        xs_list.append(x)

        if torch.rand(1).item() < replace_ratio:
            with torch.no_grad():
                generated_patch = latent_meanflow.sample_each_class(
                    model=vqdit_model,
                    n_per_class=1,
                    classes=[label.item()],
                    device=device
                )
                generated_patch = vqdit_model.vqvae.decode_continuous(generated_patch)
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
    labels_tensor = torch.tensor(labels, device=args.device)
    ys_tensor = torch.tensor(ys_list, device=args.device)
    xs_tensor = torch.tensor(xs_list, device=args.device)

    t4_dataset = TensorDataset(images_tensor, patches_tensor, labels_tensor, ys_tensor, xs_tensor)

    # 保存到缓存路径
    if hasattr(args, 'use_cached_t4') and args.use_cached_t4:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(t4_dataset, save_path)
        print(f"💾 t4数据集已保存至: {save_path}")

    return t4_dataset


def prepare_classification_trainset(base_train_dataset, full_image, pad_image, vqdit_model, args, latent_meanflow):
    """
    准备分类训练集t5，并支持缓存机制。
    修改：patches_x1随机选择同一类别下的随机一张图，而不是固定使用同一个patch_x1
    """
    # 构造缓存路径
    cache_dir = 'datasets/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # 从args中获取三个控制参数
    num_local_samples = args.num_local_samples
    num_global_samples = args.num_global_samples
    num_generated_samples = args.num_generated_samples

    # 构造唯一的缓存文件名，包含关键参数
    save_path = os.path.join(
        cache_dir,
        f't5_{args.data_name}_patch{args.patch_size}'
        f'base_samples_per_class_{args.base_samples_per_class}'
        f'base_classes_{args.base_classes}'
        f'_local{num_local_samples}'
        f'_global{num_global_samples}'
        f'_gen{num_generated_samples}.pth'
    )

    # 如果启用缓存且文件存在，直接加载
    if args.use_cached_t5 and os.path.exists(save_path):
        print(f"💾 加载缓存的 t5 数据集: {save_path}")
        return torch.load(save_path, map_location='cpu')
    elif args.use_cached_t5 and not os.path.exists(save_path):
        print(f"⚠️ 缓存文件 {save_path} 不存在，将重新生成 t5 数据集")

    half = args.patch_size // 2
    device = args.device

    # 将pad_image移动到目标设备（如果不在）
    pad_image = pad_image.to(device)

    # 首先按类别组织base_train_dataset中的样本
    class_to_indices = defaultdict(list)
    class_patches = defaultdict(list)

    print("📊 按类别组织训练样本...")
    for i in range(len(base_train_dataset)):
        patch_x1, label, y, x = base_train_dataset[i]
        label_item = label.item() if torch.is_tensor(label) else label
        class_to_indices[label_item].append(i)
        class_patches[label_item].append(patch_x1.to(device))

    patches_x1 = []
    patches_x2 = []
    labels = []

    # 添加进度条
    total_points = len(base_train_dataset)
    pbar = tqdm(total=total_points, desc="🔧 生成分类训练集t5", unit="点")
    generated_samples_count = 0

    # 遍历t2中的每个训练点
    for i in range(len(base_train_dataset)):
        # 获取当前训练点的信息
        patch_x1, label, y, x = base_train_dataset[i]
        label_item = label.item() if torch.is_tensor(label) else label

        # 移动到目标设备
        patch_x1 = patch_x1.to(device)
        if torch.is_tensor(label):
            label = label.to(device)
        else:
            label = torch.tensor(label, device=device)

        # 获取当前类别的所有patch_x1选项
        same_class_patches = class_patches[label_item]

        # 创建匹配器
        matcher = EnhancedHybridMatcher(full_image, (y, x), device=device, normalize=True)

        # 执行混合搜索（局部和全局）
        results = matcher.hybrid_selection(
            local_radius=args.local_radius,
            local_topk=num_local_samples,
            superpixel_topk=2,
            global_topk=num_global_samples,
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

                # 随机选择同一类别的patch_x1
                # random_patch_x1 = random.choice(same_class_patches)
                # patches_x1.append(random_patch_x1)
                patches_x1.append(patch_x1)
                patches_x2.append(patch)
                labels.append(label)
                generated_samples_count += 1

        # 2. 添加全局搜索样本
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

                # 随机选择同一类别的patch_x1
                # random_patch_x1 = random.choice(same_class_patches)
                # patches_x1.append(random_patch_x1)
                patches_x1.append(patch_x1)
                patches_x2.append(patch)
                labels.append(label)
                generated_samples_count += 1

        # 3. 添加生成样本
        for _ in range(num_generated_samples):
            with torch.no_grad():
                generated_z = latent_meanflow.sample_each_class(
                    model=vqdit_model,
                    n_per_class=1,
                    classes=[label_item],
                    device=device
                )
                generated_patch = vqdit_model.vqvae.decode_continuous(generated_z)

            # 随机选择同一类别的patch_x1
            # random_patch_x1 = random.choice(same_class_patches)
            # patches_x1.append(random_patch_x1)
            patches_x1.append(patch_x1)
            patches_x2.append(generated_patch.squeeze(0))
            labels.append(label)
            generated_samples_count += 1

        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            "当前点": f"{i + 1}/{total_points}",
            "已生成样本": generated_samples_count,
            "类别": label_item
        })

    # 关闭进度条
    pbar.close()

    # 处理空列表的特殊情况
    if len(patches_x1) == 0:
        print("⚠️ 警告: 未生成任何样本，返回空数据集")
        return TensorDataset(
            torch.empty(0, pad_image.shape[0], 2 * half + 1, 2 * half + 1),
            torch.empty(0, pad_image.shape[0], 2 * half + 1, 2 * half + 1),
            torch.empty(0, dtype=torch.long)
        )

    # 转换为张量
    patches_x1 = torch.stack(patches_x1)
    patches_x2 = torch.stack(patches_x2)
    labels = torch.tensor([l.item() if torch.is_tensor(l) else l for l in labels], device=device)

    print(f"✅ t5数据集生成完成: 总点数 = {total_points}, 总样本数 = {len(patches_x1)}")

    # 打印类别分布统计
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print("📊 生成数据集的类别分布:")
    for cls, cnt in zip(unique_labels, counts):
        print(f"  类别 {cls.item()}: {cnt.item()} 个样本")

    # 保存到缓存路径
    if hasattr(args, 'use_cached_t5') and args.use_cached_t5:
        t5_dataset = TensorDataset(patches_x1, patches_x2, labels)
        torch.save(t5_dataset, save_path)
        print(f"💾 t5数据集已保存至: {save_path}")

    return TensorDataset(patches_x1.cpu(), patches_x2.cpu(), labels.cpu())



def collect_replay_samples(classifier, dataset, classes, samples_per_class, args, stage="base",
                         save_path=None):
    """
    简化版：使用聚类选择多样性样本
    """
    # 创建固定的数据加载器
    fixed_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 初始化存储结构
    samples_dict = defaultdict(list)
    features_dict = defaultdict(list)

    classifier.eval()
    with torch.no_grad():
        for batch_idx, (x1, x2, labels) in enumerate(fixed_loader):
            x1, x2, labels = x1.to(args.device), x2.to(args.device), labels.to(args.device)

            # 获取特征
            features = classifier.get_features(x1, x2)

            # 获取预测结果
            logits = classifier(x1, x2)
            probs = F.softmax(logits, dim=1)
            confidences, _ = torch.max(probs, dim=1)

            for i in range(x1.size(0)):
                true_label = labels[i].item()

                if true_label in classes:
                    sample_data = {
                        'x1': x1[i].cpu().clone(),
                        'x2': x2[i].cpu().clone(),
                        'label': labels[i].cpu().clone(),
                        'confidence': confidences[i].item(),
                        'feature': features[i].cpu().clone()
                    }
                    samples_dict[true_label].append(sample_data)
                    features_dict[true_label].append(features[i].cpu())

    # 使用聚类选择样本
    replay_x1 = []
    replay_x2 = []
    replay_labels = []

    for cls_id in classes:
        samples = samples_dict.get(cls_id, [])
        if not samples:
            print(f"⚠️ 类别 {cls_id} 无可用样本")
            continue

        # 聚类选择代表性样本
        selected_samples = cluster_selection(samples, features_dict[cls_id], samples_per_class)

        for sample in selected_samples:
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
            f"replay_classes{min_class}to{max_class}_"
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

def cluster_selection(samples, features, num_select):
    """简单的聚类样本选择"""
    if len(samples) <= num_select:
        return samples

    try:
        # 转换为numpy
        features_array = torch.stack(features).numpy()

        # K-means聚类
        from sklearn.cluster import KMeans
        n_clusters = min(num_select, len(features_array))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_array)

        selected_samples = []

        # 从每个簇中选择最接近中心的样本
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id)
            if not any(cluster_mask):
                continue

            cluster_features = features_array[cluster_mask]
            center = kmeans.cluster_centers_[cluster_id]

            # 计算距离并选择最近的
            distances = np.linalg.norm(cluster_features - center, axis=1)
            best_idx = np.argmin(distances)

            # 找到对应的样本
            cluster_samples = [samples[i] for i in range(len(samples)) if cluster_mask[i]]
            selected_samples.append(cluster_samples[best_idx])

        return selected_samples

    except Exception as e:
        print(f"⚠️ 聚类失败: {e}, 使用随机选择")
        return random.sample(samples, min(num_select, len(samples)))

def print_dataset_distribution(dataset, dataset_name):
    class_counts = {}
    labels_tensor = dataset.tensors[2]  # 假设标签在第3个位置
    labels_np = labels_tensor.cpu().numpy()
    unique, counts = np.unique(labels_np, return_counts=True)

    for cls, cnt in zip(unique, counts):
        class_counts[cls] = cnt

    print(f"\n{dataset_name}类别分布:")
    print("{:<10} {:<15}".format('类别ID', '样本数量'))
    print("-" * 25)
    for cls in sorted(class_counts.keys()):
        print("{:<10} {:<15}".format(cls, class_counts[cls]))

    return class_counts

def train_base_classifier(train_dataset, test_dataset, args):
    """训练基础分类模型，支持加载预训练模型"""
    # 创建模型
    classifier = DualPromptNetwork(
        dim=args.in_channels,  # 输入高光谱图像的通道数（即波段数量）
        size=args.patch_size,  # 输入图像块的大小（如9x9像素）
        num_classes=args.base_classes,  # 基础阶段训练的类别总数
        shared_dims=args.shared_dims,  # 共享特征提取层的通道维度列表
        private_dims=args.private_dims,  # 私有提示向量的维度列表
        prompt_dim=args.prompt_dim,  # 提示向量的维度
        device=args.device,  # 使用的设备（CPU/GPU）
        contrastive_weight=args.contrastive_weight,

        # 新增原型稳定性参数
        proto_warmup_epochs=args.proto_warmup_epochs,  # 原型更新预热期，前N个epoch不更新原型
        proto_quality_threshold=args.proto_quality_threshold,  # 原型更新的特征质量阈值（0-1）
        # proto_decay_start=args.proto_decay_start,  # 原型更新的初始衰减系数
        # proto_decay_end=args.proto_decay_end,  # 原型更新的最终衰减系数
        # proto_decay_steps=args.proto_decay_steps  # 从初始衰减到最终衰减所需的步数（epoch数）
    ).to(args.device)

    # 检查是否有预训练模型
    if hasattr(args, 'classifier_pretrained_path') and args.classifier_pretrained_path:
        if os.path.exists(args.classifier_pretrained_path):
            print(f"🔍 尝试加载预训练分类模型: {args.classifier_pretrained_path}")
            try:
                # 加载模型状态
                state_dict = torch.load(args.classifier_pretrained_path, map_location=args.device)

                # 严格校验：参数键和形状必须与当前模型完全一致才允许跳过训练，
                # 防止旧架构/其他配置的权重被"部分匹配"加载后混合随机初始化参数，导致模型退化
                model_state = classifier.state_dict()
                shape_mismatch = [k for k in model_state
                                  if k not in state_dict or state_dict[k].shape != model_state[k].shape]
                if shape_mismatch:
                    print(f"⚠️ 权重与当前模型结构不匹配（{len(shape_mismatch)}个参数不一致），从头训练")
                else:
                    classifier.load_state_dict(state_dict)
                    print(f"✅ 成功加载分类模型（{len(state_dict)} 参数完全匹配）")
                    print("⏭ 跳过训练阶段，直接返回模型")
                    return classifier, None
            except Exception as e:
                print(f"⚠️ 加载分类模型失败: {str(e)}")
                print("🆕 将从头开始训练分类模型")
        else:
            print(f"⚠️ 预训练模型文件不存在: {args.classifier_pretrained_path}")
            print("🆕 将从头开始训练分类模型")

    # 准备数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"训练样本总数: {len(train_dataset)}")
    print(f"测试样本总数: {len(test_dataset)}")

    train_class_dist = print_dataset_distribution(train_dataset, "训练集")
    test_class_dist = print_dataset_distribution(test_dataset, "测试集")

    # 训练模型
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    best_acc = 0.0
    best_model_state = None  # 保存最佳模型状态

    print_model_params(classifier)

    # 初始化回放数据集
    replay_dataset = None
    best_replay_dataset = None

    for epoch in range(args.classifier_epochs):
        # classifier.update_prototype_epoch(epoch)
        classifier.train()
        total_loss = 0.0
        loss_details = {"cls": 0.0, "distill": 0.0, "sim": 0.0}
        train_correct = 0
        train_total = 0

        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(args.device), x2.to(args.device), labels.to(args.device).squeeze()

            optimizer.zero_grad()
            logits, loss, losses = classifier(x1, x2, labels)
            loss.backward()

            # 添加梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            for k in loss_details:
                loss_details[k] += losses[k]

            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        print(f"Epoch {epoch + 1}/{args.classifier_epochs}: "
              f"Acc={train_accuracy:.4f}")

        # 计算平均损失
        num_batches = len(train_loader)
        avg_total_loss = total_loss / num_batches
        avg_cls_loss = loss_details["cls"] / num_batches
        avg_distill_loss = loss_details["distill"] / num_batches
        avg_sim_loss = loss_details["sim"] / num_batches

        # # 更新原型内存
        # classifier.prompt_module.memory.update_epoch(epoch + 1)
        # classifier.prompt_module.memory.reset_quality_log()

        if (epoch + 1) % 10 == 0:
            # 在测试集上评估
            classifier.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, patches, labels, _, _ in test_loader:
                    images = images.to(args.device)
                    patches = patches.to(args.device)
                    labels = labels.to(args.device)
                    labels = labels.view(-1)  # 确保是1维张量

                    logits = classifier(images, patches)
                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total

            # 打印详细损失信息
            print(f"Epoch {epoch + 1}/{args.classifier_epochs}: "
                  f"Total Loss={avg_total_loss:.4f}, "
                  f"Cls Loss={avg_cls_loss:.4f}, "
                  f"Distill Loss={avg_distill_loss:.4f}, "
                  f"Sim Loss={avg_sim_loss:.4f}, "
                  f"Acc={accuracy:.4f}")

            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                best_model_path = f"checkpoints/base_classifier_{timestamp}_{accuracy:.4f}.pth"
                # 直接保存到文件，不保存引用
                torch.save(classifier.state_dict(), best_model_path)

                # 使用固定顺序收集样本
                replay_dataset = collect_replay_samples(
                    classifier=classifier,
                    dataset=train_dataset,  # 传入原始数据集
                    classes=list(range(args.base_classes)),
                    samples_per_class=args.replay_samples_per_class,
                    args=args
                )
                best_replay_dataset = replay_dataset

    # 训练结束后加载最佳模型状态
    if best_model_path and os.path.exists(best_model_path):
        print(f"🔁 从文件加载最佳模型: {best_model_path}")
        classifier.load_state_dict(torch.load(best_model_path))
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, patches, labels, _, _ in test_loader:
                images = images.to(args.device)
                patches = patches.to(args.device)
                labels = labels.to(args.device)
                labels = labels.view(-1)  # 确保是1维张量

                logits = classifier(images, patches)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        # 打印详细损失信息
        print(f"Acc={accuracy:.4f}")
        print(f"🔁 加载训练过程中的最佳模型 (准确率: {best_acc:.4f})")

    return classifier, best_replay_dataset


# -----------------------------
# 第一阶段训练函数
# -----------------------------

def base_training(args, t1, full_dataset):
    """第一阶段：训练基础模型"""
    print("🚀🚀 第一阶段：训练基础模型")

    # 缓存目录
    cache_dir = 'datasets/cache'
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 准备基础训练数据t2 特别注意classes要+1！！！
    base_classes = list(range(0, args.base_classes))
    base_train_dataset = HyperSpectralDataset(
        data_name=args.data_name,
        mode='train',
        classes=list(range(1, args.base_classes+1)),
        num_train_samples_per_class=args.base_samples_per_class,
        patch_size=args.patch_size,
        normalize=args.normalize,
        method=args.norm_method
    )

    # 创建数据加载器generator = {NoneType} None
    base_train_loader = DataLoader(
        base_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 为t2生成唯一缓存文件名
    t2_cache_name = (
        f"t2_{args.data_name}_"
        f"patch{args.patch_size}_"
        f"patch{args.base_classes}_"
        f"base_samples{args.base_samples_per_class}_"
        f"local{args.num_samples_local_class_t2}_"
        f"super{args.superpixel_topk_t2}_"
        f"global{args.num_samples_global_class_t2}.pth"
    )
    t2_cache_path = os.path.join(cache_dir, t2_cache_name)

    # 检查t2缓存
    if os.path.exists(t2_cache_path):
        print(f"💾 加载缓存的t2数据集: {t2_cache_path}")
        t2 = torch.load(t2_cache_path)
    else:
        print("🛠 生成t2数据集...")
        # 生成增强训练集t2
        t2 = generate_patches_with_types(
            train_loader=base_train_loader,
            image=full_dataset.full_image,
            image_pad=full_dataset.pad_image,
            device=args.device,
            half=args.patch_size // 2,
            local_radius=args.local_radius,
            local_topk=args.num_samples_local_class_t2,
            superpixel_topk=args.superpixel_topk_t2,
            global_topk=args.num_samples_global_class_t2,
            min_dist=args.min_dist,
            spectral_threshold=args.spectral_threshold,
            context_threshold=args.context_threshold
        )
        torch.save(t2, t2_cache_path)
        print(f"💾 保存t2数据集到: {t2_cache_path}")

    # 2. 准备测试数据t3（仅基础类别的x1）
    t3_indices = [i for i, (_, _, label, _, _) in enumerate(t1) if label.item() in base_classes]
    t3 = TensorDataset(
        t1.tensors[0][t3_indices],  # patches
        t1.tensors[2][t3_indices].squeeze(),  # labels
        t1.tensors[3][t3_indices].squeeze(),  # ys
        t1.tensors[4][t3_indices].squeeze()  # xs
    )
    print(len(t3))

    base_train_t2_loader = DataLoader(
        t2,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    base_test_t3_loader = DataLoader(
        t3,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 3. 训练VQDiT模型
    # save_path_t4 = os.path.join(
    #     cache_dir,
    #     f't4_{args.data_name}_patch{args.patch_size}'
    #     f'base_classes_{args.base_classes}'
    #     f'_rr{args.replace_ratio:.2f}.pth'
    # )
    #
    # save_path_t5 = os.path.join(
    #     cache_dir,
    #     f't5_{args.data_name}_patch{args.patch_size}'
    #     f'base_samples_per_class_{args.base_samples_per_class}'
    #     f'base_classes_{args.base_classes}'
    #     f'_local{args.num_local_samples}'
    #     f'_global{args.num_global_samples}'
    #     f'_gen{args.num_generated_samples}.pth'
    # )
    #
    #
    # if os.path.exists(save_path_t4) and args.use_cached_t5 and os.path.exists(save_path_t5):
    #     print("🔧 跳过训练VQDiT模型...")
    #
    # else :
    #     print("🔧 训练基础VQDiT模型...")

    vqdit_model, latent_meanflow = train_vqdit_hyperspectral(base_train_t2_loader, base_test_t3_loader, args, args.base_classes)


    # 4. 准备分类测试集t4
    print("📊 准备分类测试集t4...")
    t4 = prepare_classification_testset(t1, base_classes, vqdit_model, args.replace_ratio, args, latent_meanflow)

    # 5. 准备分类训练集t5
    print("📚 准备分类训练集t5...")
    t5 = prepare_classification_trainset(
        base_train_dataset=base_train_dataset,
        full_image=full_dataset.full_image,
        pad_image=full_dataset.pad_image,
        vqdit_model=vqdit_model,
        args=args,
        latent_meanflow=latent_meanflow  # 添加这个参数
    )

    # 6. 训练分类模型
    print("🔧🔧 训练基础分类模型...")
    classifier_model, best_replay_dataset = train_base_classifier(t5, t4, args)

    # 7. 收集基础阶段回放数据
    print("📦📦 收集基础阶段回放数据...")
    base_classes = list(range(args.base_classes))
    base_replay_dataset, base_replay_path = collect_replay_samples(
        classifier_model,
        t5,  # 使用分类训练集
        base_classes,
        args.replay_samples_per_class,
        args,
        stage="base"
    )

    # 7. 评估基础分类模型
    print("📊📊 评估基础分类模型...")
    base_test_loader = DataLoader(
        t4,  # 现在t4包含坐标信息
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # 基础阶段评估
    oa, aa, kappa, apd, base_class_acc, filename = evaluate_classifier(
        classifier_model,
        base_test_loader,
        args.device,
        'base'
    )

    # 保存基础阶段的类别准确率
    initial_class_accuracies = base_class_acc
    print("✅ 第一阶段完成")
    # 训练完成后释放内存
    del full_dataset  # 释放全量数据集
    del base_train_dataset  # 释放基础训练数据集
    torch.cuda.empty_cache()  # 清理GPU缓存

    return vqdit_model, classifier_model, t4, base_replay_dataset, base_replay_path, initial_class_accuracies
