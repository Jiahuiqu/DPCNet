import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import scipy.io as sio
import torchvision.transforms as transforms


class HyperSpectralDataset(Dataset):
    """高光谱分类数据集加载器"""

    def __init__(self, data_name, mode='train', transform=None,
                 test_size=0.2, random_state=42, patch_size=9,
                 normalize=True, method='l2', num_train_samples_per_class=None,
                 classes=None):  # 添加classes参数
        """
        Args:
            data_name (str): 数据集名称或路径（需适配 sio_load 方法）
            mode (str): 'train', 'val' 或 'test'
            transform (callable): 数据增强操作
            test_size (float): 验证集占比（仅用于非少样本模式）
            random_state (int): 随机种子
            patch_size (int): 提取的空间邻域大小（奇数）
            normalize (bool): 是否进行波段标准化
            method (str): 归一化方式，'l2' 或 'minmax'
            num_train_samples_per_class (int or None): 少样本训练时每类样本数
            classes (list, optional): 指定要加载的类别列表
        """
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.num_train_samples_per_class = num_train_samples_per_class
        self.classes = classes  # 存储指定的类别

        # 加载原始数据
        data, label = self.sio_load(data_name)

        data_norm = self.normalize(data, method=method)

        # 数据预处理
        data_norm = np.transpose(data_norm, (2, 0, 1))  # [C, H, W]

        self.data_norm = torch.from_numpy(data_norm.astype(np.float32))
        self.labels = torch.from_numpy(label.astype(np.int64))

        # 构建 padded_data
        pad = self.half_patch
        self.padded_data = torch.nn.functional.pad(
            self.data_norm,
            (pad, pad, pad, pad),
            mode='reflect'
        )

        # 生成样本索引
        self.indices = self._generate_indices(mode, test_size, random_state, num_train_samples_per_class)


    def sio_load(self, data_name):
        if data_name == 'houston13':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/HSI_data.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/All_Label.mat')
            data = data_mat['HSI_data']
            label = label_mat['All_Label']
        elif data_name == 'paviaU':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/paviaU.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/paviaU_gt.mat')
            data = data_mat['paviaU']
            label = label_mat['paviaU_gt']
        elif data_name == 'Salinas':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/Salinas_corrected.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/Salinas_gt.mat')
            data = data_mat['salinas_corrected']
            label = label_mat['salinas_gt']

        elif data_name == 'Indian':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/indian/indian/Indian_pines.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/indian/indian/Indian_pines_gt.mat')
            data = data_mat['indian_pines']
            label = label_mat['indian_pines_gt']

        elif data_name == 'WHU_Hi_HanChuan':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat')
            data = data_mat['WHU_Hi_HanChuan']
            label = label_mat['WHU_Hi_HanChuan_gt']

        elif data_name == 'Botswana':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/Botswana/Botswana.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/Botswana/Botswana_gt.mat')
            data = data_mat['Botswana']
            label = label_mat['Botswana_gt']

        elif data_name == 'Muufl':
            data_mat = sio.loadmat('/home/amax/ZYS/数据集/muufl/HSI_data.mat')
            label_mat = sio.loadmat('/home/amax/ZYS/数据集/muufl/All_Label.mat')
            data = data_mat['HSI_data']
            label = label_mat['All_Label']


        return data.astype(np.float32), label.astype(np.int64)

    def normalize(self, data, method='l2'):
        """
        根据指定方法对高光谱数据进行归一化

        Args:
            data (np.ndarray): 输入的高光谱数据，形状为 [C, H, W]
            method (str): 归一化方式，可选 'l2' 或 'minmax'

        Returns:
            np.ndarray: 归一化后的数据
        """
        if method == 'l2':
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
            return (data - mean) / (std + 1e-8)
        elif method == 'minmax':
            # min_val = np.min(data, axis=(1, 2), keepdims=True)
            # max_val = np.max(data, axis=(1, 2), keepdims=True)
            # return (data - min_val) / (max_val - min_val + 1e-8)
            min_val = np.min(data)
            max_val = np.max(data)
            # 将数组归一化到 [0, 1] 范围内
            normalized_data = (data - min_val) / (max_val - min_val)
            return normalized_data
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def _generate_indices(self, mode='train', test_size=0.2, random_state=42, num_train_samples_per_class=None):
        y, x = torch.where(self.labels != 0)
        all_indices = torch.stack([y, x], dim=1)
        labels_flat = self.labels[y, x].numpy()

        # 如果指定了classes，只保留这些类别的样本
        if self.classes is not None:
            class_mask = np.isin(labels_flat, self.classes)
            all_indices = all_indices[class_mask]
            labels_flat = labels_flat[class_mask]

        if mode in ['train', 'val'] and num_train_samples_per_class is not None:
            # 少样本模式：按类别固定采样
            unique_classes = np.unique(labels_flat)
            train_selected = []
            val_selected = []

            for cls in unique_classes:
                cls_mask = (labels_flat == cls)
                cls_indices = all_indices[cls_mask]

                if len(cls_indices) < num_train_samples_per_class:
                    raise ValueError(f"类别 {cls} 的样本数不足 {num_train_samples_per_class}")

                np.random.seed(random_state)
                selected_idx = np.random.choice(
                    len(cls_indices),
                    size=num_train_samples_per_class,
                    replace=False
                )
                train_selected.append(cls_indices[selected_idx])
                # val_selected.append(cls_indices[~np.isin(np.arange(len(cls_indices)), selected_idx)])
                val_selected.append(cls_indices)

            train_idx = torch.cat(train_selected)
            val_idx = torch.cat(val_selected)
            return train_idx if mode == 'train' else val_idx

        elif mode in ['train', 'val']:
            # 默认模式：按比例划分
            train_idx, val_idx = train_test_split(
                all_indices,
                test_size=test_size,
                random_state=random_state,
                stratify=labels_flat
            )
            return train_idx if mode == 'train' else val_idx

        else:
            # 测试模式：返回所有有标签样本
            return all_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        y, x = self.indices[idx]

        # 提取 patch
        patch = self.padded_data[
                :,
                y:y + 2 * self.half_patch + 1,
                x:x + 2 * self.half_patch + 1
                ]

        label = self.labels[y, x] - 1  # 类别从0开始

        if self.transform:
            patch = self.transform(patch)

        return patch, label, y.item(), x.item()  # 返回 patch、label、y、x

    @staticmethod
    def get_transforms(mode):
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
            ])
        return None



    @property
    def num_classes(self):
        return int(torch.max(self.labels).item())

    @property
    def full_image(self):
        """返回完整的高光谱图像 (C, H, W)，用于 EnhancedHybridMatcher"""
        print("full_image 类型:", type(self.data_norm))
        print("full_image 形状:", self.data_norm.shape)
        return self.data_norm

    @property
    def pad_image(self):
        """
        返回完整的高光谱图像 (H, W, C)，用于 EnhancedHybridMatcher。
        如果 normalize=True，则返回归一化后的 padded image；
        否则返回原始的 padded image。
        """
        print("full_image 类型:", type(self.padded_data))
        print("full_image 形状:", self.padded_data.shape)
        return self.padded_data

        # # 检查是否已经做过归一化处理
        # if not hasattr(self, '_normalized_padded_data'):
        #     data_tensor = self.padded_data  # (C, H+2*pad, W+2*pad)
        #
        #     if self.normalize:
        #         method = getattr(self, 'method', 'minmax')  # 获取归一化方式，默认 l2
        #
        #         # 使用适用于 Tensor 的归一化函数
        #         if method == 'l2':
        #             mean = data_tensor.mean(dim=(1, 2), keepdim=True)
        #             std = data_tensor.std(dim=(1, 2), keepdim=True)
        #             normalized_data = (data_tensor - mean) / (std + 1e-8)
        #         elif method == 'minmax':
        #             min_val = data_tensor.amin(dim=(1, 2), keepdim=True)
        #             max_val = data_tensor.amax(dim=(1, 2), keepdim=True)
        #             normalized_data = (data_tensor - min_val) / (max_val + 1e-8 - min_val)
        #         else:
        #             raise ValueError(f"Unsupported normalization method: {method}")
        #
        #     self._normalized_padded_data = normalized_data.float()
        #
        # print("pad_image 类型:", type(self._normalized_padded_data))
        # print("pad_image 形状:", self._normalized_padded_data.shape)
        # return self._normalized_padded_data

    def print_info(self):
        """打印当前数据集信息"""
        print(f"Mode: {self.mode}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Normalize: {'Yes' if hasattr(self, 'data') else 'No'}")
        print(f"Total Samples: {len(self)}")
        print(f"Number of Classes: {self.num_classes}")
