import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from torch.nn.functional import conv2d
import torch.nn.functional as F

FUSION_ALPHA = 0.7

class EnhancedHybridMatcher:
    # 类级别常量：颜色配置
    COLOR_BACKGROUND = [0.5, 0.5, 0.5]
    COLOR_TARGET_POINT = [0, 0, 1]  # 蓝色
    COLOR_SAME_CLASS = [0, 0, 1]
    COLOR_DIFF_CLASS = [1, 0, 0]
    COLOR_UNKNOWN = [0, 0, 0]
    COLOR_OTHER = [0.8, 0.8, 0.8]

    def __init__(self, image, target_loc, device='cuda', normalize=False,
                 similarity_method='sam', low_pass_kernel_size=5):
        """
        Args:
            image: 预归一化高光谱图像 [C, H, W]
            target_loc: 目标中心坐标 (y, x)
            device: 计算设备 ('cuda' 或 'cpu')
            normalize: 是否对输入图像执行全局归一化 (bool)
            similarity_method: 相似度计算方法 ('cosine', 'sam', 'mahalanobis', 'sid')
            low_pass_kernel_size: 低通滤波核大小 (奇数)
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        assert len(image.shape) == 3, "Input image must be a 3D tensor/array with shape [C, H, W]"
        assert 0 <= target_loc[0] < image.shape[1] and 0 <= target_loc[1] < image.shape[2], \
            "Target location out of bounds"

        self.device = device
        self.image = image.float().to(device)
        self.similarity_method = similarity_method
        self.low_pass_kernel_size = low_pass_kernel_size
        self.C, self.H, self.W = image.shape  # [C, H, W]

        # if normalize:
        #     self.image = self._normalize_image(self.image)

        # 应用低通滤波增强低频类别信息
        self.low_pass_image = self._apply_low_pass_filter(self.image, kernel_size=low_pass_kernel_size)

        # 使用低频图像计算目标光谱 [C]
        self.target = self.low_pass_image[:, target_loc[0], target_loc[1]].clone().float()
        self.raw_target = self.image[:, target_loc[0], target_loc[1]].clone().float()  # 保留原始目标光谱

        self.target_loc = (int(target_loc[0]), int(target_loc[1]))  # 确保坐标为整数

        # 计算超像素分割
        self.superpixels = self._compute_superpixels()

        # 马氏距离预计算
        if similarity_method == 'mahalanobis':
            self.inv_cov = self._compute_inverse_covariance()

    def _normalize_image(self, img):
        # 对每个像素的光谱进行归一化 [C, H, W]
        norm = torch.norm(img, p=2, dim=0, keepdim=True)  # [1, H, W]
        return img / (norm + 1e-8)

    def _apply_low_pass_filter(self, img, kernel_size=5):
        """应用低通滤波 [C, H, W]"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size * kernel_size)
        filtered_img = torch.zeros_like(img)
        for c in range(self.C):
            channel = img[c, :, :].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            filtered = conv2d(channel, kernel, padding=kernel_size // 2)
            filtered_img[c, :, :] = filtered.squeeze()
        return filtered_img

    def _compute_superpixels(self, n_segments=100, compactness=10.0):
        """计算超像素分割 [C, H, W] -> [H, W]"""
        # 将图像转换为 [H, W, C] 格式进行PCA
        img_np = self.image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        img_flat = img_np.reshape(-1, self.C)
        mean = np.mean(img_flat, axis=0)
        centered = img_flat - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:3]
        pca_vecs = eigvecs[:, idx]
        img_pca = centered @ pca_vecs
        img_pca = img_pca.reshape(self.H, self.W, 3)
        img_pca = (img_pca - img_pca.min()) / (img_pca.max() - img_pca.min())
        superpixels = slic(img_pca, n_segments=n_segments, compactness=compactness)
        return torch.from_numpy(superpixels).to(self.device)

    def _compute_inverse_covariance(self):
        y, x = self.target_loc
        half_size = min(15, self.H // 2, self.W // 2)
        y_min = max(0, y - half_size)
        y_max = min(self.H, y + half_size + 1)
        x_min = max(0, x - half_size)
        x_max = min(self.W, x + half_size + 1)

        # 提取局部区域 [C, H_local, W_local] -> [C, H_local*W_local] -> [H_local*W_local, C]
        local_region = self.low_pass_image[:, y_min:y_max, x_min:x_max]
        local_region_flat = local_region.reshape(self.C, -1).T  # [N, C]

        cov = torch.cov(local_region_flat.T)  # [C, C]
        reg = torch.eye(self.C, device=self.device) * 1e-3
        return torch.inverse(cov + reg)

    def _compute_similarity(self, pixel_spectrum):
        """计算相似度，pixel_spectrum形状为 [N, C] 或 [C]"""
        if pixel_spectrum.dim() == 1:
            pixel_spectrum = pixel_spectrum.unsqueeze(0)

        if self.similarity_method == 'cosine':
            return F.cosine_similarity(self.target.unsqueeze(0), pixel_spectrum, dim=-1)

        elif self.similarity_method == 'sam':
            dot_product = torch.sum(self.target * pixel_spectrum, dim=-1)
            norm_target = torch.norm(self.target, p=2)
            norm_pixel = torch.norm(pixel_spectrum, p=2, dim=-1)
            cos_theta = dot_product / (norm_target * norm_pixel + 1e-8)
            return 1.0 - torch.acos(torch.clamp(cos_theta, -1.0, 1.0)) / np.pi

        elif self.similarity_method == 'mahalanobis':
            diff = pixel_spectrum - self.target
            dist = torch.sqrt(torch.sum(diff @ self.inv_cov * diff, dim=-1))
            return 1.0 / (1.0 + dist)

        elif self.similarity_method == 'sid':
            p = (self.target + 1e-8) / (torch.sum(self.target) + 1e-8)
            q = (pixel_spectrum + 1e-8) / (torch.sum(pixel_spectrum, dim=-1, keepdim=True) + 1e-8)
            kl_pq = torch.sum(p * torch.log(p / q), dim=-1)
            kl_qp = torch.sum(q * torch.log(q / p), dim=-1)
            sid = kl_pq + kl_qp
            return 1.0 / (1.0 + sid)

        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

    def _spectral_texture_similarity(self, std_texture, target_std):
        """
        计算光谱-纹理联合相似度
        Args:
            std_texture: [C, H, W] 各波段纹理标准差
            target_std:  [C] 目标区域各波段标准差
        Returns:
            similarity: [H, W]
        """
        eps = 1e-6
        numerator = (std_texture * target_std.view(-1, 1, 1)).sum(dim=0)  # [H, W]
        denominator = torch.norm(std_texture, dim=0) * torch.norm(target_std) + eps
        spectral_angle = torch.acos(numerator / denominator)  # [H, W]
        return 1.0 / (1.0 + spectral_angle)

    def local_search(self, radius=40, topk=50, use_texture=True):
        y, x = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing='ij'
        )
        y, x = y.float(), x.float()

        spatial_dist = (y - self.target_loc[0]) ** 2 + (x - self.target_loc[1]) ** 2
        spatial_mask = (spatial_dist <= radius ** 2) & \
                       ~((y == self.target_loc[0]) & (x == self.target_loc[1]))

        # 将图像重塑为 [H*W, C] 格式
        pixel_spectra = self.low_pass_image.permute(1, 2, 0).reshape(-1, self.C)  # [H*W, C]
        spectral_sim = self._compute_similarity(pixel_spectra).reshape(self.H, self.W)

        texture_sim = torch.zeros_like(spectral_sim)
        if use_texture:
            window_size = 7
            pad = window_size // 2

            # 图像已经是 [C, H, W] 格式
            image_for_texture = self.low_pass_image.unsqueeze(0)  # [1, C, H, W]
            padded = F.pad(image_for_texture, (pad, pad, pad, pad), mode='reflect')

            # 计算各波段局部窗口标准差 [1, C, H, W, win, win] -> [1, C, H, W]
            unfolded = padded.unfold(2, window_size, 1).unfold(3, window_size, 1)
            std_texture = torch.std(unfolded, dim=(-1, -2))  # [1, C, H, W]
            std_texture = std_texture.squeeze(0)  # [C, H, W]

            # 获取目标窗口纹理特征 [C]
            target_window = unfolded[
                0, :,  # 批次维度为0
                self.target_loc[0] + pad - pad,  # 调整到unfolded坐标系
                self.target_loc[1] + pad - pad
            ]  # [C, win, win]
            target_std = torch.std(target_window.reshape(self.C, -1), dim=1)  # [C]

            # 计算光谱-纹理联合相似度
            texture_sim = self._spectral_texture_similarity(std_texture, target_std)

        alpha = FUSION_ALPHA if use_texture else 1.0
        beta = (1 - FUSION_ALPHA) if use_texture else 0.0
        combined_score = alpha * spectral_sim + beta * texture_sim
        combined_score -= 0.1 * (spatial_dist / (radius ** 2))

        flat_scores = combined_score.flatten()
        valid_scores = flat_scores[spatial_mask.flatten()]

        _, top_indices = torch.topk(valid_scores, min(topk, len(valid_scores)))
        valid_indices = torch.nonzero(spatial_mask.flatten(), as_tuple=True)[0][top_indices]
        y_coords = (valid_indices // self.W).float()
        x_coords = (valid_indices % self.W).float()

        return torch.stack([y_coords, x_coords], dim=1)

    def global_superpixel_search(self, topk_per_superpixel=5, global_topk=100, min_dist=10,
                                 context_scales=[3, 5, 7], spectral_threshold=0.5,
                                 context_threshold=0.6, min_points_per_sp=0):
        """
        完整的全局搜索方法，保证最终点数不超过global_topk
        """
        sp_indices = torch.unique(self.superpixels)
        candidates = []
        candidates_dict = {}  # 缓存最终得分

        target_contexts = {}
        for scale in context_scales:
            target_contexts[scale] = self._extract_local_context(self.target_loc, scale)

        target_sp = self.superpixels[self.target_loc]

        for sp_idx in sp_indices:
            if sp_idx == target_sp:
                continue

            mask = (self.superpixels == sp_idx)
            num_pixels = mask.sum().item()
            if num_pixels == 0:
                continue

            sp_coords = torch.nonzero(mask)
            # 提取光谱 [N, C]
            sp_spectra = self.low_pass_image.permute(1, 2, 0)[sp_coords[:, 0], sp_coords[:, 1]]
            spectral_sim = self._compute_similarity(sp_spectra)

            valid_mask = spectral_sim >= spectral_threshold
            valid_coords = sp_coords[valid_mask]
            valid_sim = spectral_sim[valid_mask]

            if len(valid_coords) == 0:
                continue

            actual_topk = min(topk_per_superpixel, len(valid_coords))
            _, topk_idx = torch.topk(valid_sim, actual_topk)
            candidate_coords = valid_coords[topk_idx]
            candidate_sims = valid_sim[topk_idx]

            valid_candidates = []
            candidate_scores = []

            for idx, coord in enumerate(candidate_coords):
                coord_tuple = (coord[0].item(), coord[1].item())

                context_sim = 0.0
                for scale in context_scales:
                    candidate_context = self._extract_local_context(coord_tuple, scale)
                    scale_sim = self._context_similarity(candidate_context, target_contexts[scale])
                    context_sim += scale_sim / len(context_scales)

                if context_sim < context_threshold:
                    continue

                spectral_val = candidate_sims[idx].item()
                combined_score = FUSION_ALPHA * spectral_val + (1 - FUSION_ALPHA) * context_sim

                valid_candidates.append(coord_tuple)
                candidate_scores.append(combined_score)

            if valid_candidates:
                if min_points_per_sp == 0:
                    min_keep = min(3, max(1, int(np.sqrt(num_pixels / 100))))
                else:
                    min_keep = min_points_per_sp

                if len(valid_candidates) > min_keep:
                    sorted_indices = np.argsort(candidate_scores)[::-1]
                    keep_count = min(min_keep + 1, len(valid_candidates))
                    valid_candidates = [valid_candidates[i] for i in sorted_indices[:keep_count]]

                candidates.extend(valid_candidates)

        if not candidates:
            return torch.empty(0, 2, device=self.device)

        candidate_tensor = torch.tensor(candidates, device=self.device).float()

        if len(candidate_tensor) == 0:
            return torch.empty(0, 2, device=self.device)

        spatially_filtered = self._spatial_filter(candidate_tensor, min_dist)

        if len(spatially_filtered) <= global_topk:
            final_candidates = spatially_filtered
        else:
            final_scores = []
            max_scale = max(context_scales)
            target_context_max = target_contexts[max_scale]

            for coord in spatially_filtered:
                coord_tuple = (int(coord[0]), int(coord[1]))

                if coord_tuple in candidates_dict:
                    final_scores.append(candidates_dict[coord_tuple])
                    continue

                # 提取光谱 [1, C]
                spectrum = self.low_pass_image[:, coord_tuple[0], coord_tuple[1]].unsqueeze(0)
                spectral_val = self._compute_similarity(spectrum).item()

                candidate_context = self._extract_local_context(coord_tuple, max_scale)
                context_sim = self._context_similarity(candidate_context, target_context_max)

                combined_score = FUSION_ALPHA * spectral_val + (1 - FUSION_ALPHA) * context_sim
                candidates_dict[coord_tuple] = combined_score
                final_scores.append(combined_score)

            _, top_indices = torch.topk(torch.tensor(final_scores, device=self.device), global_topk)
            final_candidates = spatially_filtered[top_indices.tolist()]

        return final_candidates

    def _extract_local_context(self, coord, window_size):
        """提取指定位置的局部上下文特征 [C, H, W]"""
        y, x = coord
        half_win = window_size // 2

        # 计算窗口边界（处理图像边缘）
        y_min = max(0, y - half_win)
        y_max = min(self.H, y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(self.W, x + half_win + 1)

        # 提取局部区域 [C, H_local, W_local]
        local_region = self.low_pass_image[:, y_min:y_max, x_min:x_max]

        # 计算关键特征
        return {
            'mean_spectrum': local_region.mean(dim=(1, 2)),  # [C]
            'std_spectrum': local_region.std(dim=(1, 2)),    # [C]
            'size': (y_max - y_min, x_max - x_min)
        }

    def _context_similarity(self, context1, context2):
        """计算两个上下文的相似度得分"""
        # 1. 平均光谱相似度
        spectral_sim = F.cosine_similarity(
            context1['mean_spectrum'].unsqueeze(0),
            context2['mean_spectrum'].unsqueeze(0)
        ).item()

        # 2. 纹理特征相似度
        texture_sim = F.cosine_similarity(
            context1['std_spectrum'].unsqueeze(0),
            context2['std_spectrum'].unsqueeze(0)
        ).item()

        # 3. 尺寸相似度惩罚
        size_ratio = min(context1['size'][0], context2['size'][0]) / max(context1['size'][0], context2['size'][0], 1e-6)

        # 综合相似度 = (0.6*光谱 + 0.4*纹理) * 尺寸比例
        return (0.6 * spectral_sim + 0.4 * texture_sim) * size_ratio

    def _spatial_filter(self, points, min_dist):
        """空间多样性筛选算法"""
        points = points.float()
        selected = []
        remaining = points[torch.randperm(len(points))]
        while len(remaining) > 0:
            current = remaining[0]
            selected.append(current)
            distances = torch.norm(remaining - current, dim=1)
            remaining = remaining[distances > min_dist]
        return torch.stack(selected) if selected else torch.empty(0, 2, device=self.device)

    def hybrid_selection(self,
                         local_radius=30,
                         local_topk=100,
                         superpixel_topk=3,
                         global_topk=200,
                         min_dist=15,
                         spectral_threshold=0.5,
                         context_threshold=0.6,
                         min_points_per_sp=0):
        """
        执行局部和全局搜索并返回合并结果
        """
        # 直接调用 local_search
        local_points = self.local_search(
            radius=local_radius,
            topk=local_topk,
            use_texture=True
        )

        # 直接调用 global_superpixel_search
        global_points = self.global_superpixel_search(
            topk_per_superpixel=superpixel_topk,
            global_topk=global_topk,
            min_dist=min_dist,
            spectral_threshold=spectral_threshold,
            context_threshold=context_threshold,
            min_points_per_sp=min_points_per_sp
        )

        # 合并去重
        combined_points = torch.cat([local_points, global_points]).unique(dim=0)

        return {
            'local': local_points.cpu().numpy(),
            'global': global_points.cpu().numpy(),
            'combined': combined_points.cpu().numpy(),
            'superpixels': self.superpixels.cpu().numpy()
        }

    def set_target_spectrum(self, spectrum):
        """设置目标光谱（不依赖位置）"""
        self.target = spectrum.float().to(self.device)
        self.raw_target = spectrum.float().to(self.device)

    def global_match_only(self, **kwargs):
        """仅执行全局匹配"""
        return self.global_superpixel_search(**kwargs)

    def visualize(self, results, labels_map=None, local_radius=40):
        plt.figure(figsize=(20, 10))
        target_y, target_x = self.target_loc
        target_label = labels_map[target_y, target_x] if labels_map is not None else None

        color_image = np.zeros((self.H, self.W, 3))
        if labels_map is not None:
            unique_labels = np.unique(labels_map)
            label_to_index = {label: i for i, label in enumerate(unique_labels)}
            cmap = plt.get_cmap('tab20', len(unique_labels) + 1)
            color_lut = np.array([cmap(i)[:3] for i in range(len(unique_labels))])
            color_image = color_lut[np.vectorize(label_to_index.get)(labels_map)]
            color_image[target_y, target_x] = self.COLOR_TARGET_POINT
        else:
            color_image[:] = self.COLOR_BACKGROUND

        # 子图1：超像素分割结果
        plt.subplot(151)
        plt.imshow(results['superpixels'], cmap='nipy_spectral')
        plt.plot(target_x, target_y, 'ro', markersize=8, markeredgecolor='w')
        plt.title('Superpixel Segmentation')
        plt.axis('off')

        # 子图2：局部点分布
        plt.subplot(152)
        plt.imshow(color_image)
        xs, ys = results['local'][:, 1], results['local'][:, 0]
        local_colors = []
        for y, x in results['local']:
            if labels_map is not None and 0 <= y < self.H and 0 <= x < self.W:
                label = labels_map[int(y), int(x)]
                if label == target_label:
                    local_colors.append(self.COLOR_SAME_CLASS)
                else:
                    local_colors.append(self.COLOR_DIFF_CLASS)
            else:
                local_colors.append(self.COLOR_UNKNOWN)
        for (x, y, c) in zip(xs, ys, local_colors):
            plt.plot(x, y, 'o', markersize=4, markerfacecolor=c, markeredgecolor='none')
        plt.title(f'Local Points\nTarget Class: {target_label}')
        plt.axis('off')

        # 子图3：全局点分布
        plt.subplot(153)
        self._plot_points(plt.gca(), results['global'], labels_map, target_label, color_image)
        plt.title('Global Points')
        plt.axis('off')

        # 子图4：组合点分布
        plt.subplot(154)
        self._plot_points(plt.gca(), results['combined'], labels_map, target_label, color_image)
        plt.title('Combined Points')
        plt.axis('off')

        # 子图5：局部放大视图
        plt.subplot(155)
        y_min, y_max = max(0, target_y - local_radius), min(self.H, target_y + local_radius)
        x_min, x_max = max(0, target_x - local_radius), min(self.W, target_x + local_radius)
        zoom_img = color_image[y_min:y_max, x_min:x_max]
        zoom_local_points = [(x - x_min, y - y_min) for y, x in results['local']
                             if y_min <= y < y_max and x_min <= x < x_max]
        plt.imshow(zoom_img)
        for x, y in zoom_local_points:
            label = labels_map[int(y + y_min), int(x + x_min)] if labels_map is not None else -1
            if label == target_label:
                plt.plot(x, y, 'o', markersize=6, markerfacecolor=self.COLOR_SAME_CLASS)
            elif label != 0:
                plt.plot(x, y, 'x', markersize=6, markerfacecolor=self.COLOR_DIFF_CLASS)
            else:
                plt.plot(x, y, 'o', markersize=4, markerfacecolor=self.COLOR_OTHER)
        plt.title(f'Zoomed Local\nRadius={local_radius}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def _plot_points(self, ax, points, labels_map, target_label, color_image):
        ax.imshow(color_image)
        for y, x in points:
            if labels_map is not None and 0 <= y < self.H and 0 <= x < self.W:
                label = labels_map[int(y), int(x)]
                if label == target_label:
                    ax.plot(x, y, 'o', markersize=4, markerfacecolor=self.COLOR_SAME_CLASS)
                elif label != 0:
                    ax.plot(x, y, 'x', markersize=4, markerfacecolor=self.COLOR_DIFF_CLASS)
                else:
                    ax.plot(x, y, 'o', markersize=2, markerfacecolor=self.COLOR_OTHER)
            else:
                ax.plot(x, y, 'o', markersize=2, markerfacecolor=self.COLOR_UNKNOWN)

    def visualize_superpixels(self):
        # 使用PCA前三个分量显示图像
        image_np = self.image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        superpixels_np = self.superpixels.cpu().numpy()

        # 简单的PCA可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        h, w, c = image_np.shape
        image_flat = image_np.reshape(-1, c)
        image_pca = pca.fit_transform(image_flat).reshape(h, w, 3)
        image_pca = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_pca)
        plt.title("PCA Visualization")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(superpixels_np, cmap='nipy_spectral')
        plt.title("Superpixel Segmentation")
        plt.colorbar(ticks=np.unique(superpixels_np))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 测试代码，创建 [C, H, W] 格式的随机数据
    C, H, W = 200, 512, 512
    target = np.random.randn(C)
    target /= np.linalg.norm(target)

    # 创建 [C, H, W] 格式数据
    data = np.random.randn(C, H, W)
    target_region = target[:, np.newaxis, np.newaxis] + np.random.normal(0, 0.1, (C, 100, 100))
    data[:, 200:300, 200:300] = target_region
    data /= np.linalg.norm(data, axis=0, keepdims=True)  # 对每个像素归一化

    labels = np.zeros((H, W), dtype=int)
    labels[200:300, 200:300] = 1
    labels[100:150, 400:450] = 1
    labels[350:400, 100:150] = 2

    matcher = EnhancedHybridMatcher(data, (250, 250), device='cpu', normalize=True,
                                   similarity_method='mahalanobis', low_pass_kernel_size=7)

    results = matcher.hybrid_selection(local_radius=40, local_topk=150,
                                      superpixel_topk=3, global_topk=300)

    print("局部点数:", len(results['local']))
    print("全局点数:", len(results['global']))
    print("总计点数:", len(results['combined']))

    matcher.visualize(results, labels_map=labels)