import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os


class EnhancedBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.l2_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)

    def forward(self, x):
        x = self.bn(x)
        return self.l2_norm(x)

class VariantPromptNetwork(nn.Module):
    """变化提示生成器 - 使用卷积处理图像维度的私有特征"""

    def __init__(self, in_channels, prompt_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=3, padding=1),
            EnhancedBatchNorm(prompt_dim),
            nn.ReLU(),
        )

    def forward(self, private_feat):
        # private_feat shape: (batch, channels, height, width)
        prompt = self.net(private_feat)
        return prompt  # shape: (batch, prompt_dim, height, width)


class InvariantPromptGenerator(nn.Module):
    """不变提示生成器 - 使用卷积融合共享和私有特征"""

    def __init__(self, shared_in_channels, private_in_channels, prompt_dim=64):
        super().__init__()
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Conv2d(shared_in_channels + private_in_channels, prompt_dim, kernel_size=3, padding=1),
            EnhancedBatchNorm(prompt_dim),
            nn.ReLU(),
        )

        # 原型增强模块
        self.proto_enhancer = nn.Sequential(
            nn.Conv2d(shared_in_channels, prompt_dim, kernel_size=1),
            nn.Tanh()
        )
        self.prototypes = None

    def _find_nearest_prototype(self, base_prompt, domain):
        # 计算基础提示的特征向量 [B, C]
        query_vector = F.adaptive_avg_pool2d(base_prompt, (1, 1)).squeeze(-1).squeeze(-1)

        best_prototypes = []
        for i in range(query_vector.size(0)):
            min_distance = float('inf')
            best_proto = None

            # 检查原型存储类型：如果是函数则调用，如果是字典则遍历
            if callable(self.prototypes[domain]):
                # 如果是函数，直接调用获取原型
                try:
                    proto = self.prototypes[domain](i)  # 假设i是class_id
                    distance = 1 - F.cosine_similarity(
                        query_vector[i:i + 1],
                        proto.unsqueeze(0),
                        dim=1
                    ).item()

                    if distance < min_distance:
                        min_distance = distance
                        best_proto = proto
                except Exception as e:
                    print(f"Error getting prototype: {e}")
            else:
                # 如果是字典，遍历所有类别原型
                for class_id, proto in self.prototypes[domain].items():
                    distance = 1 - F.cosine_similarity(
                        query_vector[i:i + 1],
                        proto.unsqueeze(0),
                        dim=1
                    ).item()

                    if distance < min_distance:
                        min_distance = distance
                        best_proto = proto

            # 确保至少返回一个原型
            if best_proto is None:
                best_proto = torch.zeros_like(query_vector[0])

            best_prototypes.append(best_proto)

        return torch.stack(best_prototypes)  # [B, C]

    def update_prototypes(self, prototypes):
        """更新类原型参考"""
        self.prototypes = prototypes

    def forward(self, shared_feat, private_feat, class_ids=None, domain=None):
        # shared_feat shape: (batch, channels, height, width)
        # private_feat shape: (batch, channels, height, width)

        # 1. 融合共享和私有特征
        combined = torch.cat([shared_feat, private_feat], dim=1)  # 在通道维度拼接
        base_prompt = self.fusion_net(combined)

        # 2. 原型增强（仅在训练阶段和原型可用时）
        if domain and class_ids is not None and self.prototypes is not None:
            proto_vectors = []
            for class_id in class_ids:
                # 直接使用当前域的原型
                if callable(self.prototypes[domain]):
                    proto = self.prototypes[domain](class_id.item())
                else:
                    proto = self.prototypes[domain](class_id.item())
                proto_vectors.append(proto)

            # 将原型向量转换为与特征图相同维度的张量
            proto_tensor = torch.stack(proto_vectors).to(base_prompt.device)
            _, _, h, w = base_prompt.shape
            proto_tensor = proto_tensor.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

            # 原型增强提示
            proto_effect = self.proto_enhancer(proto_tensor)
            # enhanced_prompt = base_prompt + proto_effect
            enhanced_prompt = proto_effect
            return enhanced_prompt
        elif self.prototypes is not None and domain:
            # 测试模式：用共享特征（与原型同维）查询最相近的原型，再走与训练分支一致的处理流程
            proto_tensor = self._find_nearest_prototype(shared_feat, domain)  # [B, C]
            _, _, h, w = base_prompt.shape
            proto_tensor = proto_tensor.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
            return self.proto_enhancer(proto_tensor)

        # return base_prompt
        # return base_prompt


class PromptDistiller(nn.Module):
    """提示蒸馏器 - 使用卷积适配器处理图像维度的提示"""

    def __init__(self, prompt_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(prompt_dim, prompt_dim * 2, kernel_size=3, padding=1),
            EnhancedBatchNorm(prompt_dim * 2),
            nn.ReLU(),
            nn.Conv2d(prompt_dim * 2, prompt_dim, kernel_size=3, padding=1)
        )

    def forward(self, student_prompt, teacher_prompt):
        # 通过卷积适配器调整学生提示
        adapted = self.adapter(student_prompt)
        # 计算蒸馏损失（像素级MSE）
        loss = F.mse_loss(adapted, teacher_prompt.detach())
        return adapted, loss


class IncrementalPrototypeMemory(nn.Module):
    def __init__(self, shared_in_channels,
                 warmup_epochs=5,
                 quality_threshold=0.7,
                 decay_start=0.5,
                 decay_end=0.95,
                 decay_steps=20,
                 device='cuda',
                 log_file=None,  # 修改为可选参数
                 round_idx=None):  # 添加增量轮次参数
        super().__init__()
        """
        Args:
            shared_in_channels: 共享特征的通道数
            warmup_epochs: 原型更新的预热期数（前N个epoch不更新原型）
            quality_threshold: 原型更新的特征质量阈值（0-1）
            decay_start: 原型更新的初始衰减系数
            decay_end: 原型更新的最终衰减系数
            decay_steps: 从初始衰减到最终衰减的步数（epoch数）
            device: 计算设备
            log_file: 日志文件路径（可选）
            round_idx: 增量轮次索引（可选）
        """
        super().__init__()
        self.shared_in_channels = shared_in_channels
        self.warmup_epochs = warmup_epochs
        self.quality_threshold = quality_threshold
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.decay_steps = decay_steps
        self.device = device
        self.current_epoch = 0
        self.round_idx = round_idx if round_idx is not None else 0

        # 计算衰减步长
        self.decay_step = (decay_end - decay_start) / decay_steps if decay_steps > 0 else 0
        self.decay = decay_start  # 初始衰减系数

        # 双域原型存储
        self.prototypes = {
            'real': defaultdict(lambda: torch.zeros(shared_in_channels, device=device)),
            'gen': defaultdict(lambda: torch.zeros(shared_in_channels, device=device))
        }
        self.prototype_counts = {
            'real': defaultdict(int),
            'gen': defaultdict(int)
        }
        self.current_classes = []  # 当前存储的类别列表

        # 动态生成日志文件名
        if log_file is None:
            self.log_file = f"feature_quality_log_round_{self.round_idx}.txt"
        else:
            self.log_file = log_file

        # 确保日志文件目录存在（仅当有目录路径时）
        log_dir = os.path.dirname(self.log_file)
        if log_dir:  # 只有当目录路径非空时才创建
            os.makedirs(log_dir, exist_ok=True)

        # 创建或清空日志文件
        with open(self.log_file, "w") as f:
            f.write("epoch,batch_idx,sample_idx,class_id,norm,quality,update_decision\n")

        self.feature_norms = []  # 存储每个样本的特征范数
        self.feature_qualities = []  # 存储每个样本的质量分数
        self.class_ids = []  # 存储对应的类别ID
        self.class_status = {}

        self.old_class_decay = 0.2  # 旧类更新衰减系数 (0.2意味着只更新20%)
        # 新增特征范数阈值
        self.feature_norm_threshold = 1000.0  # 根据实际数据调整

    def mark_class_as_old(self, class_id):
        """将类别标记为旧类"""
        if class_id in self.class_status:
            self.class_status[class_id] = 'old'

    def update_epoch(self, epoch=None):
        """更新当前epoch，并调整衰减系数"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # 当超过预热期后，逐步增加衰减系数
        if self.current_epoch > self.warmup_epochs:
            self.decay = min(self.decay_end, self.decay + self.decay_step)
            # 打印调试信息
            # print(f"[原型内存] 更新衰减系数: epoch={self.current_epoch}, decay={self.decay:.3f}")

    def add_class(self, class_id, status='new'):
        """添加新类别并指定状态"""
        if class_id not in self.current_classes:
            self.current_classes.append(class_id)
            self.class_status[class_id] = status  # 记录类别状态

            for domain in ['real', 'gen']:
                self.prototype_counts[domain][class_id] = 0
                self.prototypes[domain][class_id] = torch.zeros(
                    self.shared_in_channels, device=self.device)

    def _calculate_feature_quality(self, feature):
        """
        计算特征质量（使用特征范数作为指标）
        返回每个样本的范数和质量分数，而不是平均值

        Args:
            feature: 特征张量 [batch, channels, height, width]

        Returns:
            norms: 每个样本的特征范数 [batch]
            qualities: 每个样本的质量分数 [batch]
        """
        # 计算每个样本的特征范数
        sample_norms = torch.norm(feature, p=2, dim=(1, 2, 3))  # [batch]

        # 特征范数的经验范围在2-8之间，归一化到0-1范围
        # 5.0为最大参考值（根据实际数据分布调整）
        qualities = torch.clamp(sample_norms / 100.0, 0.0, 1.0)

        # 使用detach()断开计算图
        return sample_norms.detach().cpu().numpy(), qualities.detach().cpu().numpy()

    def _update_prototype(self, domain, class_id, new_feature):
        """
        按论文 2.4 节的动量更新规则更新原型：
          新原型 = m × 旧原型 + (1-m) × 当前batch共享特征均值
          - warm-up 阶段（前 warmup_epochs 轮）：m = 0.9
          - 该类原型质量差（簇内一致性低，质量分 < quality_threshold）：m = 0.7（快点跟上新数据）
          - 原型质量好：m = 0.95（稳住不被小波动带跑）
        """
        class_ids_t = torch.tensor([class_id])

        # 计算每个样本的质量分数
        norms, qualities = self._calculate_feature_quality(new_feature)
        batch_quality = float(qualities.mean().item()) if len(qualities) > 0 else 0.0

        # ---- 动量系数自适应切换（论文规则）----
        if self.current_epoch < self.warmup_epochs:
            momentum = 0.9          # warm-up 阶段
        elif batch_quality < self.quality_threshold:
            momentum = 0.7          # 质量差：让原型快点跟上新数据
        else:
            momentum = 0.95         # 质量好：稳住

        # 用 batch 内特征均值作为新观测
        pooled_feature = F.adaptive_avg_pool2d(new_feature, (1, 1))
        pooled_feature = pooled_feature.mean(dim=0).squeeze(-1).squeeze(-1).detach()

        update_decisions = [True] * new_feature.shape[0]
        reason = f"m={momentum:.2f},q={batch_quality:.3f}"
        self._record_quality(class_id, new_feature, update_decisions, reason)

        count = self.prototype_counts[domain][class_id]
        if count == 0:
            self.prototypes[domain][class_id] = pooled_feature
        else:
            self.prototypes[domain][class_id] = (
                    momentum * self.prototypes[domain][class_id] +
                    (1 - momentum) * pooled_feature
            )

        self.prototype_counts[domain][class_id] += new_feature.shape[0]
    def update_prototypes(self, class_ids, real_shared, gen_shared):
        """
        批量更新真实和生成图像的原型

        Args:
            class_ids: 类别ID张量 [batch]
            real_shared: 真实图像共享特征 [batch, channels, height, width]
            gen_shared: 生成图像共享特征 [batch, channels, height, width]
        """

        if self.current_epoch < self.warmup_epochs:
            return

        class_ids = class_ids.to(self.device)

        for class_id in torch.unique(class_ids):
            class_id = class_id.item()
            self.add_class(class_id)
            mask = (class_ids == class_id)

            # 更新真实图像原型
            if mask.sum() > 0:
                self._update_prototype('real', class_id, real_shared[mask])

            # 更新生成图像原型
            if mask.sum() > 0:
                self._update_prototype('gen', class_id, gen_shared[mask])

    def _record_quality(self, class_id, features, update_decisions, reason):
        """
        记录详细的调试信息
        """
        # 获取当前批次信息
        batch_size = features.shape[0]
        norms, qualities = self._calculate_feature_quality(features)

        # 准备记录数据
        for i in range(batch_size):
            self.feature_norms.append(norms[i])
            self.feature_qualities.append(qualities[i])
            self.class_ids.append(class_id)

            # 如果update_decisions是单个值，应用到所有样本
            if isinstance(update_decisions, bool):
                update_decision = update_decisions
            else:
                update_decision = update_decisions[i]

            # 写入日志文件
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.current_epoch},"
                    f"{len(self.feature_norms)},{i},{class_id},"
                    f"{norms[i]:.4f},{qualities[i]:.4f},"
                    f"{update_decision}\n"
                )

    def __call__(self, domain, class_id):
        """
        获取特定域的类原型

        Args:
            domain: 域名称 ('real' 或 'gen')
            class_id: 类别ID

        Returns:
            原型向量 [channels]
        """
        class_id = int(class_id)  # 确保整数类型
        if class_id in self.prototypes[domain]:
            return self.prototypes[domain][class_id]
        return torch.zeros(self.shared_in_channels, device=self.device)

    def get_quality_stats(self):
        """获取当前epoch的质量统计信息"""
        if not self.feature_norms:
            return None

        return {
            "epoch": self.current_epoch,
            "total_samples": len(self.feature_norms),
            "avg_norm": np.mean(self.feature_norms),
            "min_norm": np.min(self.feature_norms),
            "max_norm": np.max(self.feature_norms),
            "avg_quality": np.mean(self.feature_qualities),
            "min_quality": np.min(self.feature_qualities),
            "max_quality": np.max(self.feature_qualities),
            "update_ratio": np.mean([1 if q >= self.quality_threshold else 0
                                     for q in self.feature_qualities])
        }

    def reset_quality_log(self):
        """重置质量日志（每个epoch结束时调用）"""
        self.feature_norms = []
        self.feature_qualities = []
        self.class_ids = []


class GuidedPrompt(nn.Module):
    """完整的提示蒸馏与分类系统 - 处理图像维度的特征"""

    def __init__(self,
                 real_shared_in_channels,
                 real_private_in_channels,
                 gen_shared_in_channels,
                 gen_private_in_channels,
                 num_initial_classes,
                 prompt_dim=128,
                 device='cuda',
                 round_idx=None,  # 添加轮次参数
                 # 新增原型稳定性参数
                 proto_warmup_epochs=5,
                 proto_quality_threshold=0.7,
                 proto_decay_start=0.5,
                 proto_decay_end=0.95,
                 proto_decay_steps=20):
        super().__init__()
        self.device = device
        self.prompt_dim = prompt_dim
        self.round_idx = round_idx if round_idx is not None else 0

        # 记录输入通道数
        self.real_shared_in_channels = real_shared_in_channels
        self.real_private_in_channels = real_private_in_channels
        self.gen_shared_in_channels = gen_shared_in_channels
        self.gen_private_in_channels = gen_private_in_channels

        # 初始化原型内存，传递轮次信息
        # 论文 2.3.1/2.4：原型维度 = 128，与共享特征对齐
        self.memory = IncrementalPrototypeMemory(
            shared_in_channels=real_shared_in_channels,
            warmup_epochs=proto_warmup_epochs,
            quality_threshold=proto_quality_threshold,
            decay_start=proto_decay_start,
            decay_end=proto_decay_end,
            decay_steps=proto_decay_steps,
            device=device,
            round_idx=self.round_idx  # 传递轮次信息
        )

        # 创建提示生成器 - 每个都有自定义的输入通道数
        self.invariant_real = InvariantPromptGenerator(
            real_shared_in_channels, real_private_in_channels, prompt_dim).to(device)
        self.invariant_gen = InvariantPromptGenerator(
            gen_shared_in_channels, gen_private_in_channels, prompt_dim).to(device)

        # 变化提示生成器 - 每个都有自定义的输入通道数
        self.variant_real = VariantPromptNetwork(real_private_in_channels, prompt_dim).to(device)
        self.variant_gen = VariantPromptNetwork(gen_private_in_channels, prompt_dim).to(device)

        # 设置原型参考
        self.invariant_real.update_prototypes({
            'real': lambda cid: self.memory('real', cid)
        })
        self.invariant_gen.update_prototypes({
            'gen': lambda cid: self.memory('gen', cid)
        })

        # 蒸馏器
        self.distiller_invariant = PromptDistiller(prompt_dim).to(device)
        self.distiller_variant = PromptDistiller(prompt_dim).to(device)

    def forward(self, real_shared, real_private, gen_shared, gen_private, class_ids=None):
        # === 生成四种提示 ===
        # 1. 真实图像的不变提示（融合共享和私有特征）
        real_inv_prompt = self.invariant_real(
            real_shared, real_private, class_ids, domain='real')

        # 2. 真实图像的变化提示（仅私有特征）
        real_var_prompt = self.variant_real(real_private)

        # 3. 生成图像的不变提示（融合共享和私有特征）
        gen_inv_prompt = self.invariant_gen(
            gen_shared, gen_private, class_ids, domain='gen')

        # 4. 生成图像的变化提示（仅私有特征）
        gen_var_prompt = self.variant_gen(gen_private)

        # === 训练与测试分支 ===
        if class_ids is not None:
            # 训练阶段
            return self._train_forward(
                real_inv_prompt, real_var_prompt,
                gen_inv_prompt, gen_var_prompt,
                class_ids, real_shared, gen_shared
            )
        else:
            # 测试阶段
            return self._test_forward(
                real_inv_prompt, real_var_prompt,
                gen_inv_prompt, gen_var_prompt,
                real_shared, real_private
            )

    def _train_forward(self, real_inv_prompt, real_var_prompt,
                       gen_inv_prompt, gen_var_prompt,
                       class_ids, real_shared, gen_shared):
        """训练阶段前向传播"""
        # 1. 更新原型内存
        self.memory.update_prototypes(class_ids, real_shared, gen_shared)
        #
        # # 2. 蒸馏学习
        # # - 真实不变提示指导生成不变提示
        dist_inv_prompt, loss_inv = self.distiller_invariant(
            gen_inv_prompt, real_inv_prompt)
        #
        # # - 真实不变提示指导生成变化提示
        dist_var_prompt, loss_var = self.distiller_variant(
            gen_var_prompt, real_inv_prompt)

        # loss_inv = 0
        # loss_var = 0
        # 3. 构建双源提示 - 使用concat融合
        # 真实图像提示
        real_full_prompt = torch.cat([real_inv_prompt, real_var_prompt], dim=1)

        # 生成图像提示（蒸馏后）
        gen_full_prompt = torch.cat([dist_inv_prompt, dist_var_prompt], dim=1)
        # gen_full_prompt = torch.cat([gen_inv_prompt, gen_var_prompt], dim=1)

        # 4. 融合双源提示
        combined_prompt = torch.cat([real_full_prompt, gen_full_prompt], dim=1)

        return combined_prompt, 0.5*loss_inv+0.5*loss_var

    def _test_forward(self, real_inv_prompt, real_var_prompt,
                      gen_inv_prompt, gen_var_prompt,
                      real_shared, real_private):
        """测试阶段前向传播 - 与训练一致的流程"""
        with torch.no_grad():
            # 1. 使用真实不变提示作为教师
            # 重新计算真实不变提示（避免训练原型的影响）
            # teacher_prompt = self.invariant_real(
            #     real_shared, real_private, None, domain='real')

            # # 2. 蒸馏学生提示
            tested_inv_prompt, _ = self.distiller_invariant(
                gen_inv_prompt, real_inv_prompt)
            tested_var_prompt, _ = self.distiller_variant(
                gen_var_prompt, real_inv_prompt)

            # 3. 构建双源提示 - 与训练一致的concat融合
            # 真实图像提示
            real_full_prompt = torch.cat([real_inv_prompt, real_var_prompt], dim=1)
            # real_full_prompt = torch.cat([real_inv_prompt, real_var_prompt], dim=1)

            # 生成图像提示（蒸馏后）
            # gen_full_prompt = torch.cat([tested_inv_prompt, tested_var_prompt], dim=1)
            gen_full_prompt = torch.cat([gen_inv_prompt, gen_var_prompt], dim=1)

            # 4. 融合双源提示
            combined_prompt = torch.cat([real_full_prompt, gen_full_prompt], dim=1)

        return combined_prompt

    def prepare_for_incremental_learning(self, old_classes):
        """在增量学习开始前标记旧类别"""
        for class_id in old_classes:
            self.memory.mark_class_as_old(class_id)

# 使用示例
if __name__ == "__main__":
    # 参数设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, height, width = 16, 11, 11
    prompt_dim = 128
    num_classes = 10

    # 自定义每个特征的输入通道数
    real_shared_in_channels = 100
    real_private_in_channels = 80
    gen_shared_in_channels = 90
    gen_private_in_channels = 85

    # 创建模型
    model = GuidedPrompt(
        real_shared_in_channels=real_shared_in_channels,
        real_private_in_channels=real_private_in_channels,
        gen_shared_in_channels=gen_shared_in_channels,
        gen_private_in_channels=gen_private_in_channels,
        prompt_dim=prompt_dim,
        num_initial_classes=num_classes,
        device=device
    )

    # 创建模拟输入（图像维度）
    real_shared = torch.randn(batch_size, real_shared_in_channels, height, width).to(device)
    real_private = torch.randn(batch_size, real_private_in_channels, height, width).to(device)
    gen_shared = torch.randn(batch_size, gen_shared_in_channels, height, width).to(device)
    gen_private = torch.randn(batch_size, gen_private_in_channels, height, width).to(device)
    class_ids = torch.randint(0, num_classes, (batch_size,)).to(device)

    # 训练模式
    model.train()
    outputs = model(real_shared, real_private, gen_shared, gen_private, class_ids)
    print("Train Outputs:")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Real prompt shape: {outputs['real_prompt'].shape}")

    # 测试模式
    model.eval()
    test_outputs = model(real_shared, real_private, gen_shared, gen_private, None)
    print("\nTest Outputs:")
    print(f"Logits shape: {test_outputs['logits'].shape}")
    print(f"Combined prompt shape: {test_outputs['combined_prompt'].shape}")

    # 添加新类别
    new_classes = [10, 11, 12]
    model.add_new_classes(new_classes)

    # 使用新类别进行测试
    test_outputs_new = model(real_shared, real_private, gen_shared, gen_private, None)
    print(f"New classes logits shape: {test_outputs_new['logits'].shape}")