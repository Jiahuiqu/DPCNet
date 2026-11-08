import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ImagePromptModule(nn.Module):
    def __init__(self,
                 real_shared_in_channels: int,
                 real_private_in_channels: int,
                 gen_shared_in_channels: int,
                 gen_private_in_channels: int,
                 num_initial_classes: int,
                 prompt_dim: int = 128,
                 device: str = 'cuda',
                 # 原型稳定性参数
                 proto_warmup_epochs: int = 5,
                 proto_quality_threshold: float = 0.7):
        super().__init__()

        self.device = device
        self.prompt_dim = prompt_dim
        self.num_classes = num_initial_classes
        self.current_epoch = 0

        # 原型稳定性参数
        self.proto_warmup_epochs = proto_warmup_epochs
        self.proto_quality_threshold = proto_quality_threshold

        # 特征投影层 - 专门处理4D图像特征 [B, C, H, W]
        # 使用全局平均池化 + 线性投影
        self.real_shared_proj = self._create_projection(real_shared_in_channels, prompt_dim)
        self.real_private_proj = self._create_projection(real_private_in_channels, prompt_dim)
        self.gen_shared_proj = self._create_projection(gen_shared_in_channels, prompt_dim)
        self.gen_private_proj = self._create_projection(gen_private_in_channels, prompt_dim)

        # 不变提示生成器 - 用于保留旧知识
        self.invariant_prompt_generator = InvariantPromptGenerator(
            prompt_dim=prompt_dim,
            num_classes=num_initial_classes,
            device=device
        )

        # 变化提示生成器 - 用于学习新特征
        self.variant_prompt_generator = VariantPromptGenerator(
            prompt_dim=prompt_dim
        )

        # 提示融合模块
        self.prompt_fusion = PromptFusionModule(prompt_dim)

        # 原型内存
        self.prototype_memory = PrototypeMemory(
            prompt_dim=prompt_dim,
            num_classes=num_initial_classes,
            warmup_epochs=proto_warmup_epochs,
            quality_threshold=proto_quality_threshold,
            device=device
        )

        # 初始化权重
        self._init_weights()

    def _create_projection(self, in_channels, out_channels):
        """创建特征投影模块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(in_channels, out_channels),  # [B, prompt_dim]
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, real_shared_feat, real_private_feat,
                gen_shared_feat, gen_private_feat, class_ids: Optional[torch.Tensor] = None):
        """
        前向传播

        Args:
            real_shared_feat: 真实图像共享特征 [B, C, H, W]
            real_private_feat: 真实图像私有特征 [B, C, H, W]
            gen_shared_feat: 生成图像共享特征 [B, C, H, W]
            gen_private_feat: 生成图像私有特征 [B, C, H, W]
            class_ids: 类别标签 [B]
        """
        # 1. 特征投影到提示空间
        real_shared_proj = self.real_shared_proj(real_shared_feat)  # [B, prompt_dim]
        real_private_proj = self.real_private_proj(real_private_feat)
        gen_shared_proj = self.gen_shared_proj(gen_shared_feat)
        gen_private_proj = self.gen_private_proj(gen_private_feat)

        # 2. 生成不变提示（用于知识保留）
        real_invariant_prompt = self.invariant_prompt_generator(
            real_shared_proj, real_private_proj, class_ids
        )

        gen_invariant_prompt = self.invariant_prompt_generator(
            gen_shared_proj, gen_private_proj, class_ids
        )

        # 3. 生成变化提示（用于新特征学习）
        real_variant_prompt = self.variant_prompt_generator(
            real_shared_proj, real_private_proj, class_ids
        )

        gen_variant_prompt = self.variant_prompt_generator(
            gen_shared_proj, gen_private_proj, class_ids
        )

        # 4. 融合提示
        fused_prompt = self.prompt_fusion(real_invariant_prompt, gen_invariant_prompt,
                                          real_variant_prompt, gen_variant_prompt)

        # 5. 训练时的蒸馏损失
        distill_loss = torch.tensor(0.0).to(self.device)
        if class_ids is not None and self.training:
            # 更新原型内存
            self.prototype_memory.update(real_shared_proj.detach(), class_ids)

            # 计算蒸馏损失
            # distill_loss = self._compute_distillation_loss(
            #     real_shared_proj, real_private_proj, class_ids
            # )

        return fused_prompt, distill_loss

    def _compute_distillation_loss(self, shared_feat, private_feat, class_ids):
        """计算知识蒸馏损失"""
        # 1. 原型一致性损失
        proto_loss = self.prototype_memory.consistency_loss(shared_feat, class_ids)

        # 2. 特征稳定性损失
        stability_loss = F.mse_loss(shared_feat, private_feat.detach())

        # 3. 根据训练阶段调整损失权重
        if self.current_epoch < self.proto_warmup_epochs:
            w_proto, w_stability = 0.1, 0.5  # 预热阶段
        else:
            w_proto, w_stability = 0.3, 0.2  # 稳定阶段

        return w_proto * proto_loss + w_stability * stability_loss

    def add_new_classes(self, new_class_ids):
        """添加新类别支持"""
        self.invariant_prompt_generator.add_new_classes(len(new_class_ids))
        self.prototype_memory.add_new_classes(new_class_ids)
        self.num_classes += len(new_class_ids)

    def update_epoch(self, epoch):
        """更新训练周期"""
        self.current_epoch = epoch
        self.prototype_memory.update_epoch(epoch)


class InvariantPromptGenerator(nn.Module):
    """不变提示生成器 - 专注于知识保留"""

    def __init__(self, prompt_dim: int, num_classes: int, device: str):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_classes = num_classes
        self.device = device

        # 类别特定的提示基向量
        self.class_bases = nn.Parameter(
            torch.randn(num_classes, prompt_dim, device=device)
        )

        # 稳定性控制器
        self.stability_controller = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim // 2),
            nn.ReLU(),
            nn.Linear(prompt_dim // 2, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.class_bases)

    def forward(self, shared_feat, private_feat, class_ids=None):
        """
        生成不变提示

        Args:
            shared_feat: 共享特征 [B, prompt_dim]
            private_feat: 私有特征 [B, prompt_dim]
            class_ids: 类别标签 [B]
        """
        if class_ids is not None and self.training:
            # 训练模式：使用真实类别信息
            return self._training_forward(shared_feat, private_feat, class_ids)
        else:
            # 推理模式：基于相似度选择
            return self._inference_forward(shared_feat)

    def _training_forward(self, shared_feat, private_feat, class_ids):
        """训练时前向传播"""
        # 获取对应类别的基提示
        class_prompts = self.class_bases[class_ids]  # [B, prompt_dim]

        # 计算稳定性权重
        stability_weights = self.stability_controller(shared_feat)  # [B, 1]

        # 融合当前特征和类别基提示
        invariant_prompt = stability_weights * class_prompts + (1 - stability_weights) * shared_feat

        return invariant_prompt

    def _inference_forward(self, shared_feat):
        """推理时前向传播"""
        # 计算与所有类别基的相似度
        similarities = F.cosine_similarity(
            shared_feat.unsqueeze(1),  # [B, 1, prompt_dim]
            self.class_bases.unsqueeze(0),  # [1, num_classes, prompt_dim]
            dim=-1
        )  # [B, num_classes]

        # 选择最相关的几个类别
        topk_values, topk_indices = torch.topk(similarities, k=min(3, self.num_classes), dim=-1)

        # 加权组合基提示
        weights = F.softmax(topk_values, dim=-1)
        selected_bases = self.class_bases[topk_indices]  # [B, k, prompt_dim]
        weighted_prompt = torch.sum(weights.unsqueeze(-1) * selected_bases, dim=1)

        return weighted_prompt

    def add_new_classes(self, num_new_classes):
        """扩展新类别支持"""
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + num_new_classes

        # 创建新的基向量参数
        new_bases = nn.Parameter(
            torch.randn(new_num_classes, self.prompt_dim, device=self.device)
        )

        # 复制旧权重，新类别用较小值初始化
        with torch.no_grad():
            new_bases[:old_num_classes] = self.class_bases.data
            new_bases[old_num_classes:].normal_(mean=0.0, std=0.01)

        self.class_bases = new_bases
        self.num_classes = new_num_classes


class VariantPromptGenerator(nn.Module):
    """变化提示生成器 - 专注于新特征学习"""

    def __init__(self, prompt_dim: int):
        super().__init__()
        self.prompt_dim = prompt_dim

        # 变化提示生成网络
        self.variant_net = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prompt_dim * 4, prompt_dim * 2),
            nn.ReLU(),
            nn.Linear(prompt_dim * 2, prompt_dim)
        )

        # 新颖性检测器
        self.novelty_detector = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim // 2),
            nn.ReLU(),
            nn.Linear(prompt_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, gen_shared_feat, gen_private_feat, class_ids=None):
        """
        生成变化提示

        Args:
            gen_shared_feat: 生成图像共享特征 [B, prompt_dim]
            gen_private_feat: 生成图像私有特征 [B, prompt_dim]
        """
        # 拼接生成特征
        combined_feat = torch.cat([gen_shared_feat, gen_private_feat], dim=-1)

        # 生成基础变化提示
        base_variant = self.variant_net(combined_feat)

        # 检测新颖性程度
        novelty_score = self.novelty_detector(gen_shared_feat)

        # 根据新颖性调整提示
        variant_prompt = novelty_score * base_variant + (1 - novelty_score) * gen_shared_feat

        return variant_prompt


class PromptFusionModule(nn.Module):
    """提示融合模块"""

    def __init__(self, prompt_dim: int):
        super().__init__()
        self.prompt_dim = prompt_dim

        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.fusion_gate_2 = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, real_invariant_prompt, gen_invariant_prompt,
                real_variant_prompt, gen_variant_prompt):
        """融合不变提示和变化提示"""
        # 计算融合权重
        invariant_combined = torch.cat([real_invariant_prompt, gen_invariant_prompt], dim=-1)
        variant_combined = torch.cat([real_variant_prompt, gen_variant_prompt], dim=-1)
        gate_weights = self.fusion_gate(invariant_combined)  # [B, 2]
        gate_weights_2 = self.fusion_gate_2(variant_combined)  # [B, 2]

        # 加权融合
        fused = (gate_weights[:, 0:1] * real_invariant_prompt +
                 gate_weights[:, 1:2] * gen_invariant_prompt)

        fused_2 = (gate_weights_2[:, 0:1] * real_variant_prompt +
                   gate_weights_2[:, 1:2] * gen_variant_prompt)

        fused = torch.cat([fused, fused_2], dim=-1)
        return fused


class PrototypeMemory(nn.Module):
    """原型内存 - 存储各类别原型特征"""

    def __init__(self, prompt_dim: int, num_classes: int, warmup_epochs: int,
                 quality_threshold: float, device: str):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_classes = num_classes
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.quality_threshold = quality_threshold

        # 原型存储
        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, prompt_dim, device=device)
        )
        self.prototype_counts = torch.zeros(num_classes, device=device)
        self.prototype_quality = torch.zeros(num_classes, device=device)

        self.current_epoch = 0

    def update(self, features: torch.Tensor, class_ids: torch.Tensor):
        """更新原型内存"""
        if not self.training:
            return

        for cls_id in torch.unique(class_ids):
            cls_mask = (class_ids == cls_id)
            if cls_mask.sum() == 0:
                continue

            cls_features = features[cls_mask]
            cls_id_int = cls_id.item()

            # 计算新原型
            new_proto = cls_features.mean(dim=0)

            if self.prototype_counts[cls_id_int] == 0:
                # 第一次见到该类
                self.prototypes.data[cls_id_int] = new_proto
            else:
                # 动量更新，根据阶段调整动量
                if self.current_epoch < self.warmup_epochs:
                    momentum = 0.9  # 预热阶段快速学习
                elif self.prototype_quality[cls_id_int] < self.quality_threshold:
                    momentum = 0.7  # 低质量原型快速更新
                else:
                    momentum = 0.95  # 高质量原型缓慢更新

                self.prototypes.data[cls_id_int] = (
                        momentum * self.prototypes[cls_id_int] +
                        (1 - momentum) * new_proto
                )

            self.prototype_counts[cls_id_int] += cls_mask.sum().item()

            # 更新质量评估
            consistency = F.cosine_similarity(
                cls_features,
                self.prototypes[cls_id_int].unsqueeze(0),
                dim=-1
            ).mean()
            self.prototype_quality[cls_id_int] = consistency.item()

    def consistency_loss(self, features: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """计算原型一致性损失"""
        if not self.training:
            return torch.tensor(0.0).to(self.device)

        # 获取对应类别的原型
        target_prototypes = self.prototypes[class_ids]  # [B, prompt_dim]

        # 计算余弦相似度损失
        similarity = F.cosine_similarity(features, target_prototypes, dim=-1)
        loss = 1 - similarity.mean()

        return loss

    def add_new_classes(self, new_class_ids):
        """添加新类别原型"""
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + len(new_class_ids)

        # 扩展原型存储
        new_prototypes = nn.Parameter(
            torch.zeros(new_num_classes, self.prompt_dim, device=self.device)
        )
        new_prototypes.data[:old_num_classes] = self.prototypes.data

        # 扩展计数和质量数组
        new_counts = torch.zeros(new_num_classes, device=self.device)
        new_quality = torch.zeros(new_num_classes, device=self.device)
        new_counts[:old_num_classes] = self.prototype_counts
        new_quality[:old_num_classes] = self.prototype_quality

        self.prototypes = new_prototypes
        self.prototype_counts = new_counts
        self.prototype_quality = new_quality
        self.num_classes = new_num_classes

    def update_epoch(self, epoch: int):
        """更新训练周期"""
        self.current_epoch = epoch


class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 硬目标损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 软目标损失
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def test_dynamic_prompt_module():
    # Test parameters
    batch_size = 4
    feature_channels = [64, 32, 64, 32]  # [real_shared, real_private, gen_shared, gen_private]
    sequence_length = 16
    num_classes = 10
    prompt_dim = 128

    # Create test input tensors
    real_shared_seq = torch.randn(batch_size, feature_channels[0], sequence_length, sequence_length)
    real_private_seq = torch.randn(batch_size, feature_channels[1], sequence_length, sequence_length)
    gen_shared_seq = torch.randn(batch_size, feature_channels[2], sequence_length, sequence_length)
    gen_private_seq = torch.randn(batch_size, feature_channels[3], sequence_length, sequence_length)

    # Create class labels
    class_ids = torch.randint(0, num_classes, (batch_size,))

    # Initialize the DynamicPromptModule
    prompt_module = ImagePromptModule(
        real_shared_in_channels=feature_channels[0],
        real_private_in_channels=feature_channels[1],
        gen_shared_in_channels=feature_channels[2],
        gen_private_in_channels=feature_channels[3],
        num_initial_classes=num_classes,
        prompt_dim=prompt_dim,
        device='cpu'  # Using CPU for testing
    )

    # Test forward pass in training mode
    prompt_module.train()
    fused_prompt, distill_loss = prompt_module(
        real_shared_seq, real_private_seq,
        gen_shared_seq, gen_private_seq,
        class_ids
    )

    print(f"Fused prompt shape: {fused_prompt.shape}")  # Should be [B, prompt_dim]
    print(f"Distillation loss: {distill_loss.item()}")

    # Test forward pass in evaluation mode
    prompt_module.eval()
    fused_prompt_eval, _ = prompt_module(
        real_shared_seq, real_private_seq,
        gen_shared_seq, gen_private_seq
    )

    print(f"Evaluation mode prompt shape: {fused_prompt_eval.shape}")

    # Test adding new classes
    new_class_ids = [10, 11, 12]
    prompt_module.add_new_classes(new_class_ids)
    print(f"New number of classes: {prompt_module.num_classes}")

    # Test updating epoch
    prompt_module.update_epoch(10)
    print(f"Current epoch: {prompt_module.current_epoch}")


if __name__ == "__main__":
    test_dynamic_prompt_module()
