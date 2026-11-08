import torch
import torch.nn as nn
import torch.nn.functional as F
from Image_enhancement import figure_rand
from mamba_ssm.modules.mamba_simple import Mamba
# 导入新的ImagePromptModule
from memory_2 import ImagePromptModule  # 假设文档1保存为image_prompt_module.py
from model import base_module

class conv3x3(nn.Module):
    def __init__(self, in_feature, out_feature, stride=1):
        super(conv3x3, self).__init__()
        self.bn = nn.BatchNorm2d(in_feature)
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=stride, padding=1)
        self.GELU = nn.GELU()

    def forward(self, x):
        out = self.bn(x)
        out = self.GELU(out)
        out = self.conv(out)
        return out

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 卷积用于通道调整"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

class SpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super(SpatialAttention, self).__init__()
        self.q_conv = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.k_conv = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        D, C, H, W = x.size()
        q = self.q_conv(x).view(D, -1, W * H).permute(0, 2, 1)
        k = self.k_conv(x).view(D, -1, W * H)
        v = x.view(D, -1, H*W)
        attention = torch.bmm(q, k)
        attention = self.softmax(attention)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(D, C, H, W)
        out = self.gamma*out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        D, C, H, W = x.size()
        q = x.view(D, C, -1)
        k = x.view(D, C, -1).permute(0, 2, 1)
        v = x.view(D, C, H * W)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        out = torch.bmm(attention.permute(0, 2, 1), v).view(D, C, H, W)
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        self.use_ca = out_channels >= 64
        self.use_sa = out_channels >= 128
        if self.use_ca:
            self.ca = ChannelAttention()
        if self.use_sa:
            self.sa = SpatialAttention(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
        self.use_ca = out_channels >= 64
        self.use_sa = out_channels >= 128
        if self.use_ca:
            self.ca = ChannelAttention()
        if self.use_sa:
            self.sa = SpatialAttention(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(self.GELU(self.bn1(x)))
        out = self.conv2(self.bn2(out))
        out += residual
        out = F.relu(out)
        return out

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fuse = conv1x1(out_channels * 3, out_channels)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        out = torch.cat([branch1, branch3, branch5], dim=1)
        out = self.fuse(out)
        out = F.relu(self.bn(out))
        return out

class DualBranchNetwork(nn.Module):
    def __init__(self, dim=100, size=11, num_classes=9,
                 shared_dims=[32, 64, 128], private_dims=[32, 32, 64],
                 use_dilation=True, use_multi_scale=True, attention_config='auto',
                 feature_dim=256, pred_dim=128, reduce_ratio=4):
        super().__init__()
        self.dim = dim
        self.size = size
        self.reduce_ratio = reduce_ratio
        self.shared_dims = shared_dims
        self.private_dims = private_dims

        self.shared_branch = base_module(dim, 4, growthRate=64, hidden_ch=128, out_ch=shared_dims[2], num_layers=2)
        self.private_branch_1 = base_module(dim, 3, growthRate=64, hidden_ch=128, out_ch=private_dims[2], num_layers=2)  # 修正：使用private_dims[2]
        self.private_branch_2 = base_module(dim, 3, growthRate=64, hidden_ch=128, out_ch=private_dims[2], num_layers=2)  # 修正：使用private_dims[2]

        with torch.no_grad():
            # 获取共享分支输出维度
            dummy_input = torch.randn(2, dim, size, size)
            shared_feat = self.shared_branch(dummy_input)
            _, shared_C, shared_H, shared_W = shared_feat.shape
            self.shared_output_shape = (shared_C, shared_H, shared_W)

            # 获取私有分支输出维度
            private_feat = self.private_branch_1(dummy_input)
            _, private_C, private_H, private_W = private_feat.shape
            self.private_output_shape = (private_C, private_H, private_W)

        print(f"Shared branch output shape: {self.shared_output_shape}")
        print(f"Private branch output shape: {self.private_output_shape}")

        reduced_channels = max(16, shared_C // reduce_ratio)
        self.conv_reducer = nn.Sequential(
            nn.Conv2d(shared_C, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projector = nn.Sequential(
            nn.Linear(reduced_channels, feature_dim // self.reduce_ratio, bias=False),
            nn.BatchNorm1d(feature_dim // self.reduce_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // self.reduce_ratio, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim, affine=False)
        )

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, feature_dim)
        )

    def forward(self, x1, x2):
        B = x1.shape[0]
        feat_shared1 = self.shared_branch(x1)
        feat_shared2 = self.shared_branch(x2)

        reduced1 = self.conv_reducer(feat_shared1).view(B, -1)
        reduced2 = self.conv_reducer(feat_shared2).view(B, -1)

        z1 = self.projector(reduced1)
        z2 = self.projector(reduced2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self.compute_loss(p1, p2, z1, z2)

        feat_private1 = self.private_branch_1(x1)
        feat_private2 = self.private_branch_2(x2)

        return feat_shared1, feat_shared2, feat_private1, feat_private2, loss

    @staticmethod
    def compute_loss(p1, p2, z1, z2):
        z1 = z1.detach()
        z2 = z2.detach()
        loss1 = (1 - F.cosine_similarity(p1, z2, dim=-1)).mean()
        loss2 = (1 - F.cosine_similarity(p2, z1, dim=-1)).mean()
        return (loss1 + loss2) * 0.5

    def get_feature_dims(self):
        """获取特征维度信息"""
        return {
            'shared_dim': self.shared_output_shape[0],
            'private_dim': self.private_output_shape[0]
        }

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, size, prompt_dim):
        super().__init__()
        self.pre_dim = 100
        self.pre_dim_2 = 256
        self.pre_dim_3 = 512
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.pre_dim_2, kernel_size=3, stride=1, padding=1)
        self.projector1 = nn.Sequential(
            nn.Conv2d(self.pre_dim_2, self.pre_dim_3, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.pre_dim_3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projector2 = nn.Sequential(
            # nn.Linear(self.pre_dim_3 + prompt_dim*2, self.pre_dim),
            nn.Linear(self.pre_dim_3, self.pre_dim),
            nn.Dropout(0.5),
            nn.GELU(),
        )
        self.linear = nn.Linear(self.pre_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x = self.projector1(x).flatten(1)
        # x = torch.cat([x, prompt], dim=-1)
        x = self.projector2(x)
        out = self.linear(x)
        return out

    def add_new_classes(self, num_new_classes):
        """添加新类别的分类能力"""
        old_weight = self.linear.weight.data
        old_bias = self.linear.bias.data

        new_num_classes = self.linear.out_features + num_new_classes
        new_linear = nn.Linear(self.pre_dim + self.linear.in_features - self.pre_dim, new_num_classes).to(old_weight.device)

        with torch.no_grad():
            new_linear.weight[:self.linear.out_features] = old_weight
            new_linear.bias[:self.linear.out_features] = old_bias

            for i in range(self.linear.out_features, new_num_classes):
                new_linear.weight[i].normal_(mean=0.0, std=0.01)
                new_linear.bias.data[i] = 0

        self.linear = new_linear

class DualPromptNetwork(nn.Module):
    def __init__(self, dim=100, size=11, num_classes=9,
                 shared_dims=[64, 128, 256], private_dims=[64, 64, 128],
                 prompt_dim=128, device='cuda',
                 contrastive_weight=0.01,
                 # 原型稳定性参数
                 proto_warmup_epochs=5,
                 proto_quality_threshold=0.7):
        super().__init__()
        self.device = device
        self.num_initial_classes = num_classes
        self.prompt_dim = prompt_dim
        self.size = size

        # 原型稳定性参数
        self.proto_warmup_epochs = proto_warmup_epochs
        self.proto_quality_threshold = proto_quality_threshold
        self.current_epoch = 0
        self.contrastive_weight = contrastive_weight

        # 双分支特征提取网络
        self.feature_extractor = DualBranchNetwork(
            dim, size, num_classes,
            shared_dims=shared_dims,
            private_dims=private_dims
        )

        # 动态获取特征维度信息
        feature_dims = self.feature_extractor.get_feature_dims()
        self.shared_dim = feature_dims['shared_dim']
        self.private_dim = feature_dims['private_dim']

        print(f"Detected feature dimensions - Shared: {self.shared_dim}, Private: {self.private_dim}")

        # # 使用新的ImagePromptModule替换原有的DynamicPromptModule
        self.prompt_module = ImagePromptModule(
            real_shared_in_channels=self.shared_dim,
            real_private_in_channels=self.private_dim,
            gen_shared_in_channels=self.shared_dim,
            gen_private_in_channels=self.private_dim,
            num_initial_classes=num_classes,
            prompt_dim=prompt_dim,
            device=device,
            # 传递原型稳定性参数
            proto_warmup_epochs=proto_warmup_epochs,
            proto_quality_threshold=proto_quality_threshold
        )

        # 计算输入分类器的总通道数
        self.other_feat_dim = self.shared_dim * 2 + self.private_dim * 2



        # 分类器
        self.classifier = Classifier(in_channels=self.other_feat_dim, num_classes=num_classes,
                                     size=size, prompt_dim=prompt_dim).to(device)

        self.figure_rand1 = figure_rand(p=0.5)
        self.figure_rand2 = figure_rand(p=0.5)

    def forward(self, x1, x2, class_ids=None):
        x1 = self.figure_rand1(x1)

        # 提取双分支特征
        s1, s2, p1, p2, sim_loss = self.feature_extractor(x1, x2)

        # # 生成提示 - 直接传入4D图像特征
        # if class_ids is not None:
        #     # 训练模式
        #     fused_prompt, distill_loss = self.prompt_module(
        #         s1, p1, s2, p2, class_ids
        #     )
        # else:
        #     # 测试模式
        #     fused_prompt, distill_loss = self.prompt_module(s1, p1, s2, p2)
        #     distill_loss = torch.tensor(0.0).to(self.device)

        # 拼接所有其他特征
        other_features = torch.cat([s1, s2, p1, p2], dim=1)



        # 分类
        # logits = self.classifier(other_features, fused_prompt)
        logits = self.classifier(other_features)

        if class_ids is not None:
            # 训练损失
            cls_loss = F.cross_entropy(logits, class_ids)
            # total_loss = cls_loss + sim_loss * 0.01
            total_loss = cls_loss + sim_loss * self.contrastive_weight

            # 添加详细损失打印
            return logits, total_loss, {
                "total": total_loss.item(),
                "cls": cls_loss.item(),
                "distill": 0,
                "sim": sim_loss.item()
            }

        return logits

    def add_new_classes(self, new_classes):
        """
        完整增量学习支持：
        1. 添加新类别的原型存储空间
        2. 扩展分类器输出层

        参数:
            new_classes (list): 新增类别的ID列表
        """
        if not isinstance(new_classes, (list, tuple)):
            raise ValueError("new_classes must be a list or tuple")
        if len(new_classes) == 0:
            return

        # 更新提示模块和分类器
        # self.prompt_module.add_new_classes(new_classes)
        self.classifier.add_new_classes(len(new_classes))
        self.num_initial_classes += len(new_classes)

        print(f"Added {len(new_classes)} new classes. Total classes now: {self.num_initial_classes}")

    def update_epoch(self, epoch):
        """更新训练周期"""
        self.current_epoch = epoch
        # self.prompt_module.update_epoch(epoch)

    def freeze_for_incremental_learning(self, old_class_range=None, new_class_range=None):
        """冻结模型参数用于增量学习，支持按类别范围冻结分类头

        Args:
            old_class_range: 旧类别范围，例如 (0, 5) 表示冻结0-5类
            new_class_range: 新类别范围，例如 (6, 10) 表示训练6-10类
        """
        # 冻结特征提取器
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        # 冻结特征提取器
        # for param in self.prompt_module.parameters():
        #     param.requires_grad = True

        # 分类器特殊处理 - 按类别范围冻结
        if old_class_range is not None and new_class_range is not None:
            self._freeze_classifier_by_range(old_class_range, new_class_range)
        else:
            # 默认行为：只冻结特征处理部分，不冻结分类头
            for name, param in self.classifier.named_parameters():
                if 'linear' not in name:
                    param.requires_grad = True

        print("Model frozen for incremental learning")

    def _freeze_classifier_by_range(self, old_class_range, new_class_range):
        """按类别范围冻结分类器权重"""
        old_start, old_end = old_class_range
        new_start, new_end = new_class_range

        # 获取分类器线性层的权重和偏置
        classifier_linear = self.classifier.linear
        total_classes = classifier_linear.out_features

        # 验证范围有效性
        if old_end >= total_classes or new_end >= total_classes:
            raise ValueError(f"类别范围超出模型当前类别数 {total_classes}")

        # 创建冻结掩码
        with torch.no_grad():
            # 冻结旧类别对应的权重
            for i in range(old_start, old_end + 1):
                classifier_linear.weight[i].requires_grad = False
                if classifier_linear.bias is not None:
                    classifier_linear.bias[i].requires_grad = False

            # 确保新类别可训练
            for i in range(new_start, new_end + 1):
                classifier_linear.weight[i].requires_grad = True
                if classifier_linear.bias is not None:
                    classifier_linear.bias[i].requires_grad = True

        print(f"冻结分类头: 类别 {old_start}-{old_end}")
        print(f"训练分类头: 类别 {new_start}-{new_end}")

    def get_classifier_frozen_status(self):
        """获取分类器冻结状态"""
        classifier_linear = self.classifier.linear
        frozen_indices = []
        trainable_indices = []

        for i in range(classifier_linear.out_features):
            if not classifier_linear.weight[i].requires_grad:
                frozen_indices.append(i)
            else:
                trainable_indices.append(i)

        return {
            'frozen_classes': frozen_indices,
            'trainable_classes': trainable_indices,
            'total_classes': classifier_linear.out_features
        }

    def get_features(self, x1, x2):
        """简单的特征提取：使用分类前的拼接特征"""
        # 提取双分支特征
        s1, s2, p1, p2, _ = self.feature_extractor(x1, x2)

        # 拼接所有特征
        features = torch.cat([s1, s2, p1, p2], dim=1)

        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        return features

def print_model_params(model, model_name="DualPromptNetwork"):
    """打印模型各组成部分的参数量"""
    total_params = 0
    print(f"\n{' Model Parameter Analysis ':=^80}")
    print(f"{model_name} Structure:")

    def print_module_params(module, prefix=""):
        nonlocal total_params
        module_name = prefix + module.__class__.__name__
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {module_name:<40} | {params:,} parameters")
            total_params += params
        for name, child in module.named_children():
            print_module_params(child, prefix=prefix + name + ".")

    print_module_params(model)
    print("\n" + "-" * 80)
    print(f"  {'Total Parameters':<40} | {total_params:,}")
    print(f"  {'Trainable Parameters':<40} | {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 参数设置
    dim = 100
    size = 11
    num_classes = 9
    batch_size = 4

    # 模型初始化
    model = DualPromptNetwork(
        dim=dim,
        size=size,
        num_classes=num_classes,
        prompt_dim=128,
        device=device
    ).to(device)

    print_model_params(model)

    # 创建模拟输入数据
    x1 = torch.randn(batch_size, dim, size, size).to(device)
    x2 = torch.randn(batch_size, dim, size, size).to(device)

    # 测试训练模式
    print("\n=== Testing Train Mode ===")
    model.train()
    class_ids = torch.randint(0, num_classes, (batch_size,)).to(device)

    logits, total_loss, loss_dict = model(x1, x2, class_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss value: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

    # 测试测试模式
    print("\n=== Testing Eval Mode ===")
    model.eval()
    with torch.no_grad():
        logits = model(x1, x2)
        print(f"Logits shape: {logits.shape}")

    # 测试增量学习功能
    print("\n=== Testing Incremental Learning ===")
    new_classes = [10, 11, 12]
    model.add_new_classes(new_classes)
    print(f"Classifier output dim after adding classes: {model.classifier.linear.out_features}")
    print_model_params(model)

    # 再次运行测试模式
    model.eval()
    with torch.no_grad():
        logits = model(x1, x2)
        print(f"Logits shape after adding classes: {logits.shape}")