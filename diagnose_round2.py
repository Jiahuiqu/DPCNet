""""诊断实验：round1 权重在 round2 测试集上的表现，区分评估状态问题 vs 训练破坏问题"""
import torch
from torch.utils.data import DataLoader
from args import parse_args
from Class_Increment import DualPromptNetwork
from increment_training import evaluate_classifier

args = parse_args()

model = DualPromptNetwork(
    dim=args.in_channels, size=args.patch_size, num_classes=args.base_classes,
    shared_dims=args.shared_dims, private_dims=args.private_dims, prompt_dim=args.prompt_dim,
    device=args.device,
    proto_warmup_epochs=args.proto_warmup_epochs,
    proto_quality_threshold=args.proto_quality_threshold,
    proto_decay_start=args.proto_decay_start,
    proto_decay_end=args.proto_decay_end,
    proto_decay_steps=args.proto_decay_steps
).to(args.device)

# ===== 实验 A：round1 权重 + 轮次1语义 =====
model.set_round_idx(1)
model.prompt_module.prepare_for_incremental_learning([0, 1, 2, 3, 4])
model.add_new_classes([5, 6, 7, 8, 9])
sd = torch.load('checkpoints/incremental_classifier_round1_20261107_205512_0.8345.pth',
                map_location=args.device)
model.load_state_dict(sd)
print("\n========== 实验 A：round1权重(轮次1语义) 在15类测试集上 ==========")
inc = torch.load('datasets/cache/t_incremental_houston13_round2_classes10to14_patch9.pth')
loader = DataLoader(inc, batch_size=args.batch_size, shuffle=False, num_workers=0)
evaluate_classifier(model, loader, args.device, 'diagA', round_idx=1)

# ===== 实验 B：同一权重，切换到轮次2语义（round2 训练前的真实状态）=====
model.set_round_idx(2)
model.prompt_module.prepare_for_incremental_learning(list(range(10)))
model.add_new_classes([10, 11, 12, 13, 14])
print("\n========== 实验 B：同一权重(轮次2语义、扩到15类、新行随机) ==========")
evaluate_classifier(model, loader, args.device, 'diagB', round_idx=2)
