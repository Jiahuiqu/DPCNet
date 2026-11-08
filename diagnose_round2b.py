"""诊断 round2：新类 10/11/13/14 的预测去向 + 训练/测试样本的类中心相似度"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from args import parse_args
from Class_Increment import DualPromptNetwork
import glob
import numpy as np

args = parse_args()
device = args.device

# ===== 1. 加载当前最佳 round2 权重，看混淆 =====
ckpts = sorted(glob.glob('checkpoints/incremental_classifier_round2_*.pth'))
print("round2 权重:", ckpts)
if ckpts:
    from Class_Increment import DualPromptNetwork as DPN
    model = DPN(dim=args.in_channels, size=args.patch_size, num_classes=args.base_classes,
                shared_dims=args.shared_dims, private_dims=args.private_dims, prompt_dim=args.prompt_dim,
                device=device,
                proto_warmup_epochs=args.proto_warmup_epochs,
                proto_quality_threshold=args.proto_quality_threshold,
                proto_decay_start=args.proto_decay_start, proto_decay_end=args.proto_decay_end,
                proto_decay_steps=args.proto_decay_steps).to(device)
    model.set_round_idx(1)
    model.prompt_module.prepare_for_incremental_learning([0, 1, 2, 3, 4])
    model.add_new_classes([5, 6, 7, 8, 9])
    # 先加载 round1 权重，再扩展到 15 类，然后加载 round2 权重（15类，严格匹配）
    r1 = sorted(glob.glob('checkpoints/incremental_classifier_round1_*.pth'))
    model.load_state_dict(torch.load(r1[-1], map_location=device))
    model.set_round_idx(2)
    model.prompt_module.prepare_for_incremental_learning(list(range(10)))
    model.add_new_classes([10, 11, 12, 13, 14])
    model.load_state_dict(torch.load(ckpts[-1], map_location=device))
    model.eval()

    inc = torch.load('datasets/cache/t_incremental_houston13_round2_classes10to14_patch9.pth')
    loader = DataLoader(inc, batch_size=128, shuffle=False, num_workers=0)
    cm = np.zeros((15, 15), dtype=int)
    with torch.no_grad():
        for images, patches, labels, ys, xs in loader:
            logits = model(images.to(device), patches.to(device))
            pred = logits.argmax(1).cpu().numpy()
            for t, p in zip(labels.cpu().numpy(), pred):
                cm[t, p] += 1
    print("\n混淆（行=真实，列=预测；只显示错分大户）:")
    for t in range(15):
        row = cm[t].copy()
        row[t] = 0
        top = np.argsort(row)[::-1][:2]
        tot = cm[t].sum()
        if tot > 0 and cm[t, t] / tot < 0.9:
            print(f"  真实{t} (共{tot}): 正确{cm[t,t]/tot:.1%}, "
                  f"错→{top[0]}:{row[top[0]]/tot:.1%}, 错→{top[1]}:{row[top[1]]/tot:.1%}")

# ===== 2. t10 训练样本类中心 vs 测试集类中心 的余弦相似度 =====
t10 = torch.load('datasets/cache/t10_houston13_round2_classes10to14_patch9_local3_global4_gen3.pth')
inc = torch.load('datasets/cache/t_incremental_houston13_round2_classes10to14_patch9.pth')

def class_means(ds, label_idx, x_idx=0):
    xs, ls = ds.tensors[x_idx], ds.tensors[label_idx].view(-1)
    return {c: xs[ls == c].mean(0) for c in ls.unique().tolist()}

tm = class_means(t10, 2, 0)     # t10 的 x1 类中心
vm = class_means(inc, 2, 0)     # 测试集的 x1 类中心
print("\n训练样本类中心 与 测试集同类的余弦相似度（越接近1越像真实同类）:")
for c in sorted(tm):
    if c in vm:
        sim = F.cosine_similarity(tm[c].flatten().unsqueeze(0), vm[c].flatten().unsqueeze(0)).item()
        print(f"  类 {c}: {sim:.4f}")
