"""
独立测试脚本：加载训练好的分类模型权重，在原始 patch 测试集上评估 OA / AA / Kappa，
并将分类结果图保存为 mat 文件。

用法示例：
    # 单个模型（base 阶段）：
    python test.py --device cuda:0 --ckpt ./checkp/base_classifier_20250728_024041_0.8992.pth
    # 指定类别：
    python test.py --device cuda:0 --ckpt ./checkp/base_classifier_20250728_024041_0.8992.pth --test_classes 0 1 2 3
    # 多阶段类增量评估（依次传入各 session 的权重，按已见类别测试）：
    python test.py --device cuda:0 \
        --ckpt ./checkp/base_classifier_20250728_024041_0.8992.pth \
               ./checkp/base_classifier_20250725_084919_0.9841.pth \
               ./checkp/base_classifier_20250726_023137_0.9175.pth

说明：
    - 测试集直接取自 t1 数据集中的原始 patch（不使用生成样本，不依赖 VQDiT 模型）。
    - 多权重模式下遵循类增量协议：每个 session 的模型在其"已见过的所有类别"
      （即 0 ~ num_classes-1，num_classes 由 classifier.linear.weight 推断）上评估。
    - 首次运行会先生成并缓存 t1 数据集（datasets/cache/t1_*.pth），耗时较长；之后直接加载缓存。
"""

import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from args import parse_args
from base_training import prepare_dataset, evaluate_classifier
from Class_Increment import DualPromptNetwork


def build_model(args, num_classes):
    model = DualPromptNetwork(
        dim=args.in_channels,
        size=args.patch_size,
        num_classes=num_classes,
        shared_dims=args.shared_dims,
        private_dims=args.private_dims,
        prompt_dim=args.prompt_dim,
        device=args.device,
        proto_warmup_epochs=args.proto_warmup_epochs,
        proto_quality_threshold=args.proto_quality_threshold,
        proto_decay_start=args.proto_decay_start,
        proto_decay_end=args.proto_decay_end,
        proto_decay_steps=args.proto_decay_steps,
    ).to(args.device)
    return model


def load_model(args, ckpt_path):
    """加载权重并返回模型，num_classes 由 classifier.linear.weight 推断"""
    state_dict = torch.load(ckpt_path, map_location=args.device)
    num_classes = state_dict['classifier.linear.weight'].shape[0]

    model = build_model(args, num_classes)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # 兼容少量参数不匹配的情况（与训练代码一致的处理方式）
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.size() == model_state[k].size()}
        model_state.update(filtered)
        model.load_state_dict(model_state)
        print(f"⚠️ 部分参数不匹配，已加载 {len(filtered)}/{len(model_state)} 个参数")
    model.eval()
    print(f"已加载模型: {ckpt_path} (num_classes={num_classes})")
    return model, num_classes


def build_test_dataset(t1, test_classes):
    """从 t1 中抽取指定类别的原始 patch 测试集"""
    labels_all = t1.tensors[2].view(-1).tolist()
    indices = [i for i, lbl in enumerate(labels_all) if lbl in test_classes]

    test_dataset = TensorDataset(
        t1.tensors[0][indices],            # images
        t1.tensors[1][indices],            # patches
        t1.tensors[2][indices].view(-1),   # labels
        t1.tensors[3][indices].view(-1),   # ys
        t1.tensors[4][indices].view(-1),   # xs
    )
    print(f"测试集大小: {len(test_dataset)}，类别: {test_classes}")
    return test_dataset


def evaluate_one(args, t1, ckpt_path, test_classes):
    """评估单个权重，返回 (num_classes, oa, aa, kappa)"""
    model, num_classes = load_model(args, ckpt_path)
    test_dataset = build_test_dataset(t1, test_classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    oa, aa, kappa, mat_file = evaluate_classifier(model, test_loader, args.device, 'test')
    print(f"结果: OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}")
    print(f"分类结果图: {mat_file}\n")
    return num_classes, oa, aa, kappa


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert args.ckpt and all(os.path.exists(p) for p in args.ckpt), \
        f"模型权重不存在: {args.ckpt}"

    # 1. 准备 t1 数据集（有缓存则直接加载）
    t1, full_dataset = prepare_dataset(args)

    # 2. 逐个 session 评估
    results = []
    for idx, ckpt_path in enumerate(args.ckpt):
        # 多权重模式：类增量协议，每个 session 测已见过的所有类别
        # 单权重模式：默认测基础类别，或使用 --test_classes 指定
        if len(args.ckpt) > 1 or args.test_classes is None:
            state_dict = torch.load(ckpt_path, map_location=args.device)
            num_classes = state_dict['classifier.linear.weight'].shape[0]
            test_classes = args.test_classes if args.test_classes else list(range(num_classes))
        else:
            test_classes = args.test_classes

        print(f"\n{'=' * 60}\nSession {idx}: {ckpt_path}")
        num_classes, oa, aa, kappa = evaluate_one(args, t1, ckpt_path, test_classes)
        results.append((idx, num_classes, oa, aa, kappa))

    # 3. 汇总表
    if len(results) > 1:
        print("=" * 60)
        print("各阶段汇总:")
        print(f"{'Session':<10}{'类别数':<8}{'OA':<10}{'AA':<10}{'Kappa':<10}")
        for idx, num_classes, oa, aa, kappa in results:
            print(f"{idx:<10}{num_classes:<8}{oa:<10.4f}{aa:<10.4f}{kappa:<10.4f}")


if __name__ == "__main__":
    main()
