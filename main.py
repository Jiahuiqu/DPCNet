import torch
import random
from args import parse_args
import numpy as np
from base_training import base_training, prepare_dataset
from increment_training import  incremental_training


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 准备阶段
    t1, full_dataset = prepare_dataset(args)

    # 第一阶段
    vqdit_model, classifier_model, t4, best_replay_dataset, base_replay_path, initial_class_accuracies = base_training(args, t1, full_dataset)
    torch.cuda.empty_cache()
    t_incremental = t4

    # 增量阶段
    for round_idx in range(args.incremental_rounds):
        vqdit_model, classifier_model, t_incremental, best_replay_dataset, initial_class_accuracies = incremental_training(
            args, vqdit_model, classifier_model, t1, full_dataset, round_idx+1,
            t_incremental, best_replay_dataset, initial_class_accuracies)

        torch.cuda.empty_cache()

    print("🎉 训练完成!")




if __name__ == "__main__":
    main()
