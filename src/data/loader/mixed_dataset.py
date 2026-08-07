from datasets.distributed import split_dataset_by_node

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.dataset.hf_datasets import interleave_datasets
from src.utils.basic_utils import print_master
import torch

def init_mixed_dataset(dataset_config, model_args, data_args, training_args):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    dataset_names = list(dataset_config.keys())
    weights = [d['weight'] for d in dataset_config.values()]
    train_datasets = []
    for subset_config in dataset_config.values():
        train_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **subset_config)
        train_datasets.append(train_dataset)

    # 采样概率 ∝ weight * num_rows ** alpha。alpha 必须在这里算而不是在上面的循环之前，
    # 因为 num_rows 要等子集实例化（含截断到 num_sample_per_subset）之后才确定。
    #
    # alpha=0 是历史行为：每个子集恒定占 1/N，与大小无关。修好 all_exhausted 的循环重采样后，
    # 这意味着小子集会被反复重读、大子集则有大半永远读不到 —— 全量 20 子集配置下
    # VOC2007(7,844) 一个 epoch 过 6.8 遍，而 VisDial(123,287) 只过 0.43 遍。
    # alpha=1 让每条样本等概率，各子集恰好各过一遍。alpha=0.5 把极差从 15.7 倍压到 4.0 倍。
    alpha = getattr(training_args, "dataset_size_alpha", 0.0)
    scaled = [w * (d.num_rows ** alpha) for w, d in zip(weights, train_datasets)]
    w_sum = sum(scaled)
    probs = [w / w_sum for w in scaled]

    total_num_rows = sum([d.num_rows for d in train_datasets])
    print_master(f"\nDataset mixture (dataset_size_alpha={alpha}):")
    for data_idx, name in enumerate(dataset_names):
        num_rows = train_datasets[data_idx].num_rows
        parser = dataset_config[name].get('dataset_parser', 'n/a')
        # 遍数：一个 epoch 按投喂 total_num_rows 条计，该子集被完整读过的次数。
        passes = total_num_rows * probs[data_idx] / num_rows if num_rows else float('nan')
        print_master(f"\t\tDataset#{data_idx} (dataset_parser={parser}): {name}, "
                     f"num_rows={num_rows}, prob={probs[data_idx] * 100.0:.2f}%, passes_per_epoch={passes:.2f}")

    # Handle Deprecation
    if training_args.homogeneous_batch_size_per_device == 0 and training_args.interleave_batch_size != 0:
        print_master("WARNING: `interleave_batch_size` is deprecated. Please use `homogeneous_batch_size_per_device`.")
        training_args.homogeneous_batch_size_per_device = training_args.interleave_batch_size

    if training_args.homogeneous_batch_size_per_device and training_args.homogeneous_batch_size_per_device <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.homogeneous_batch_size_per_device
    else:
        interleave_batch_size = training_args.homogeneous_batch_size_per_device * world_size

    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={world_size}"
                 f"\n\t\ttotal num rows={total_num_rows}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={total_num_rows/(training_args.per_device_train_batch_size * world_size)}"
                 f"\n\t\thomogeneous_batch_size_per_device={training_args.homogeneous_batch_size_per_device}"
                 f"\n\t\tinterleave_batch_size (global)={interleave_batch_size}"
                 )
    assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
        f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    if len(train_datasets) > 1:
        train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)
    else:
        train_dataset = train_datasets[0]
    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)

    return train_dataset

