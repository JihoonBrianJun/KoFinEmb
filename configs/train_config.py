from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy

@dataclass
class TrainConfig:
    train_data_path: str = "dataset/train.csv"
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    optimizer: str = "AdamW"
    model_name: str = "EleutherAI/polyglot-ko-3.8b"
    enable_fsdp: bool = True
    low_cpu_fsdp: bool = True
    batch_size_training: int = 4
    num_epochs: int = 1
    num_workers_dataloader: int = 1
    gamma: float = 0.85
    seed: int = 2
    micro_batch_size: int = 4
    save_model: bool = True
    checkpoint_root_folder: str = "model_checkpoints"
    checkpoint_folder: str = "KoFinEmbInitial"
    save_optimizer: bool = False
    lr: float = 2e-5
    max_length: int = 512