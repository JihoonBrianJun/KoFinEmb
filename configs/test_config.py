from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy

@dataclass
class TestConfig:
    eval_data_path: str = "dataset/eval.csv"
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    model_name: str = "EleutherAI/polyglot-ko-3.8b"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    num_workers_dataloader: int = 1
    seed: int = 2
    val_batch_size: int = 1
    save_model: bool = True
    checkpoint_root_folder: str = "model_checkpoints"
    checkpoint_folder: str = "KoFinEmbInitial"
    save_optimizer: bool = False
    max_length: int = 512