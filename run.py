import os
from pathlib import Path
from pkg_resources import packaging
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from configs.train_config import TrainConfig
from configs.test_config import TestConfig
from utils.dataset_utils import DatasetGenerator, DataCollator
from utils.train_utils import train
from utils.test_utils import test


def get_checkpoint_save_dir(config):
    folder_name = (
    config.checkpoint_root_folder
    + "/"
    + config.checkpoint_folder
    + "-"
    + config.model_name
    )
    save_dir = Path.cwd() / folder_name

    return save_dir


def main(args):
    if args.run_mode == "train":
        config = TrainConfig() 
    else:
        config = TestConfig()

    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    
    if config.enable_fsdp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            print(f"Clearing GPU cache for all ranks")
        torch.cuda.empty_cache()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
        if rank == 0:
            print(f"--> Running with torch dist debug set to detail")
    
    gradient_accumulation_steps = config.batch_size_training // config.micro_batch_size
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ["[상승]", "[하락]"]})
    
    model_config = AutoConfig.from_pretrained(config.model_name)
    
    if config.enable_fsdp and config.low_cpu_fsdp:
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
        else:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    if config.enable_fsdp:
        model = FSDP(model,
                     sharding_strategy=config.sharding_strategy,
                     device_id=torch.cuda.current_device(),
                     limit_all_gathers=True,
                     sync_module_states=config.low_cpu_fsdp,
                     param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
                     if config.low_cpu_fsdp and rank!=0 else None)
    else:
        model.to("cuda")
        
    if args.run_mode == "train":
        dataset = DatasetGenerator(tokenizer, config, dataset_type="train")
    else:
        dataset = DatasetGenerator(tokenizer, config, dataset_type="eval")
    data_collator = DataCollator(tokenizer)
    
    train_sampler = None
    val_sampler = None
    if config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if args.run_mode == "eval":  
            val_sampler = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
    
    # Create DataLoaders for the training and validation dataset
    if args.run_mode == "train":    
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size_training,
            num_workers=config.num_workers_dataloader,
            pin_memory=True,
            sampler=train_sampler if train_sampler else None,
            drop_last=True,
            collate_fn=data_collator,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.val_batch_size,
            num_workers=config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=data_collator,
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    checkpoint_dir = get_checkpoint_save_dir(config)
    
    if args.run_mode == "train":
        train_results = train(
            model,
            dataloader,
            optimizer,
            scheduler,
            gradient_accumulation_steps,
            config,
            checkpoint_dir,
            local_rank if config.enable_fsdp else None,
            rank if config.enable_fsdp else None
        )
    else:   
        model = AutoModelForCausalLM.from_config(model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
        model = model.to(torch.device("cuda:0"))
        test_results = test(model, config, dataloader, tokenizer, "cuda:0", 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_mode", type=str, choices=["train", "eval"], default="train")
    args = parser.parse_args() 
    main(args)