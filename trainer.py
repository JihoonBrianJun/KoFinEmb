import os
import time
import datetime
import yaml
import copy
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Sequence
from pathlib import Path
from pkg_resources import packaging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, 
                                    StateDictType,
                                    FullStateDictConfig)
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from argparse import ArgumentParser


@dataclass
class Config:
    train_data_path: str = "dataset/train.csv"
    eval_data_path: str = "dataset/eval.csv"
    checkpoint_type: StateDictType = "StateDictType.SHARDED_STATE_DICT"
    optimizer: str = "AdamW"
    model_name: str = "EleutherAI/polyglot-ko-5.8b"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 2
    num_epochs: int = 1
    num_workers_dataloader: int = 1
    gamma: float = 0.85
    seed: int = 2
    val_batch_size: int = 1
    micro_batch_size: int = 2
    save_model: bool = False
    dist_checkpoint_root_folder: str = "model_checkpoints"
    dist_checkpoint_folder: str = "KoFinEmbInitial"
    save_optimizer: bool = False
    lr: float = 2e-5
    max_length: int = 512

class DatasetGenerator(Dataset):
    def __init__(self, tokenizer, config, dataset_type):
        super(DatasetGenerator, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100
        
        if dataset_type == "train":
            data_path = config.train_data_path
        else:
            data_path = config.eval_data_path
        
        instruction = """주식 종목 토론방에 [제목]의 글이 올라온 다음 날, 주가는 어떻게 되었을까?
        [상승] 혹은 [하락] 중 하나로 답하여라. 
        
        [제목]: 장마감 이후 올라온 공시 자료를 보니, 분기 실적이 예상치를 크게 밑도네요ㅠ
        [정답]: [하락]
        
        [제목]: 간밤에 미국 연준의 깜짝 금리 인하 결정에 힘입어 미국 증시 폭등 중이네요. 내일 한국 증시도 기대해 봅니다!
        [정답]: [상승]
        """
        
        df = pd.read_csv(data_path)
        df["title_with_instruction"] = df["title"].apply(lambda x: f"{instruction}\n\n[제목]: {x}\n[정답]: ")
        df["classify_label"] = df["label"].apply(lambda x: f"{x}{self.tokenizer.eos_token}")
        
        input_list = df["title_with_instruction"].tolist()
        if dataset_type == "train":
            target_list = df["classify_label"].tolist()
        else:
            target_list = None

        print("Tokenizing inputs... This may take some time...")
        if target_list != None:
            example_list = [input+target for input, target in zip(input_list, target_list)]
        else:
            example_list = input_list
        
        example_tokenized, input_tokenized = [self.tokenize(data_list) 
                                            for data_list in (example_list, input_list)]
        input_ids = example_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        print("Tokenizing All done!")
        
        for i in tqdm(range(len(labels))):
            input_len = input_tokenized["input_ids_lens"][i] 
            labels[i][:input_len] = self.IGNORE_INDEX
        
        data_dict = dict(input_ids=input_ids, labels=labels)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.dataset_type = dataset_type
        print(f"Example dataset..\nInput: {input_list[0]}\ninput_ids: {self.input_ids[0]}\nlabels: {self.labels[0]}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], dataset_type=self.dataset_type)
    
    def tokenize(self, data_list):
        tokenized_list = []
        for data in data_list:
            tokenized_list.append(self.tokenizer(data,
                                                 return_tensors="pt",
                                                 padding="longest",
                                                 max_length = self.config.max_length,
                                                 truncation=True))
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
                          for tokenized in tokenized_list]
        
        return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


@dataclass
class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, dataset_type = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "dataset_type"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=self.tokenizer.pad_token_id)
        
        return dict(input_ids=input_ids, 
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


# class PolyglotClassifier(AutoModelForCausalLM):
#     def __init__(self, args):
#         self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
#     def forward(self, input_ids, labels=None, attention_mask=None):
#         model_outputs = self.model(input_ids,
#                                    attention_mask=attention_mask,
#                                    output_attentions=True,
#                                    output_hidden_states=True)
#         lm_logits = model_outputs[0]   # (BS, Max_Seq_Len, vocab_size)
        
#         loss = None
#         if labels is not None:
#             logits = lm_logits[:,:-1,:].view(-1, lm_logits.size()[-1])   # (BS * (Max_Seq_Len-1), vocab_size)
#             labels = labels[:,1:].view(-1).to(logits.device)   # (BS * (Max_Seq_Len-1))
#             loss_function = CrossEntropyLoss()
#             loss = loss_function(logits, labels)
        
#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=model_outputs.past_key_values,
#             hidden_states=model_outputs.hidden_states,
#             attentions=model_outputs.attentions,
#         )


def train(model, train_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, config, local_rank=None, rank=None):
    if config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    
    train_perp = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = dict()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader,colour="blue", desc=f"Training Epoch{epoch}")):
            new_batch = dict()
            for key in batch.keys():
                if config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    new_batch[key] = batch[key].to('cuda:0')
                    print(f"{key} device: {new_batch[key].device}")
                batch_size = batch["input_ids"].shape[0]
                
                # outputs = model(**batch)
                
                outputs = model(**new_batch)
                lm_logits = outputs[0]   # (BS, Max_Seq_Len, vocab_size)
                loss = None
                labels = batch["labels"]
                
                logits = lm_logits[:,:-1,:].contiguous().view(-1, lm_logits.size()[-1])   # (BS * (Max_Seq_Len-1), vocab_size)
            
                labels = labels[:,1:].contiguous().view(-1).to(logits.device)   # (BS * (Max_Seq_Len-1))
                loss_function = CrossEntropyLoss()
                loss = loss_function(logits, labels)                
            
                # loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                
                loss.backward()
                    
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if config.enable_fsdp:
                    if rank == 0:
                        print(f"\n step {step} is completed and loss is {loss.detach().float()}")
                else:
                    print(f"\n step {step} is completed and loss is {loss.detach().float()}")
        
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        
        if torch.cuda.device_count() > 1 and config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        
        train_epoch_loss = total_loss / len(train_dataloader)
        if config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_perp.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        lr_scheduler.step()
        
        if config.run_validation:
            print("Epoch ended")
            checkpoint_start_time = time.perf_counter()
            if config.save_model:
                if config.enable_fsdp:
                    dist.barrier()
                    
                if config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    save_model_checkpoint(model, optimizer, rank, config, epoch=epoch)
                elif config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                    print("=====================================================")
                    save_model_and_optimizer_sharded(model, rank, config)
                    if config.save_optimizer:
                        save_model_and_optimizer_sharded(model, rank, config, optim=optimizer)
                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        print("=====================================================")
                if config.save_optimizer:
                    save_optimizer_checkpoint(model, optimizer, rank, config, epoch=epoch)
                    print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                    print("=====================================================")     
                
                if config.enable_fsdp:
                    dist.barrier()
            
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
        
        if config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
    
    avg_epoch_time = sum(epoch_times)/ len(epoch_times) 
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times)   
    avg_train_prep = sum(train_perp)/len(train_perp)
    avg_train_loss = sum(train_loss)/len(train_loss)    
    
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    if config.enable_fsdp:
        save_train_params(config, rank)
    
    return results


def test(model, config, eval_dataloader, tokenizer, world_size):
    results = []
    model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="Testing Epoch")):
        with torch.no_grad():
            if type(model).__name__ == "DistributedDataParallel":
                outputs = model.module.generate(
                    input_ids=batch["input_ids"],
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    max_length=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )
            else:
                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    max_length=config.max_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True
                )                  

            results.append(tokenizer.decode(outputs[0]))
    
    if world_size>1:
        gathered_results = [list() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        for i in range(1, world_size):
            gathered_results[0].extend(gathered_results[i])
        return gathered_results[0]
    else:
        return results
    

def save_model_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """saving model via rank0 cpu streaming and full_state_dict"""

    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)
        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )

def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""
   
    print(f"--> optim state call on rank {rank}\n")
    # pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")
        torch.save(optim_state, opt_save_full_path)
        print(f"--> saved {opt_save_full_path} to disk")

def save_train_params(config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    config.dist_checkpoint_root_folder
    + "/"
    + config.dist_checkpoint_folder
    + "-"
    + config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")
    

def main(args):    
    config = Config()
    
    now = datetime.datetime.now()
    date_dir = os.path.join(os.getcwd(),'output',f'{now.year}-{now.month}-{now.day}')
    time_dir = os.path.join(date_dir, f'{now.hour}-{now.minute}-{now.second}')
    score_dir = os.path.join(time_dir, "score.txt")    

    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    
    if config.enable_fsdp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            if not os.path.isdir(time_dir):
                os.makedirs(time_dir)
            with open(score_dir, "w") as f:
                f.write("")
    else:
        if torch.cuda.current_device()==0:
            if not os.path.isdir(date_dir):
                os.mkdir(date_dir)
            if not os.path.isdir(time_dir):
                os.mkdir(time_dir)
            with open(score_dir, "w") as f:
                f.write("")   

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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ["[상승]", "[하락]"]})
    
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    if config.enable_fsdp and config.low_cpu_fsdp:
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            # model = PolyglotClassifier.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        else:
            with torch.device("meta"):
                # model = PolyglotClassifier(model_config)
                model = AutoModelForCausalLM(model_config)
    else:
        # model = PolyglotClassifier.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    if config.enable_fsdp:
        model = FSDP(model)
    else:
        model.to("cuda")
        
    dataset_train = DatasetGenerator(tokenizer, config, dataset_type="train")
    data_collator_train = DataCollator(tokenizer)
    
    dataset_eval = DatasetGenerator(tokenizer, config, dataset_type="eval")
    data_collator_eval = DataCollator(tokenizer)
    
    train_sampler = None
    val_sampler = None
    if config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if config.run_validation:  
            val_sampler = DistributedSampler(
                dataset_eval,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
    
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size_training,
        num_workers=config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=data_collator_train,
    )

    if config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=config.val_batch_size,
            num_workers=config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=data_collator_eval,
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    results = train(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        config,
        local_rank if config.enable_fsdp else None,
        rank if config.enable_fsdp else None,
    )
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/polyglot-ko-5.8b")
    args = parser.parse_args() 
    main(args)