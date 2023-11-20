import os
import time
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import torch.distributed as dist


def train(model, train_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, config, checkpoint_dir, local_rank=None, rank=None):
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
            for key in batch.keys():
                if config.enable_fsdp:
                    input_ids = batch["input_ids"].to(local_rank)
                    labels = batch["labels"].to(local_rank)
                    attention_mask = batch["attention_mask"].to(local_rank)
                else:
                    input_ids = batch["input_ids"].to('cuda:0')
                    labels = batch["labels"].to('cuda:0')
                    attention_mask = batch["attention_mask"].to('cuda:0')
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            lm_logits = outputs[0]   # (BS, Max_Seq_Len, vocab_size)
            
            loss = None
            
            logits = lm_logits[:,:-1,:].contiguous().view(-1, lm_logits.size()[-1])   # (BS * (Max_Seq_Len-1), vocab_size)
            labels = labels[:,1:].contiguous().view(-1).to(logits.device)   # (BS * (Max_Seq_Len-1))
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits, labels)                
        
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
                    
                states = model.state_dict()
                if rank == 0:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(states, os.path.join(checkpoint_dir, "model.pt"))
            
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
    
    return results