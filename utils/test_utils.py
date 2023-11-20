from tqdm import tqdm
import torch
import torch.distributed as dist


def test(model, config, eval_dataloader, tokenizer, local_rank, world_size):
    results = []
    model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="Testing Epoch")):
        with torch.no_grad():
            if type(model).__name__ == "DistributedDataParallel":
                outputs = model.module.generate(
                    input_ids=batch["input_ids"].to(local_rank),
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
                    input_ids=batch["input_ids"].to(local_rank),
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

            result = tokenizer.decode(outputs[0].view(-1))
            print(result)
            results.append(result)
    
    if world_size>1:
        gathered_results = [list() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        for i in range(1, world_size):
            gathered_results[0].extend(gathered_results[i])
        return gathered_results[0]
    else:
        return results