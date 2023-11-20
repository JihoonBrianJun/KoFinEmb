from tqdm import tqdm
import torch
import torch.distributed as dist


def test(model, config, eval_dataloader, tokenizer, local_rank, world_size):
    results = []
    correct = 0
    
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

            if "[정답]: " in result and "<|endoftext|>" in result:
                model_answer = result.split("[정답]: ")[-1].split("<|endoftext|>")[0]
            else:
                model_answer = None

            gold_answer = batch["gold_answers"][0]
            if model_answer == gold_answer:
                correct += 1
            
            print(result + f"\nGold Answer: {gold_answer}")
            results.append(result + f"\nGold Answer: {gold_answer}")
    
    print(f"Correct: {correct} out of {step+1}\nCorrect Rate: {correct / (step+1) * 100}%")
    
    if world_size>1:
        gathered_results = [list() for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)
        for i in range(1, world_size):
            gathered_results[0].extend(gathered_results[i])
        return gathered_results[0]
    else:
        return results