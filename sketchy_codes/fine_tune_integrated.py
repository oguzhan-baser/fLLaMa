import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig
import json

os.environ["WANDB_DISABLED"] = "true"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_LAUNCH_TIMEOUT"] = "1200"
os.environ["NCCL_DEBUG"] = "INFO"
hf_token = os.environ["HF_TOKEN"] 


def append_loss_to_file(file_path, new_loss):
    # Load existing loss data if file exists, otherwise create a new list
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            loss_data = json.load(f)
    else:
        loss_data = []

    # Append new loss value
    loss_data.append(new_loss)

    # Save updated loss data back to file
    with open(file_path, "w") as f:
        json.dump(loss_data, f, indent=4)
        
###############################
# Helper: Partition the dataset
###############################
def partition_dataset(dataset, rank, world_size):
    total_samples = len(dataset)
    samples_per_client = total_samples // (world_size+1)
    start = rank * samples_per_client
    # Ensure the last client picks up any remaining samples.
    end = total_samples if rank == world_size else start + samples_per_client
    # print(dataset.keys())
    return dataset.select(range(start, end))

##########################################################
# Helper: Federated averaging of trainable model parameters
##########################################################
def federated_average(model, world_size):
    print("Performing: Federated Averaging....")
    # Only average parameters that require gradients (e.g. LoRA weights).
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Sum across all clients.
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            # Average by dividing by the number of clients.
            param.data /= world_size
            
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # Sum across all clients
    #         dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    #         # Rank 0 computes the average
    #         if dist.get_rank() == 0:
    #             param.data /= world_size
    print("Complete: Federated Averaging....")
    print("Performing: Updating client models....")
    # for param in model.parameters():
    #     dist.broadcast(param.data, src=0)
    print("Complete: Updating client models....")
##########################################################
# Modified conversion function that takes a tokenizer argument.
##########################################################
def convert_squad_sample_to_llama_conversation(sample, tokenizer):
    question = sample['question']
    context = sample['context']
    answers = sample['answers'][0]['text']
    if len(answers) == 0:
        answer = "The context does not provide an answer..."
    else:
        answer = answers[0]
    instruction_prompt_template = '''
    You are a helpful assistant tasked with extracting passages that answer users questions from a given context. Output exact passages word for word that answer the users question. Do not output any other text other than passages in the context passage. Output the minimal amount to answer the question, for example only 2-3 words from the passage. If you cannot find the answer in the context passage output 'The context does not provide an answer...'

    Context: {context}'''
    messages = [
        {"role": "system", "content": instruction_prompt_template.format(context=context)},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    # Apply the chat template.
    sample_conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": sample_conversation, "messages": messages, "answer": answer}

##########################################################
# Local training function to be run on each GPU (client)
##########################################################
def local_train(rank, world_size, num_rounds, local_steps, hf_token):
    # Explicitly set the CUDA device for this process.
    torch.cuda.set_device(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    device = torch.device("cuda", rank)

    # Initialize the distributed process group.
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    #########################################
    # Load and partition the dataset locally
    #########################################
    ds = load_dataset("/home/ob3942/datasets/formatted_mimic_notes")
    # num_training_samples = 15000
    training_samples = ds['train']#.select(range(num_training_samples))
    
    # Load the tokenizer. (Each client loads its own copy.)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Convert samples into our conversation format.
    conversation_training_samples = training_samples.map(
        lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer)
    )
    # Partition the conversation samples across clients.
    local_dataset = partition_dataset(conversation_training_samples, rank, world_size)
    print(f"Local dataset size for client {rank}: {len(local_dataset)}")
    if rank==0: # TODO initialize global test dataset for one gpu
        test_dataset = partition_dataset(conversation_training_samples, world_size, world_size)
        print(f"Local test dataset size for client {rank}: {len(test_dataset)}")
    #########################################
    # Load model with 4-bit quantization and LoRA
    #########################################
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print(f"rank {rank}: device: {torch.cuda.current_device()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": rank},  # Explicitly load on GPU index equal to rank.
        token=hf_token
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # model._set_static_graph()
    # Define the LoRA configuration.
    rank_lora = 128 # reduced to save memory originally: 128
    lora_alpha = rank_lora * 2
    peft_config = LoraConfig(
        r=rank_lora,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    ######################################################
    # Set up training arguments for local training rounds.
    # (Here we use a modest number of local steps per FL round.)
    ######################################################
    training_args = TrainingArguments(
        output_dir=f"./model/client_{rank}",
        optim='paged_adamw_32bit',
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        logging_steps=8,
        evaluation_strategy='steps',
        eval_steps=1,
        save_steps=1,
        learning_rate=1e-4,
        fp16=True,
        num_train_epochs=1,
        max_steps=local_steps,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        lr_scheduler_type='linear',
        local_rank=rank,  # Ensure accelerator uses the correct device.
        ddp_find_unused_parameters=False,  # Disable unused parameter check.
    )

    ######################################################
    # Initialize the trainer (SFTTrainer) for local training.
    ######################################################
    if rank:
        trainer = SFTTrainer(
            model=model,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,  # For demonstration, we use the same partition for eval.
            peft_config=peft_config,
            dataset_text_field='text',
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=local_dataset,
            eval_dataset=test_dataset,  # For demonstration, we use the same partition for eval.
            peft_config=peft_config,
            dataset_text_field='text',
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args
        )
        

    ######################################################
    # Federated Learning Rounds: local training then aggregation
    ######################################################
    for round_idx in range(num_rounds):
        print(f"Client {rank}: Starting FL round {round_idx+1}... Training...")
        
        # Run local training for the given number of steps.
        trainer.train()
        torch.cuda.synchronize()
        # Synchronize all clients before averaging.
        dist.barrier()
        print(f"Client {rank}: Complete: Training")
        
        # Compute and save average training loss
        print(trainer.state.log_history)
        train_loss = trainer.state.log_history[-1]['train_loss']  # Last logged loss
        train_loss_file = f"./model/client_{rank}_train_loss.json"
        append_loss_to_file(train_loss_file, train_loss)
    
        print(f"Federated averaging for round {round_idx+1}")
        # Perform federated averaging on the trainable parameters.
        federated_average(model, world_size)
        dist.barrier()
        eval_results = trainer.evaluate()


        if rank==0:
            #print("Evaluating the global model...")
            # (Optional) Evaluate locally and print out results.
            print(f"Global Evaluation: Round {round_idx+1} eval results: {eval_results}")
            eval_loss_file = f"./model/global_eval_loss.json"
            append_loss_to_file(eval_loss_file, eval_results["eval_loss"])


    if rank==0:	
        trainer.save_model("./model/federated_global_model")
        # Synchronize again to ensure all clients have the updated model.
        
        # Save the final global model from one client (e.g. rank 0).
        
            
    
    # Clean up the process group.
    dist.destroy_process_group()

##########################################################
# Main function: spawn 8 processes (one per GPU/FL client)
##########################################################
def main():
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    world_size = 8        # 8 GPUs -> 8 FL participants
    num_rounds = 200        # Number of federated rounds (adjust as needed)
    local_steps = 1#8      # Local training steps per round (adjust as needed)
    mp.spawn(local_train, args=(world_size, num_rounds, local_steps, hf_token), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    main()
