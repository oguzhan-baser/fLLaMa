# Author: O. Baser
# Date: 3/18/25
# Affiliation: The University of Texas at Austin
# Details: Multi-GPU One-Node Federated Learning Simulations for LLM Fine-Tuning
# Presets: None

import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig
from utils import (
    append_loss_to_file,
    partition_dataset,
    federated_average,
    convert_squad_sample_to_llama_conversation,
)

def local_train(rank, args):
    # Set device for this process.
    torch.cuda.set_device(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    device = torch.device("cuda", rank)

    # Initialize the distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
    
    #########################################
    # Load and partition the dataset locally
    #########################################
    ds = load_dataset(args.dataset_path)
    training_samples = ds["train"]
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Convert and partition the dataset.
    conversation_training_samples = training_samples.map(
        lambda sample: convert_squad_sample_to_llama_conversation(sample, tokenizer)
    )
    local_dataset = partition_dataset(conversation_training_samples, rank, args.world_size)
    print(f"Local dataset size for client {rank}: {len(local_dataset)}")
    
    if rank == 0:
        # Use an extra partition as a test dataset on rank 0.
        test_dataset = partition_dataset(conversation_training_samples, args.world_size, args.world_size)
        print(f"Local test dataset size for client {rank}: {len(test_dataset)}")
    
    #########################################
    # Load model with 4-bit quantization and LoRA
    #########################################
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print(f"Rank {rank}: device: {torch.cuda.current_device()}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": rank},  # Force model to load on the GPU with index = rank.
        token=args.hf_token
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Define LoRA configuration.
    rank_lora = 128
    lora_alpha = rank_lora * 2
    peft_config = LoraConfig(
        r=rank_lora,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )
    
    ##########################################################
    # Set up training arguments.
    ##########################################################
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"client_{rank}"),
        optim="paged_adamw_32bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        logging_steps=8,
        evaluation_strategy="steps",
        eval_steps=1,
        save_steps=1,
        learning_rate=1e-4,
        fp16=True,
        num_train_epochs=1,
        max_steps=args.local_steps,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        lr_scheduler_type="linear",
        local_rank=rank,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize SFTTrainer.
    if rank == 0:
        trainer = SFTTrainer(
            model=model,
            train_dataset=local_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args
        )
    
    ##########################################################
    # Federated Learning Rounds: local training then aggregation.
    ##########################################################
    for round_idx in range(args.num_rounds):
        print(f"Client {rank}: Starting FL round {round_idx+1}... Training...")
        trainer.train()
        torch.cuda.synchronize()  # Ensure all CUDA ops complete before barrier.
        dist.barrier()
        print(f"Client {rank}: Complete: Training")
        
        # Log training loss.
        train_loss = trainer.state.log_history[-1]["train_loss"]
        train_loss_file = os.path.join(args.output_dir, f"client_{rank}_train_loss.json")
        append_loss_to_file(train_loss_file, train_loss)
    
        print(f"Federated averaging for round {round_idx+1}")
        federated_average(model, args.world_size)
        dist.barrier()
        eval_results = trainer.evaluate()

        if rank == 0:
            print(f"Global Evaluation: Round {round_idx+1} eval results: {eval_results}")
            eval_loss_file = os.path.join(args.output_dir, "global_eval_loss.json")
            append_loss_to_file(eval_loss_file, eval_results["eval_loss"])

    if rank == 0:
        trainer.save_model(os.path.join(args.output_dir, "federated_global_model"))
    
    dist.destroy_process_group()
