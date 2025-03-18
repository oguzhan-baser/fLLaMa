# Author: O. Baser
# Date: 3/18/25
# Affiliation: The University of Texas at Austin
# Details: Side utility functions for federated LLM fine tuning operations
# Presets: None

import os
import json
import torch.distributed as dist

def append_loss_to_file(file_path, new_loss):
    """Append new loss value to a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            loss_data = json.load(f)
    else:
        loss_data = []
    loss_data.append(new_loss)
    with open(file_path, "w") as f:
        json.dump(loss_data, f, indent=4)

def partition_dataset(dataset, rank, world_size):
    """Partition the dataset among FL clients."""
    total_samples = len(dataset)
    samples_per_client = total_samples // (world_size + 1)
    start = rank * samples_per_client
    end = total_samples if rank == world_size else start + samples_per_client
    return dataset.select(range(start, end))

def federated_average(model, world_size):
    """Averages trainable parameters (e.g. LoRA weights) across FL clients."""
    print("Performing: Federated Averaging....")
    for name, param in model.named_parameters():
        if param.requires_grad:
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= world_size
    print("Complete: Federated Averaging....")
    print("Performing: Updating client models....")
    # Optionally, you can broadcast updated parameters here.
    print("Complete: Updating client models....")

def convert_squad_sample_to_llama_conversation(sample, tokenizer):
    """
    Convert a SQuAD/MIMIC sample to a conversation format expected by LLaMA.
    """
    question = sample["question"]
    context = sample["context"]
    # Assumes sample['answers'] is structured as a list of dicts.
    answers = sample["answers"][0]["text"]
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
        {"role": "assistant", "content": answer},
    ]
    sample_conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": sample_conversation, "messages": messages, "answer": answer}
