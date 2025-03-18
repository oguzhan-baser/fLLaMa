# Author: O. Baser
# Date: 3/18/25
# Affiliation: The University of Texas at Austin
# Details: flLaMa simulation parser main script
# Presets: activate the env specified in env.yml prior and ensure CUDA devices are visible
import argparse
import os
import torch.multiprocessing as mp
from federated import local_train

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Finetuning with LLaMA")
    parser.add_argument("--world_size", type=int, default=8, 
                        help="Number of GPUs (FL participants)")
    parser.add_argument("--num_rounds", type=int, default=200, 
                        help="Number of federated rounds")
    parser.add_argument("--local_steps", type=int, default=1, 
                        help="Local training steps per federated round")
    parser.add_argument("--dataset_path", type=str, 
                        default="/home/ob3942/datasets/formatted_mimic_notes", 
                        help="Path to formatted MIMIC Notes dataset")
    parser.add_argument("--model_name", type=str, 
                        default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="HF LLaMa Model dir/name")
    parser.add_argument("--hf_token", type=str, 
                        help="Hugging Face token")
    parser.add_argument("--output_dir", type=str, default="./model", 
                        help="Directory to save models and logs")
    # Add other arguments as needed.
    return parser.parse_args()

def main():
    args = parse_args()

    # Set distributed training environment variables.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_LAUNCH_TIMEOUT"] = "1200"  # timeout in seconds
    os.environ["NCCL_DEBUG"] = "INFO"
    
    mp.spawn(local_train, args=(args,), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()
