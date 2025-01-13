import torch.distributed as dist
from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from torch.optim import AdamW, Optimizer
import wandb
import psutil
from tqdm import tqdm


# Loss function
def compute_loss(model: torch.nn.Module, inputs: List[str], tokenizer) -> torch.Tensor:
    """
    Compute the loss for a batch of input text using a causal language modeling objective.

    Args:
        model (torch.nn.Module): The pre-trained model (e.g., Llama).
        inputs (List[str]): A batch of input text strings.
        tokenizer: The tokenizer associated with the model.

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Tokenize input text and prepare for model input
    input_ids = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings
    ).input_ids
    input_ids = input_ids.to(model.device)
    labels = input_ids.clone()

    # Compute the loss
    outputs = model(input_ids, labels=labels)
    return outputs.loss

def print_memory_usage():
    # Get CPU memory usage
    memory_info = psutil.virtual_memory()
    cpu_memory_used = memory_info.used / (1024 ** 2)
    cpu_memory_total = memory_info.total / (1024 ** 2)

    print(f"CPU Memory Usage:")
    print(f"Used: {cpu_memory_used:.2f} MB")
    print(f"Total: {cpu_memory_total:.2f} MB")
    print(f"Percentage: {memory_info.percent}%\n")

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get current device
        device = torch.device('cuda')
        gpu_memory_used = torch.cuda.memory_allocated(device=device)
        gpu_memory_reserved = torch.cuda.memory_reserved(device=device)
        gpu_memory_total = torch.cuda.get_device_properties(device).total_memory

        print(f"GPU Memory Usage (Device: {torch.cuda.get_device_name(device)}):")
        print(f"Allocated: {gpu_memory_used / (1024 ** 2):.2f} MB")
        print(f"Reserved: {gpu_memory_reserved / (1024 ** 2):.2f} MB")
        print(f"Total: {gpu_memory_total / (1024 ** 2):.2f} MB\n")
    else:
        print("CUDA is not available.")

# Main function
def main():
    batch_size = 8
    # Load dataset
    dataset = load_dataset("/root/prime/prime/datasets/fineweb-edu", split="train", streaming=True)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Load model and tokenizer
    model_name = "llama-debug"  # Replace with actual Llama model if available
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}, Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print_memory_usage()
    theta_t = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer_copy = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    reduce_work = []
    model.to("cuda")

    # Define optimizer
    optimizer = AdamW(optimizer_copy, lr=1e-4)

    # Run ACCO algorithm
    num_steps = 100

    first_step = True
    print("Post Init")
    print_memory_usage()
    for step, batch in tqdm(enumerate(data_loader), total=num_steps):
        if step >= num_steps:
            break

        # Split the batch into two halves
        batch_text = batch["text"]
        mid_point = len(batch_text) // 2
        first_half, second_half = batch_text[:mid_point], batch_text[mid_point:]

        # Stage 1: Compute gradients g_tilde and theta
        for p in model.parameters():
            p.grad = None
        loss = compute_loss(model, first_half, tokenizer)
        loss.backward()  # Compute gradients for g_t
        for work in reduce_work:
            work.wait()

        if not first_step:
            for opt_param, cpu_param, _g_t, _g_tilde in zip(optimizer_copy, theta_t, g_t, g_tilde):
                opt_param.data = cpu_param.data
                opt_param.grad = (_g_t + _g_tilde) / (batch_size * dist.get_world_size())
            optimizer.step()
            for param, cpu_param, opt_param in zip(model.parameters(), theta_t, optimizer_copy):
                param.data.copy_(opt_param.data, non_blocking=True)
                cpu_param.data.copy_(opt_param.data, non_blocking=True)
        first_step = False 

        g_tilde = [p.grad.cpu() for p in model.parameters() if p.requires_grad]
        reduce_work = [dist.all_reduce(_g_tilde, op=dist.ReduceOp.SUM, async_op=True) for _g_tilde in g_tilde]

        # Stage 2: Compute g_t and theta_tilde
        for p in model.parameters():
            p.grad = None
        loss = compute_loss(model, second_half, tokenizer)
        loss.backward()
        g_t = [p.grad.cpu() for p in model.parameters() if p.requires_grad]
        for work in reduce_work:
            work.wait()
        reduce_work = [dist.all_reduce(_g_t, op=dist.ReduceOp.SUM, async_op=True) for _g_t in g_t]

        # theta_tilde
        for param, _g_tilde in zip(optimizer_copy, g_tilde):
            ## TODO: Weight by seen by batches
            param.grad = _g_tilde / (batch_size // 2 * dist.get_world_size())
        optimizer.step()
        for param, param_tilde in zip(model.parameters(), optimizer_copy):
            param.data.copy_(param_tilde, non_blocking=True)

        print(f"Step {step + 1}/{num_steps}: Loss = {loss.item()}")
        wandb.log({"loss": loss.item()})
        print(f"End of step {step}")
        print_memory_usage()

# Entry point
if __name__ == "__main__":
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    torch.cuda.set_device(dist.get_rank())
    wandb.init()
    main()
    wandb.finish()
    dist.destroy_process_group()
