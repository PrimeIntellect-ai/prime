from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from torch.optim import AdamW, Optimizer


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


# Optimizer update step
def optimizer_step(optimizer, model_params, gradients):
    for param, grad in zip(model_params, gradients):
        if param.grad is not None:
            param.grad = grad  # Set gradients
    optimizer.step()  # Perform optimizer step
    optimizer.zero_grad()  # Reset gradients


def acco_algorithm(
    model: torch.nn.Module, tokenizer, data_loader: DataLoader, optimizer: Optimizer, num_steps: int
) -> None:
    """
    ACCO algorithm implementation without memory leaks.
    """
    model_params = [p for p in model.parameters() if p.requires_grad]

    for step, batch in enumerate(data_loader):
        if step >= num_steps:
            break

        # Split the batch into two halves
        batch_text = batch["text"]
        mid_point = len(batch_text) // 2
        first_half, second_half = batch_text[:mid_point], batch_text[mid_point:]

        # Stage 1: Compute gradients g_t using the second half of the batch
        loss_t = compute_loss(model, second_half, tokenizer)
        loss_t.backward()  # Compute gradients for g_t
        g_t = [p.grad.clone() for p in model_params]  # Copy gradients
        optimizer.zero_grad()  # Clear gradients to avoid accumulation

        # Stage 2: Estimate next parameters (tilde_theta_t+1)
        with torch.no_grad():
            tilde_theta_t1 = [param - optimizer.defaults["lr"] * grad for param, grad in zip(model_params, g_t)]

        # Temporarily update model parameters to tilde_theta_t+1
        for param, tilde_param in zip(model_params, tilde_theta_t1):
            param.data.copy_(tilde_param)

        # Compute estimated gradients tilde_g_t+1 using the first half of the batch
        loss_tilde = compute_loss(model, first_half, tokenizer)
        tilde_g_t1 = torch.autograd.grad(loss_tilde, model_params)  # No need for retain_graph

        # Restore original parameters
        for param, original_param in zip(model_params, tilde_theta_t1):
            param.data.copy_(original_param)

        # Update parameters theta_t+1 using combined gradients
        combined_gradients = [g_t_i + tilde_g_t1_i for g_t_i, tilde_g_t1_i in zip(g_t, tilde_g_t1)]
        optimizer_step(optimizer, model_params, combined_gradients)

        print(f"Step {step + 1}/{num_steps}: Loss = {loss_t.item()}")


# Main function
def main():
    # Load dataset
    dataset = load_dataset(
        "/root/prime/prime/datasets/fineweb-edu", split="train", streaming=True
    )  # Small subset for example
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Load model and tokenizer
    model_name = "llama-debug"  # Replace with actual Llama model if available
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Run ACCO algorithm
    num_steps = 100
    acco_algorithm(model, tokenizer, data_loader, optimizer, num_steps)


# Entry point
if __name__ == "__main__":
    main()
