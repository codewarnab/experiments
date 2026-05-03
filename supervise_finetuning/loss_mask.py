"""

Experiment: The real difference between pre-training and SFT?
It's just the "Mask" 

The Core Idea: 
- Pre-training: The model learns from every single word in the sequence.
- SFT (Supervised Fine-tuning): The model only learns from the answer, not the question.

The Mask (0s and 1s):
We use a vector of the same length as our text: 
- 0 = "Ignore this" (for the prompt/question)
- 1 = "Learn this" (for the response/answer)

Why does this matter? 
By multiplying the loss by this mask, we ensure the model focuses its learning purely on how to respond, rather than just repeating the prompt.

In this file, we: 
1. Calculate loss across everything (Pre-training style).
2. Calculate loss only on the response (SFT style).
3. Verify that the "learning signal" (gradient) for the prompt is zero in SFT.

Why this matters:
If you forgot to apply the mask (a famously common implementation bug), the training loss still goes down -- because the model happily learns to predict prompt tokens. But it does not learn to follow instructions. The model improves at the wrong task silently.

SETUP: 
We use a tiny vocabulary of size V=8 and a sequence of length T=7.
Positions 0-3 are "prompt tokens" (mask = 0).
Positions 4-6 are "response tokens" (mask = 1).
The "model" is just a random logit matrix -- we are testing the loss function, not a real model.

"""

import torch 
import torch.nn.functional as F 

def compute_loss_pretrain(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
    """ 
    Compute pre-training style cross-entropy loss: 
    All tokens contribute.

    Args: 
        logits: Shape (T, V). Model's raw scores for each token at each sequence position.
        targets: Shape (T,). Ground-truth token IDs.

    Returns: 
        A scalar tensor: the mean cross-entropy loss over all T positions.
    """
    loss = F.cross_entropy(logits, targets, reduction='mean')
    return loss 

def compute_loss_sft(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
    """
    Compute SFT-style cross-entropy loss: ONLY response tokens contribute.

    Args:
        logits: Shape (T, V). Model logits at each sequence position.
        targets: Shape (T,). Ground-truth token IDs.
        mask: Shape (T,). Binary: 1 for response positions, 0 for prompt.

    Returns:
        A scalar tensor: mean cross-entropy loss over response tokens only.

    THE KEY IMPLEMENTATION:
    We compute per-token loss first (reduction='none' -> shape (T,)).
    Then we multiply by the mask: prompt losses become 0 * loss = 0.
    Then we average over the response tokens only (sum / mask.sum()).

    WHY NOT mask.mean():
    If we divided by T instead of mask.sum(), the loss would be artificially
    smaller whenever prompts are long (because we are summing fewer terms
    but dividing by the full T). We normalise by the number of RESPONSE
    tokens so the loss scale is consistent.
    """
    # Step 1: Per-token cross-entropy. Shape: (T,)
    # reduction='none' means: compute loss for each position independently.
    per_token_loss = F.cross_entropy(logits, targets, reduction='none')

    # Step 2: Multiply by the mask.
    # mask[t] = 0 for prompt tokens -> their loss contribution is wiped to 0.
    # mask[t] = 1 for response tokens -> their loss is kept as-is.
    masked_loss = per_token_loss * mask

    # Step 3: Average over response tokens only.
    # mask.sum() counts how many response tokens there are (= number of 1s).
    loss = masked_loss.sum() / mask.sum()
    return loss

def run_experiment():
    """
    Run the comparison and inspect gradients to confirm masking behaviour.

    WHAT WE INSPECT:
    After calling loss.backward(), each leaf tensor that required_grad=True
    will have a .grad attribute. The shape of this .grad matches the tensor.
    For a logit matrix of shape (T, V), logits.grad[t, :] will be all zeros
    if token position t contributed nothing to the loss.

    In the pre-training case: logits.grad[t, :] != 0 for ALL t.
    In the SFT case: logits.grad[t, :] == 0 for prompt positions t.

    This is the numerical proof that the mask controls which positions the
    model is trained on.
    """
    torch.manual_seed(42) # Reproducibility

    T = 7 # sequence length (positions 0-6)
    V = 8 # vocabulary size

    # Prompt: positions 0, 1, 2, 3 (4 tokens)
    # Response: positions 4, 5, 6 (3 tokens)

    # Fake target token IDs -- just random integers in [0, V)
    targets = torch.randint(0, V, (T,))

    # PRE-TRAINING CASE
    # requires_grad=True: we want PyTorch to track gradients through these logits
    # so we can inspect them after backward().
    logits_pt = torch.randn(T, V, requires_grad=True)

    loss_pt = compute_loss_pretrain(logits_pt, targets)
    loss_pt.backward() # Compute gradients via backpropagation

    print("=" * 60)
    print("PRE-TRAINING LOSS (all tokens)")
    print(f"Loss value: {loss_pt.item():.4f}")
    print("Gradient norm at each position:")
    for t in range(T):
        # .grad[t] is a vector of length V. We take its L2 norm to get a scalar.
        grad_norm = logits_pt.grad[t].norm().item()
        label = "PROMPT" if t < 4 else "RESPONSE"
        print(f"t={t} ({label}): grad_norm = {grad_norm:.4f}")

    print()

    # SFT CASE (WITH MASK)
    logits_sft = torch.randn(T, V, requires_grad=True)

    # The mask: 0 for prompt positions (0-3), 1 for response positions (4-6).
    mask = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.float)

    loss_sft = compute_loss_sft(logits_sft, targets, mask)
    loss_sft.backward()

    print("SFT LOSS (response tokens only, mask applied)")
    print(f"Loss value: {loss_sft.item():.4f}")
    print("Gradient norm at each position:")
    for t in range(T):
        grad_norm = logits_sft.grad[t].norm().item()
        label = "PROMPT" if t < 4 else "RESPONSE"
        print(f"t={t} ({label}): grad_norm = {grad_norm:.6f}")

    print()
    print("KEY TAKEAWAY:")
    print("Prompt positions under SFT have zero gradient.")
    print("The mask IS the difference between pre-training and SFT.")

if __name__ == "__main__":
    run_experiment()
