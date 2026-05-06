"""
EXPERIMENT 3: The Logit Lens
============================

Question we are answering:
At which layer does the model first "know" the correct next token?
What garbage does it predict in early layers?

This version:
1. Shows the top predicted tokens at every layer.
2. Checks whether the final model predicts the correct token.
3. Tracks the probability and rank of the correct token across layers.

Important tokenizer note:
For GPT-style tokenizers, the correct next token is usually " Paris"
with a leading space, not "Paris".
"""

import torch
import torch.nn.functional as F
from setup import model, run


# Try this prompt first.
# It usually cues factual recall better than:
# "The capital of France is"
PROMPT = "Q: What is the capital of France?\nA:"

CORRECT_TOKEN = " Paris"
TOP_K = 5


tokens, logits, cache = run(PROMPT)

token_strs = [
    model.tokenizer.decode([t])
    for t in tokens[0].tolist()
]

last_pos = tokens.shape[1] - 1

print(f"Prompt: {PROMPT!r}")
print(f"Tokens: {token_strs}")
print()


# The unembedding matrix maps d_model -> vocab_size.
# Shape: (d_model, vocab_size)
W_U = model.W_U


def get_token_id(token_str):
    """
    Convert a string like " Paris" into the token id for the final token.

    Some GPT-style tokenizers automatically prepend <|endoftext|>.
    So we take the last token id.
    """

    token_ids = model.tokenizer.encode(token_str)

    print(f"Raw encoding for {token_str!r}: {token_ids}")
    print(
        "Decoded pieces:",
        [model.tokenizer.decode([token_id]) for token_id in token_ids]
    )

    token_id = token_ids[-1]

    return token_id


correct_id = get_token_id(CORRECT_TOKEN)

print(f"Correct token string: {CORRECT_TOKEN!r}")
print(f"Correct token id: {correct_id}")
print(f"Correct token decoded: {model.tokenizer.decode([correct_id])!r}")
print()


def logits_from_resid(resid):
    """
    Apply the final layer norm and unembedding to a residual stream vector.

    Args:
        resid: Tensor of shape (d_model,)

    Returns:
        logits_flat: Tensor of shape (vocab_size,)
    """

    # Add batch and sequence dimensions:
    # (d_model,) -> (1, 1, d_model)
    resid = resid.unsqueeze(0).unsqueeze(0)

    # Apply final layer norm.
    resid_normed = model.ln_final(resid)

    # Apply unembedding:
    # (1, 1, d_model) @ (d_model, vocab_size)
    # -> (1, 1, vocab_size)
    logits = resid_normed @ W_U

    return logits[0, 0, :]


def top_tokens_from_logits(logits_flat, k=5):
    """
    Return the top-k predicted tokens and probabilities from logits.

    Args:
        logits_flat: Tensor of shape (vocab_size,)
        k: number of tokens to return

    Returns:
        List of (decoded_token, probability) tuples.
    """

    probs = F.softmax(logits_flat, dim=-1)

    # Use tensor method to avoid environments where torch.topk is not exported.
    top = probs.topk(k=k)

    return [
        (
            model.tokenizer.decode([token_id.item()]),
            prob.item()
        )
        for token_id, prob in zip(top.indices, top.values)
    ]


def correct_token_stats(logits_flat, correct_token_id):
    """
    Compute probability and rank of the correct token.

    Rank 1 means the correct token is the most likely token.
    """

    probs = F.softmax(logits_flat, dim=-1)

    correct_prob = probs[correct_token_id].item()

    # Number of tokens with a higher logit than the correct token, plus 1.
    correct_rank = (
        (logits_flat > logits_flat[correct_token_id])
        .sum()
        .item()
        + 1
    )

    return correct_prob, correct_rank


print("Final model top predictions")
print("-" * 80)

final_logits = logits[0, -1, :]
final_top = top_tokens_from_logits(final_logits, k=10)

for token, prob in final_top:
    print(f"{token!r:>18} | {prob:>8.4f}")

final_correct_prob, final_correct_rank = correct_token_stats(
    final_logits,
    correct_id
)

print()
print(f"Final P({CORRECT_TOKEN!r}): {final_correct_prob:.6f}")
print(f"Final rank of {CORRECT_TOKEN!r}: {final_correct_rank}")
print()


print("Logit lens: top predicted next tokens at each layer")
print(
    f"{'Layer':>7} | "
    f"{'Top-1 Token':>18} | "
    f"{'Prob':>8} | "
    f"{'Top-2':>15} | "
    f"{'Top-3':>15}"
)
print("-" * 80)

for L in range(model.cfg.n_layers):
    # Residual stream after layer L, at the final token position.
    resid = cache["resid_post", L][0, last_pos, :]

    layer_logits = logits_from_resid(resid)
    top5 = top_tokens_from_logits(layer_logits, k=TOP_K)

    t1, p1 = top5[0]
    t2, _ = top5[1]
    t3, _ = top5[2]

    print(
        f"L={L:>2}   | "
        f"{repr(t1):>18} | "
        f"{p1:>8.4f} | "
        f"{repr(t2):>15} | "
        f"{repr(t3):>15}"
    )


print()
print("Correct token probability and rank by layer")
print(f"{'Layer':>7} | {f'P({CORRECT_TOKEN!r})':>14} | {'Rank':>8}")
print("-" * 40)

for L in range(model.cfg.n_layers):
    resid = cache["resid_post", L][0, last_pos, :]

    layer_logits = logits_from_resid(resid)

    correct_prob, correct_rank = correct_token_stats(
        layer_logits,
        correct_id
    )

    print(
        f"L={L:>2}   | "
        f"{correct_prob:>14.6f} | "
        f"{correct_rank:>8}"
    )


print()
print("Embedding layer, before any transformer block")
print("-" * 80)

resid_embed = cache["hook_embed"][0, last_pos, :]
embed_logits = logits_from_resid(resid_embed)
embed_top = top_tokens_from_logits(embed_logits, k=TOP_K)
embed_correct_prob, embed_correct_rank = correct_token_stats(
    embed_logits,
    correct_id
)

print(f"Top predictions at L=-1: {embed_top}")
print(f"P({CORRECT_TOKEN!r}) at L=-1: {embed_correct_prob:.6f}")
print(f"Rank of {CORRECT_TOKEN!r} at L=-1: {embed_correct_rank}")


print()
print("WHAT TO TRY NEXT:")
print("1. Try PROMPT = 'The capital of France is'")
print("   Compare whether ' Paris' appears earlier or later.")
print()
print("2. Try PROMPT = 'Q: What is the capital of Germany?\\nA:'")
print("   Set CORRECT_TOKEN = ' Berlin'")
print()
print("3. Try PROMPT = 'The cat sat on the'")
print("   Set CORRECT_TOKEN to a likely completion, such as ' mat'.")
print()
print("4. Plot P(correct token) vs layer number.")