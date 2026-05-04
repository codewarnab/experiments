"""
Attention head inspection for an IOI-style prompt.

Each attention head computes a (T x T) matrix of attention weights for a given
input. Entry (i, j) is how much position i attends to position j.

Why this matters:
Circuits are composed of attention heads plus MLP layers. Before identifying a
circuit, you need to know what a head does. Attention pattern visualization is
often the first step in circuit analysis.
"""

import torch
import numpy as np

from setup import model, run


PROMPT = "When Mary and John went to the store, John gave a bag to"

# This is an Indirect Object Identification-style prompt.
# The model should predict " Mary", because John gave something to someone
# other than John.

tokens, logits, cache = run(PROMPT)

# Decode each token individually
token_strs = [model.tokenizer.decode([t]) for t in tokens[0].tolist()]
seq_len = tokens.shape[1]

print("Prompt:", PROMPT)
print(f"Token strings: {token_strs}")
print(f"Sequence length: {seq_len}")
print()


# ------------------------------------------------------------
# Step 1: Print top predictions
# ------------------------------------------------------------

# logits shape: (1, seq_len, vocab_size)
# We want the next-token prediction at the final position.
last_logits = logits[0, -1, :]

top_k = last_logits.topk(10)

print("Top 10 next-token predictions:")
print(f"{'Rank':>5} | {'Token':>15} | {'Logit':>10}")
print("-" * 38)

for rank, (tok_id, logit_val) in enumerate(zip(top_k.indices, top_k.values), 1):
    tok_str = model.tokenizer.decode([tok_id.item()])
    print(f"{rank:>5} | {repr(tok_str):>15} | {logit_val.item():>10.4f}")

print()


# ------------------------------------------------------------
# Step 2: Inspect selected attention heads
# ------------------------------------------------------------

# In GPT-2 Small IOI work, some famous name mover heads are often around
# layer 9 and layer 10. These exact heads can depend on model/setup.
LAYERS_TO_INSPECT = [9, 10]
HEADS_TO_INSPECT = [6, 9]

for L in LAYERS_TO_INSPECT:
    # cache["pattern", L] shape: (1, n_heads, seq_len, seq_len)
    attn_pattern = cache["pattern", L][0]

    for H in HEADS_TO_INSPECT:
        head_pattern = attn_pattern[H]  # shape: (seq_len, seq_len)

        # Look at attention FROM the final token TO all previous tokens.
        last_token_attention = head_pattern[-1, :]

        print(f"Layer {L} Head {H} -- attention from last token to each position")
        print("-" * 72)

        for pos, (tok, weight) in enumerate(
            zip(token_strs, last_token_attention.tolist())
        ):
            attn_bar = "█" * int(weight * 40)
            print(f"pos {pos:>2} {repr(tok):>15}: {weight:.3f} {attn_bar}")

        print()


# ------------------------------------------------------------
# Step 3: Find the token position for Mary
# ------------------------------------------------------------

# Tokenization may include a leading space, so search robustly.
mary_candidates = [
    i for i, tok in enumerate(token_strs)
    if tok.strip().lower() == "mary"
]

if len(mary_candidates) == 0:
    raise ValueError(f"Could not find Mary in tokenized prompt: {token_strs}")

mary_pos = mary_candidates[0]

print(f"Mary token found at position {mary_pos}: {repr(token_strs[mary_pos])}")
print()


# ------------------------------------------------------------
# Step 4: Scan all heads for attention to Mary from the last position
# ------------------------------------------------------------

print(f"Scanning all heads for maximum attention to {repr(token_strs[mary_pos])}")
print("(from the last token position)")
print()

scores = []

for L in range(model.cfg.n_layers):
    attn_pattern = cache["pattern", L][0]  # shape: (n_heads, seq_len, seq_len)

    for H in range(model.cfg.n_heads):
        # Attention from final token to Mary position
        weight = attn_pattern[H, -1, mary_pos].item()
        scores.append((weight, L, H))

scores.sort(reverse=True)

print(f"Top 10 layer/head pairs attending to {repr(token_strs[mary_pos])}:")
print(f"{'Rank':>5} | {'Layer':>5} | {'Head':>5} | {'Attn Weight':>12}")
print("-" * 42)

for rank, (weight, L, H) in enumerate(scores[:10], 1):
    print(f"{rank:>5} | {L:>5} | {H:>5} | {weight:>12.4f}")

print()


# ------------------------------------------------------------
# Step 5: Visualize the best head
# ------------------------------------------------------------

best_weight, best_layer, best_head = scores[0]

print(
    f"Best head: Layer {best_layer}, Head {best_head}, "
    f"attention to Mary = {best_weight:.4f}"
)
print()

best_pattern = cache["pattern", best_layer][0][best_head]
best_last_token_attention = best_pattern[-1, :]

print(
    f"Layer {best_layer} Head {best_head} -- full attention pattern "
    "from last token"
)
print("-" * 72)

for pos, (tok, weight) in enumerate(
    zip(token_strs, best_last_token_attention.tolist())
):
    attn_bar = "█" * int(weight * 40)
    marker = "<-- Mary" if pos == mary_pos else ""
    print(f"pos {pos:>2} {repr(tok):>15}: {weight:.3f} {attn_bar} {marker}")

print()


# ------------------------------------------------------------
# Step 6: Optional average attention pattern across heads
# ------------------------------------------------------------

print("Average attention from last token across all heads in selected layers:")
print()

for L in LAYERS_TO_INSPECT:
    attn_pattern = cache["pattern", L][0]  # shape: (n_heads, seq_len, seq_len)

    # Average over heads, then take final-token row
    avg_last_attention = attn_pattern.mean(dim=0)[-1, :]

    print(f"Layer {L} average attention from last token")
    print("-" * 72)

    for pos, (tok, weight) in enumerate(zip(token_strs, avg_last_attention.tolist())):
        attn_bar = "█" * int(weight * 40)
        print(f"pos {pos:>2} {repr(tok):>15}: {weight:.3f} {attn_bar}")

    print()


# ------------------------------------------------------------
# Suggestions
# ------------------------------------------------------------

print("WHAT TO TRY NEXT:")
print("1. Change the prompt but keep the structure:")
print("   'When Alice and Bob went to the park, Bob gave a book to'")
print("   Do the same heads attend to Alice?")
print()
print("2. Set:")
print("   LAYERS_TO_INSPECT = list(range(model.cfg.n_layers))")
print("   to inspect every layer.")
print()
print("3. Try changing the repeated subject:")
print("   'When Mary and John went to the store, Mary gave a bag to'")
print("   Now the correct answer should be John.")