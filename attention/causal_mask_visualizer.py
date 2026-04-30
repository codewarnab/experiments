#Goal : Implement causal masking and visualize the masked attention matrix 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

np.random.seed(42) 
n = 6 # number of tokens (sequence length)
dk = 8 # key/query dimension 

# generate random query, key and value matrices 
queries = np.random.randn(n,dk)
keys = np.random.randn(n,dk)
values = np.random.randn(n,dk)

token_labels = ["The" , "cat" , "sat" , "on" , "the" , "mat"] # input text 

#Step 1 : Unmasked (bidirectional) attenttion scores 
raw_scores = (queries @ keys.T ) / np.sqrt(dk)

def softmax_2d(x) : 
    x_shifted = x - np.max(x , axis = -1 , keepdims = True ) 
    e = np.exp(x_shifted)
    sum_e = np.sum(e , axis = -1 , keepdims = True)
    return e / sum_e 

A_unmasked = softmax_2d(raw_scores)


#Step 2 : Build the causal mask 
MASK_VALUE = -1e9 # large negative number 
n_tokens = queries.shape[0]
causal_mask = np.triu(np.ones((n_tokens, n_tokens)), k=1) * MASK_VALUE  # 1s represent the 'future' tokens we want to hide

#Step 3 : Apply the causal mask to raw scores 
masked_scores = raw_scores +causal_mask 
A_masked = softmax_2d(masked_scores) 


# Step 4 verify the mask 
print("\n---Verification---") 
print(f"Sum of row 0 (The): {np.sum(A_masked[0]):.3f}")
print(f"Values in row 0 (The):")
for tok, val in zip(token_labels, A_masked[0]):
    print(f"  {tok: <5s}: {val:.3f}")

# Step 5: Visualize the unmasked and masked attention matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Unmasked Attention
sns.heatmap(A_unmasked, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=token_labels, yticklabels=token_labels, ax=axes[0])
axes[0].set_title("Unmasked Attention Matrix\n(Tokens see future tokens)")
axes[0].set_xlabel("Key (attended to)")
axes[0].set_ylabel("Query (attending from)")

# Masked (Causal) Attention
sns.heatmap(A_masked, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=token_labels, yticklabels=token_labels, ax=axes[1])
axes[1].set_title("Causal Masked Attention Matrix\n(Tokens only see past/present)")
axes[1].set_xlabel("Key (attended to)")
axes[1].set_ylabel("Query (attending from)")

plt.tight_layout()
plt.show()
