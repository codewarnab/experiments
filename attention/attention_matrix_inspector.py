# Goal: Load DistilBERT and see REAL learned attention patterns 

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from transformers import AutoTokenizer, AutoModel 

print("Loading DistilBERT...")
model_name = "distilbert-base-uncased"
# Load the tokenizer associated with DistilBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the model and explicitly request it to output attention weights
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# Set model to evaluation mode (disables dropout layers to ensure deterministic outputs)
model.eval() 
print(f"Layers: {model.config.n_layers}, Heads: {model.config.n_heads}")
# The dimension of each attention head is the total hidden dimension divided by the number of heads
print(f"Hidden dim: {model.config.dim}, Head dim: {model.config.dim // model.config.n_heads}")

def get_attention_weights(sentence):
    """
    Passes a sentence through DistilBERT and extracts the self-attention weights
    from all layers and heads.
    """
    # 1. Tokenize the input string into model-understandable tensor IDs
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    
    # 2. Extract the actual token IDs from the batch (batch size is 1) and convert to list
    token_ids = inputs["input_ids"][0].tolist()
    
    # 3. Convert token IDs back to human-readable string tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # 4. Pass the inputs through the model without calculating gradients (saves memory/compute)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 5. Return the human-readable tokens and the extracted attention weights 
    # (outputs.attentions is a tuple of tensors, one for each layer)
    return tokens, outputs.attentions


def plot_attention_layer(tokens, attentions, layer_idx, title=""):
    """
    Visualizes the attention matrix for every head in a specific layer.
    """
    # Find out how many attention heads are in this layer
    n_heads = attentions[layer_idx].shape[1] 
    
    # Create a grid of subplots: 2 rows, and enough columns for all heads
    fig, axes = plt.subplots(2, n_heads // 2, figsize=(20, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easy iteration 
    
    # Add a main title for the whole figure
    fig.suptitle(f"Layer {layer_idx+1} Attention Heads\n{title}", fontsize=12)

    for head_idx in range(n_heads):
        # Extract the attention matrix for this specific head
        # Shape is (batch_size, num_heads, seq_len, seq_len), so we select batch 0, head_idx
        # We use .cpu().detach().numpy() to convert PyTorch tensor to NumPy array
        attn_matrix = attentions[layer_idx][0, head_idx].cpu().detach().numpy()
        
        # Plot the attention matrix as a heatmap
        # cmap='viridis' gives a nice color scheme from dark purple (low) to yellow (high)
        # vmin=0, vmax=1 ensures the color scale is fixed between 0 and 1 (attention is a probability)
        sns.heatmap(attn_matrix, ax=axes[head_idx], cmap='viridis', 
                    xticklabels=tokens, yticklabels=tokens, 
                    vmin=0, vmax=1, cbar=False)
        
        # Set the title and rotate x-axis labels so they don't overlap
        axes[head_idx].set_title(f"Head {head_idx+1}")
        axes[head_idx].tick_params(axis='x', rotation=45)
        
    # Adjust spacing between plots so they look nice
    plt.tight_layout()
    return fig

# --- Examples ---

# Let's test the attention mechanism on two sentences with the ambiguous word 'bank'
sentence_finance = "I deposited money at the bank"
tokens_f, attentions_f = get_attention_weights(sentence_finance)
print(f"\nTokens: {tokens_f}")

sentence_geo = "She sat on the river bank"
tokens_g, attentions_g = get_attention_weights(sentence_geo)
print(f"Tokens: {tokens_g}")

# Plot layer 0 (the first layer) for both sentences to see low-level attention patterns
# We save the figures as PNG images
fig1 = plot_attention_layer(tokens_f, attentions_f, layer_idx=0, title=sentence_finance)
fig1.savefig('attention_finance.png', dpi=120, bbox_inches='tight')

fig2 = plot_attention_layer(tokens_g, attentions_g, layer_idx=0, title=sentence_geo)
fig2.savefig('attention_geo.png', dpi=120, bbox_inches='tight')

# Show the plots on the screen
plt.show()

def find_token_attention(tokens, attentions, target, layer_idx=5, head_idx=0):
    """
    Finds and prints what a specific 'target' token is paying attention to.
    This helps us inspect how the model contextualizes a word!
    """
    # Check if the target word is actually in our token list
    if target not in tokens:
        print(f"Token '{target}' not found!")
        return
        
    # Find the index position of our target token
    idx = tokens.index(target)
    
    # Get the attention scores for this token. 
    # This row tells us how much 'target' attends to every other token in the sentence.
    attn_row = attentions[layer_idx][0, head_idx, idx].cpu().detach().numpy()
    
    print(f"\nWhere does '{target}' attend? (Layer {layer_idx+1}, Head {head_idx+1})")
    
    # Combine tokens with their attention weights, sort them from highest weight to lowest
    for tok, w in sorted(zip(tokens, attn_row), key=lambda x: -x[1]):
        # Print a simple text-based bar chart using the '|' character!
        # The number of bars is scaled by the attention weight.
        print(f" {tok:>15}: {w:.4f} {'|' * int(w * 30)}")

# Let's see how the model understands 'bank' in both contexts!
# Layer 5 (the last layer in a 6-layer model like DistilBERT) usually captures high-level context
print("\n=== Finance: where does 'bank' attend? ===")
find_token_attention(tokens_f, attentions_f, target="bank", layer_idx=5, head_idx=0)

print("\n=== Geography: where does 'bank' attend? ===")
find_token_attention(tokens_g, attentions_g, target="bank", layer_idx=5, head_idx=0)