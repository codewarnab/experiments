# Task given [ 3 , 1 , 4 , 5 ] predict [1 , 1, 3, 4 , 5 ]

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt


SEQ_LEN  = 5 
VOCAB_SIZE = 10 
D_MODEL = 32 
D_FF = 64 
N_EPOCHS = 500 
BATCH_SIZE = 64 
LR = 1e-3  # Learning Rate 

def generate_sort_data(batch_size, seq_len, vocab_size):
    """
    Generates synthetic data for a sorting task.
    
    This function creates a batch of sequences where each sequence contains 
    random integers. It also produces the "ground truth" (targets) by 
    sorting those sequences. In deep learning, we use these pairs to 
    train a model to learn the rule of sorting.

    Args:
        batch_size (int): Number of sequences to generate in one go.
        seq_len (int): The length of each individual sequence.
        vocab_size (int): The range of possible integers (from 0 to vocab_size-1).

    Returns:
        inputs (torch.Tensor): A matrix of shape (batch_size, seq_len) with random values.
        targets (torch.Tensor): The same matrix, but with each row sorted.
    """
    # Create a batch of random integers. 
    # torch.randint(low, high, size) generates random whole numbers.
    # size=(batch_size, seq_len) means we get a 2D grid (matrix) of numbers.
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Sort each sequence (row) individually.
    # dim=1 tells PyTorch to sort along the second dimension (the columns/sequence).
    # .values extracts the actual sorted numbers (sort() also returns indices).
    targets = inputs.sort(dim=1).values 
    
    return inputs, targets 

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        """
        Initializes the Single-Head Self-Attention layer.
        
        In Self-Attention, every word in a sequence "looks" at every other word 
        to understand context. To do this, we transform each word into three 
        different roles: Query, Key, and Value.
        """
        super().__init__()  # Standard PyTorch boilerplate to initialize the base class
        
        self.dk = d_model  # The dimension (size) of our vectors (e.g., 32 numbers per word)
        
        # These are the 'learnable' parts of the layer. 
        # nn.Linear(in, out) is essentially a matrix of weights that the model 
        # will adjust during training to learn how to transform inputs.
        
        # W_Q (Query): Represents "What am I looking for?"
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        
        # W_K (Key): Represents "What information do I contain?"
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        
        # W_V (Value): Represents "What information should I pass on if I'm a match?"
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # A placeholder to store attention scores so we can visualize them later
        self.last_weights = None 

    def forward(self, X): 
        """
        The forward pass of the attention mechanism.
        X is the input tensor of shape (batch_size, sequence_length, d_model).
        """
        # 1. Linear Projections
        # We transform the same input X into three different spaces: Query, Key, and Value.
        Q = self.W_Q(X) 
        K = self.W_K(X)
        V = self.W_V(X)
        
        # 2. Scaled Dot-Product Attention
        # Calculate scores: How much does each 'Query' match each 'Key'?
        # K.transpose(-2, -1) flips the last two dimensions so we can do matrix multiplication.
        # We divide by sqrt(dk) to keep values from getting too large (which helps training).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)
        
        # 3. Softmax
        # Convert the raw scores into probabilities (0 to 1) that sum to 1.0.
        # This tells us exactly what percentage of attention to pay to each word.
        weights = F.softmax(scores, dim=-1) 
        
        # 4. Save for visualization
        # We 'detach' from the neural network math and convert to a regular NumPy array.
        self.last_weights = weights.detach().cpu().numpy() 
        
        # 5. Output
        # Multiply the weights by the 'Values' to get the final context-aware representation.
        return torch.matmul(weights, V)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff):
        """
        A very simple Transformer model.
        It consists of an embedding layer, one attention layer, 
        a normalization layer, and a feed-forward network.
        """
        super().__init__()

        # 1. Embedding Layer: Turns word IDs into vectors of size d_model.
        # This is like a lookup table where each word has its own unique vector.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Self-Attention: The custom layer we built above.
        # It allows the model to focus on different parts of the input sequence.
        self.attention = SingleHeadSelfAttention(d_model)
        
        # 3. Layer Normalization: Helps with training stability.
        # It ensures that the output of the attention layer has a consistent mean and variance.
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 4. Feed-Forward Network: A simple two-layer neural network.
        # It processes the information extracted by the attention layer.
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model)
        )
        
        # 5. Output Projection: Maps the hidden representation back to word probabilities.
        # This tells us which word from our vocabulary is most likely.
        self.output_proj = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        """
        The flow of data through the Transformer:
        Embedding -> Attention (with Residual) -> FeedForward (with Residual) -> Output
        """
        # 1. Turn IDs into vectors
        h = self.embedding(x)
        
        # 2. Attention Block with "Residual Connection"
        # Notice: h = layer_norm( h + attention(h) )
        # We ADD the original 'h' back to the result of the attention. 
        # This is a 'Residual Connection' - it helps deep networks learn better 
        # by giving the signal a "shortcut" to flow through.
        h = self.layer_norm(h + self.attention(h))
        
        # 3. Feed-Forward Block with "Residual Connection"
        # We do the same thing here: process the data, add it back to the input, and normalize.
        h = self.layer_norm(h + self.ff(h))
        
        # 4. Final Projection
        # Maps our vectors back to the size of the vocabulary so we can pick a word.
        return self.output_proj(h)

torch.manual_seed(42)
model = TinyTransformer(VOCAB_SIZE,D_MODEL,D_FF)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"Parameters : {sum(p.numel() for p in model.parameters())}")
print(f"Trainable Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

losses, accuracies = [], []
snapshot_epochs = [1, 5, 10, 50, 100, 499]
snapshots = {}

for epoch in range(N_EPOCHS):
    # 1. Training Phase
    model.train()
    inputs, targets = generate_sort_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    optimizer.zero_grad()
    logits = model(inputs) # Shape: (batch, seq_len, vocab_size)
    
    # CrossEntropyLoss expects (N, C) for logits and (N) for targets
    # We flatten the batch and sequence dimensions for the loss calculation
    loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    # 2. Evaluation Phase
    model.eval()
    with torch.no_grad():
        # Test on a larger batch to get a stable accuracy
        ti, tt = generate_sort_data(256, SEQ_LEN, VOCAB_SIZE)
        predictions = model(ti).argmax(dim=-1)
        acc = (predictions == tt).float().mean().item()
        accuracies.append(acc)
    
    # 3. Snapshotting for visualization
    # We save the attention pattern of the first sequence in the batch
    if epoch in snapshot_epochs:
        snapshots[epoch] = model.attention.last_weights[0].copy()
    
    if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
        print(f"Epoch {epoch:>4}: loss={loss.item():.4f}, acc={acc:.3f}")

print(f"\nFinal accuracy: {accuracies[-1]:.3f}")

# --- Visualization ---

import seaborn as sns

# Create a grid of heatmaps to show how attention evolves
# As the model learns to sort, you'll see the attention pattern change!
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Attention Pattern Evolution During Training", fontsize=16)

for idx, epoch in enumerate(snapshot_epochs):
    if epoch not in snapshots: continue
    ax = axes[idx // 3][idx % 3]
    sns.heatmap(snapshots[epoch], ax=ax, cmap='Blues', vmin=0, vmax=1,
                annot=True, fmt='.2f',
                xticklabels=[f'p{i}' for i in range(SEQ_LEN)],
                yticklabels=[f'p{i}' for i in range(SEQ_LEN)])
    ax.set_title(f"Epoch {epoch} (acc={accuracies[epoch]:.2f})")

plt.tight_layout()
plt.savefig('attention_evolution.png')
print("Saved attention_evolution.png")

# Also plot Loss and Accuracy curves to see the learning progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss (Lower is Better)")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Accuracy (Higher is Better)")
plt.xlabel("Epoch")

plt.savefig('training_curves.png')
print("Saved training_curves.png")
plt.show()
