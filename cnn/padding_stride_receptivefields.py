""" 
This script demonstrates fundamental concepts in Convolutional Neural Networks (CNNs):
1. Padding: How adding borders around input images controls the output spatial size.
2. Stride: How the step size of the filter movement downsamples (reduces) feature maps.
3. Receptive Field: A conceptual look at how layers interact with input regions.

Concepts for Beginners:
- Convolution: A mathematical operation where a 'filter' or 'kernel' slides over an image to extract features.
- Output Size: The dimensions of the resulting image after the convolution.
"""

# Import necessary libraries
import numpy as np           # NumPy is used for numerical operations on arrays
import matplotlib           # Matplotlib is the standard plotting library in Python
matplotlib.use('Agg')       # Use a non-interactive backend (good for servers/headless environments)
import matplotlib.pyplot as plt # Interface for creating plots
import torch                # PyTorch: The main Deep Learning framework used here
import torch.nn as nn       # nn (Neural Network) module contains layers like Conv2d

# ---------------------------------------------------------
# Part 1: Calculating Convolution Output Size
# ---------------------------------------------------------

def output_size(H, K, p, s):
    """ 
    Formula to compute the output dimension (Height or Width) of a convolution.
    
    Parameters:
    H : Input Size (Height or Width)
    K : Kernel/Filter Size (usually 3 or 5)
    p : Padding (pixels added to each side)
    s : Stride (step size of the filter)
    
    Formula Breakdown:
    (H - K + 2*p) / s + 1
    
    Note on '//' in Python:
    The '//' operator is 'floor division' or 'integer division'. 
    It divides and rounds down to the nearest whole number.
    """
    return (H - K + 2 * p) // s + 1


# Print headers for clarity
print("=" * 60)
print("DEMO: HOW PADDING AND STRIDE CONTROL OUTPUT SIZE")
print("=" * 60)

# Setup initial parameters
H = 32  # We'll assume a 32x32 pixel input image
K = 3   # A standard 3x3 convolution kernel (filter)

# F-strings (f"...") allow us to embed variables directly into strings using {}
print(f"\nInput size: {H}x{H} | Kernel: {K}x{K}")

# Column headers for the output table
# :>10 means "right-align in a space of 10 characters"
print(f"\n{'Padding':>10} | {'Stride': >8} | {'Output Size': >14}")
print("-" * 40)

# Loop through different padding types and strides
# (0, 'valid') means No Padding. The output will be smaller than input.
# (1, 'same')  means Padding = 1. With K=3 and s=1, the output remains 32x32 (same as input).
for p, name in [(0, 'valid'), (1, 'same'), (2, 'full-ish')]:
    for s in [1, 2]:
        # Calculate the output size using our function
        out = output_size(H, K, p, s)
        
        # Print the results in a formatted table
        # out:>5 means right-aligned with 5 spaces
        print(f"{p:>3} ({name:7}) | s={s}      | {out:>5}x{out:>5}")

print("\nKey Takeaways:")
print("- 'Valid' padding reduces image size.")
print("- 'Same' padding (p=1 for 3x3 kernel) preserves size when stride=1.")
print("- Stride > 1 effectively halves the resolution (downsampling).")


# ---------------------------------------------------------
# Part 2: Visualizing the Stride Effect (Downsampling)
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("VISUALIZING STRIDE EFFECT ON FEATURE MAP SIZE")
print("=" * 60)

# 1. Create a simple grayscale image: A 32x32 Checkerboard
# np.indices creates a grid of coordinates.
# .sum(axis=0) % 2 creates the alternating 0s and 1s pattern.
size = 32
checkerboard = np.indices((size, size)).sum(axis=0) % 2
checkerboard = checkerboard.astype(np.float32) # Convert to floating point (required for DL)

# 2. Prepare the data for PyTorch
# PyTorch expects images in the format: (Batch Size, Channels, Height, Width)
# - Batch Size: How many images are processed at once (we have 1)
# - Channels: Colors (1 for grayscale, 3 for RGB). We have 1.
# .unsqueeze(0) adds a new dimension of size 1 at the specified position.
img_t = torch.tensor(checkerboard).unsqueeze(0).unsqueeze(0) # Becomes (1, 1, 32, 32)

# Create a figure with 4 subplots side-by-side
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot the original image in the first subplot
axes[0].imshow(checkerboard, cmap='gray')
axes[0].set_title(f"Input\n({size}x{size})")
axes[0].axis('off')

# 3. Apply convolutions with different strides
# We'll use a fixed kernel size (K=3) and padding=1 to keep things simple.
strides = [1, 2, 4]

for i, s in enumerate(strides):
    # Define a simple convolution layer
    # in_channels=1, out_channels=1, kernel_size=3
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=s, padding=1, bias=False)
    
    # Initialize weights to 1/9 (average) so it acts like a blur/downsample
    # This helps keep the pattern visible
    nn.init.constant_(conv.weight, 1.0 / 9.0)
    
    # Apply the convolution
    # torch.no_grad() tells PyTorch we aren't training, so it saves memory
    with torch.no_grad():
        output = conv(img_t)
    
    # The output is still a 4D tensor (1, 1, H_out, W_out)
    # We need to remove the first two dimensions to plot it: .squeeze()
    out_img = output.squeeze().numpy()
    
    # Plot the result
    idx = i + 1 # Subplots are 0-indexed, so 1, 2, 3
    axes[idx].imshow(out_img, cmap='gray')
    axes[idx].set_title(f"Stride {s}\n({out_img.shape[0]}x{out_img.shape[1]})")
    axes[idx].axis('off')

plt.suptitle("Effect of stride on feature Map size ",fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig("stride_visualization.png")
print("[SUCCESS] Visualization saved as 'stride_visualization.png'")
print("Notice how 'Stride 2' makes the image 16x16, and 'Stride 4' makes it 8x8.")
print("This 'Downsampling' is why CNNs can see larger patterns in deeper layers.")
