"""

Lab 1 : why CNN exist 
========================
This script demontrates :
1. The parameter expolostion in MLPs for images 
2. Translation sensitvity of MLPs 
3. CNN translation invariance via weight sharing 
"""

import numpy as np 
import torch 
import torch.nn as nn 

import matplotlib   
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# 

# The parameter explosiion 

#

def count_mlp_params(img_h , img_w , channels , hidden_units ):
    """ count parameters in a single FC layer from image to hidden """
    # Flatten image -> 1D vector: H x W x C values feed into each hidden unit.
    input_size = img_h * img_w * channels 
    # FC params = (input_size * hidden_units) weights + hidden_units biases.
    return input_size * hidden_units + hidden_units 

def count_cnn_params(kernel_size,c_in , c_out ):
    """ count parameters in one convolutional layer """
    # Each output channel has one kernel per input channel.
    # Kernel weights per output channel = kernel_size^2 * c_in.
    # Total conv params = (kernel_size^2 * c_in * c_out) weights + c_out biases.
    return (kernel_size ** 2 ) * c_in * c_out + c_out # weights + biases 

print("="*60)
print("Parameter count comparision ")
print("="*60)
# Total raw image inputs when flattened for an MLP.
print ( f"\nImage : 224 x224 RGB (224 x 224 x 3 = {224*224*3: ,} inputs )")
print ( f"\nMLP : one FC layer to 1000 hidden units :")
# MLP here connects every pixel-channel value to every hidden unit.
mlp_params = count_mlp_params(224,224,3,1000)
print ( f" {mlp_params: ,} parameters ")
print ( f"\nCNN : one convolutional layer with 3x3 kernel , 3 input channels , 1000 output channels :")
# CNN reuses the same 3x3 filters spatially, so params are independent of image width/height.
cnn_params = count_cnn_params(3,3,1000)
print ( f" {cnn_params: ,} parameters ")
print ( f"\nCNN : two convolutional layers with 3x3 kernel , 3 input channels , 1000 output channels :")
# Recomputed for one layer again to keep direct side-by-side comparison visible.
cnn_params = count_cnn_params(3,3,1000)
print ( f" {cnn_params: ,} parameters ")
print ( f"\nCNN : three convolutional layers with 3x3 kernel , 3 input channels , 1000 output channels :")
# Same per-layer formula; total model params would scale by number of such layers.
cnn_params = count_cnn_params(3,3,1000)
print ( f" {cnn_params: ,} parameters ")