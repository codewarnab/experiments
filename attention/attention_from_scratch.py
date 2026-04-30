#Project 1 : Bare-Bones scaled Dot product Attention 
#Goal : Implement Attention *(Q,K,V) = softmax(QK^T / sqrt(d_k))V 
#using only numpy - no pytorch , no abstraction 

#we verify our result against the worked "cat sat mat" example from 
#Module 6 , Section 3.5 


import numpy as np 

#step 0 :Define the input matrices 
#These Q,K,V matrices come directly from module 6 worked example 
#each row is one token . There are 3 tokens : "cat" , "sat" , "mat"
#each token has 2 dimensional representation 

Q =  np.array([
    [1,0], # Query for "cat" 
    [0,1], # Query for "sat"
    [1,1]  # Query for "mat"
],dtype=float)

K = np.array([
    [1,0], # Key for "cat" 
    [0,1], # Key for "sat"
    [1,0]  # Key for "mat"
],dtype=float)

V = np.array([
    [1,0], # Value for "cat" 
    [0,1], # Value for "sat"
    [1,1]  # Value for "mat"
],dtype=float)

print("===Input Matrices ===\n"
"Q =\n",Q,"\n"
"K =\n",K,"\n"
"V =\n",V,"\n")

#step 1 : Compute the dimensions of the key / query space 
dk = Q.shape[1] # = 2 for our example 
print(f"\n dk = {dk} (divide by the sqrt of this )") ; 

#Step 2 compute raw attention scores -- QK^T 
raw_scores = Q @ K.T 
print("\nStep 2: Raw Scores (QK^T)") 
print(raw_scores) 

#Step 3 : Scale the scores 
scaled_scores = raw_scores / np.sqrt(dk)
print("\nStep 3: Scaled Scores")
print(scaled_scores) 

#Step 4 : Define softmax function 
def softmax(x, axis=-1):
    """
    Softmax converts a vector of raw scores (logits) into a probability distribution.
    The resulting values are between 0 and 1 and sum to 1.
    """
    # Step 1: Numerical Stability Trick
    # We subtract the maximum value from each row before exponentiating.
    # Why? Large values in np.exp() can lead to 'overflow' (infinity).
    # Mathematically, softmax(x) == softmax(x - constant).
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # Step 2: Exponentiation
    # This turns all values positive. It also makes the largest values much larger 
    # than the others (the 'max' part of 'softmax').
    exp_x = np.exp(x_shifted)
    
    # Step 3: Normalization
    # We divide each exponentiated value by the sum of all exponentiated values 
    # in that row. This ensures that the final values sum to exactly 1.0.
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

#step 5 Apply the row-wise softmax to get attention weights 
attention_weights = softmax(scaled_scores,axis=-1) 
print("\nStep 5: Attention Weights (Softmax)")
print(np.round(attention_weights,4))
print("\n Row sums (should all be 1.0 ):",np.round(attention_weights.sum(axis=1),6))

#Step 6 : Compute the final output -- weighted sum of values 
output = attention_weights @  V 
print("\nStep 6 : Final output ( weighted avg of V)")
print(np.round(output,4))
# shape of the output 
print(f"Shape of the output : {output.shape} -- same as input (n_samples , d_model)")



print("\n\n Attention Pattern for 'Cat' (first row of weights)")
print(f"Cat pays attention to : {attention_weights[0]}") 