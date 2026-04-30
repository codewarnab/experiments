#Project 1 : Bare - Bones scaled Dot product Attention 
#Goal : Implement Attention *(Q,K,V) = softmax(QK^T / sqrt(d_k))V 
#using only numpyt - no pytorch , no abstraction 

#we verify our result agains the worked "cat sat mat" example from 
#Module 6 , Section 3.5 

import numpy as np 

#step 0 :Define the input matrices 
#These Q,K,V matrices come directly from module 6 worked example 
#each row is one token . There are 3 tokens : "cat" , "sat" , "mat"
#each token has 2 dimentional representation 

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

#step 1 : Compute the dimenstion sof the key / query space 
dk = Q.shape[1] # = 2 for our example 
print(f)