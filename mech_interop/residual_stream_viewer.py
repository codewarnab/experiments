import torch.nn.functional as F 
from setup import model,run 
"""
concept what is residual stream 
every transformer block adds output to the existsing residual stream vector 
By the time tokens reaches layer 12 its representation is the sum of contriutions 
from all previous blocks , this experiment lets us look at the this numericially 
how much does the representation of a single token change between layer 0 or layer 11 
"""

"""
Question we are answering 
How much does the representation of a single token change between layer 0 or layer 11
as it moves through the residual stream

why this matters 
if the residual stream barely changes bwteen layers 8 - 11 , those layers are doing 
little for this particular prompt . If it changes a lot between layers 0 - 3 
early layers are doing most of the work . 

Key intution 
The residual stream at layer l is 
x_l = embed(token) + sum of (attention_output + MLP_output )
 for layer 0 ... L-1 


Each block adds a delta The total represenation drifts as deltas accumulate 
coside similariy measures the direction change not maginitude A similarity 
of 1.0 means no direction change ( block addded zero meaningful information)
A similarity of 0.2 means the representation has changed dramatically 


"""


PROMPT = "The cats are"
tokens , logits , cache = run(PROMPT)
#cache["resid_post",L] gives the residual stream after layer L
# shape : (1 , seq_len , d_model) = 1.T , 768 
# 
# Get token strinsg for display 
token_strs = [model.tokenizer.decode([t]) for t in tokens[0].tolist()]
seq_len = tokens.shape[1]
n_layers = model.cfg.n_layers

print(f"Prompt: {PROMPT}")
print(f"seq_len: {seq_len}")
print(f"n_layers: {n_layers}")
print(f"Seq length :{seq_len} , d_model : {model.cfg.d_model}")

# We will track how the representation of the last token evolves 
# The last toke's representation is what produces the next word predictions 
# #position index of the last token 
last_pos = seq_len - 1

#Grab the residual stream at every layer for the lasty token 
# cache["resid_post",L] has shape (1, seq_len, d_model)
# we index [0, last_pos, :] to get the last token's representationlast_token_repr = [cache["resid_post", L][0, last_pos, :] for L in range(n_layers)]r
residuals = [] 
for L in range(n_layers): 
    resid = cache["resid_post", L][0, last_pos, :] #shape (768 , )
    residuals.append(resid)


#Also grab the final residual (= residuals at layer 11 , after layer norm )
# This is the vector that gets multiplied by the unembedding matrix to produce logits 
final_resid = residuals[-1] # layer 11 
print("Cosine similariy between each layer's residual and the final layer'rs residual ")
print("for the last token position ")
print("-" * 50 )

for L , resid in enumerate(residuals): 
    #F.cosine_similarity expects batcehc input ,we unsqueeze to add add a bacth dim 
    cos_sim = F.cosine_similarity(resid.unsqueeze(0) , final_resid.unsqueeze(0))
    sim_val = cos_sim.item()
    # Visul bar to make the numbers easier interpret at a glance 
    bar_len = int(sim_val * 30)
    bar = "█" * bar_len
    print(f"Layer {L}: cosine similarity = {sim_val:.4f} | {bar}")

print()

#Now lets look at the Magnitude (L2 norm ) of the residual at each layer 
# The norm grows as more blocks add their outputs , This shpws you which layers
# #contiribute most to the residual stream in terms of the raw vector magnitude 
print("L2 norm of residual at each layer (last token)")
print(f"{'Layer':<10} | {'L2 Norm':<10}")
print("-" * 20)
for L, resid in enumerate(residuals):
    norm = resid.norm().item()
    print(f"{L:<10} | {norm:<10.4f}")




"""
You will see cosine similariy low in early layers ( the represenation is still 
being built up from the prompt tokens ) and high in later layers ( the represenation 
has established ) . The exact pattern depends on the prompt 
for factual completetions liek this one middle layers (4-8) tens to do the most work 
"""





print ()
print (" WHAT TO TRY NEXT :")
print (" 1. Change PROMPT to something factual vs. something creative .")
print (" Does the cosine similarity profile change ?")
print (" 2. Track a MIDDLE token instead of the last one.")
print (" 3. Print which layer has the largest cosine distance from the next layer.")
print("That is the most 'active' layer for this prompt.")