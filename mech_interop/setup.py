"""

setup.py 

One shared module every experiment does 
    from setup import model , tokenizer , devicde , run , tokens , logits , cache 

What this file does 
1. Loads Gpt2 small(117M)  via transformerLens 
2. Defines 'run(prompt)' which runs a forwards pass and returns 
(tokens , logits , cache ) . 'cache' is the improtant object -- 
it holds every intermediate activattions from every layer 
3. prints the model's dimenstions so you can refer to them 

GPT - 2 small dimenstions (you need these for indexing ) : 
n_layer = 12 (transformer blocks )
n_heads = 12 ( attention heads per layer )
d_model = 768 ( residual stream dimenstion)
d_head = 64 ( each head's QKV dimesntions)
d_mlp = 3072 (MLP hidden dimenstions , 4 * d_model )
vocab_size = 50257 
"""

import torch 
from transformer_lens import HookedTransformer
#Load GPT2 first run downloads 500 mb cached afterwards 
# #run embed = True tells transformerlens to also cache embedding layer outputs 
model = HookedTransformer.from_pretrained("gpt2-small",center_unembed = True , center_writing_weights = True , fold_ln = True )
#center_unembed means the unembed layer is centered around zero
#center_writing_weights means the writing weights are centered around zero
#fold_ln means the layer norm is folded into the residual stream
# benefits of using this is that the model is easier to work with and debug\


device = 'cuda' if torch.cuda.is_available() else "cpu"
model = model.to(device)

tokenizer = model.tokenizer 

print(f"Loaded :{model.cfg.model_name}")
print(f" Layers : {model.cfg.n_layers}")
print(f" Heads : {model.cfg.n_heads}")
print(f" d_model : {model.cfg.d_model}")
print(f" d_head : {model.cfg.d_head}")
print(f" d_mlp : {model.cfg.d_mlp}")
print(f" vocab_size : {model.cfg.d_vocab}")
print()

def run(prompt:str ):
    """ 
    Run a forwards pass on "prompt" and return everything 

    Returns :
            tokens -- integer tensor of shape (1,seq_len)
            logits -- model output , shape (1 , seq_len , vocab_size)
            cache -- ActivationCache object . Access any activation like : 
                cache [" resid_post " , layer_idx ] # residual stream after
                block
                 cache [" attn ", layer_idx ] # attention patterns
                 cache [" mlp_post " , layer_idx ] # MLP output
                 cache [" hook_embed "] # token embeddin
    
    """
    tokens = model.to_tokens(prompt) 
    tokens = tokens.to(device) 

    #run_with_cache records all activations in "cache"
    # this is the core api you will use in every experiment 
    logits , cache = model.run_with_cache(tokens)
    return tokens , logits , cache 
    
    













