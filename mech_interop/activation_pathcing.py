"""
Activation patching (Causal Tracing)
Quetsion we are answering 
which layers components are causally responsible for GPT-2 
predicting 'Paris' after 'The capital of France is '

soecifically when we corrupt 'france' -> 'Germany', the model chanages its prediction to 'Berlin' we then patch bcak the clean activations one component at a time and check : does the prediction recover to 'Paris' The layer /component that restores the preedictin is the causal mechanism 

causal mechanism is the specific part of the nerual netweork ( a certain layer or attention head ) that is directly responsible for a particular output 

Why this matters:
    This experiment is the published methord from meng et al 2022 'ROME' (Rank One model editing ) and Goldowsky-Dill et al 2023 'Localizing Model Behaviour ; 

Key insight :
    Causality != correlation just because an activation is different between clean and corrupted does not mean it CAUSES the output to differ . Patching tests causallity : we forcibly restore an activation and see if the output changes accordingly 

"""

import torch 
import torch.nn.functional as F 
from setup import model , run 

#clean prompt : should predict 'Paris'
CLEAN_PROMPT = "The capital of France is"
#corrupted prompt : should predict 'Berlin'
CORRUPTED_PROMPT = "The capital of Germany is"

#Target token we want to track 
TARGET_TOKEN = " Paris"
TARGET_ID = model.to_single_token(TARGET_TOKEN)

def get_logit_for_token(logits,token_id):
    """Extract the logit for a specific token at the last sequence position"""
    return logits[0,-1, token_id].item()

#Step 1 : Run clean and corrupted forward passes 
tokens_clean,logits_clean , cache_clean = run(CLEAN_PROMPT)
tokens_corrupted,logits_corrupted , cache_corrupted = run(CORRUPTED_PROMPT)

logit_clean = get_logit_for_token(logits_clean, TARGET_ID)
logit_corrupted = get_logit_for_token(logits_corrupted, TARGET_ID)

print ( f" Clean prompt : '{ CLEAN_PROMPT } '")
print ( f" Corrupted prompt : '{ CORRUPTED_PROMPT } '")
print ( f" Target token : '{ TARGET_TOKEN } ' (id ={ TARGET_ID })")
print ()
print ( f" Clean logit for '{ TARGET_TOKEN } ': { logit_clean :.3f}")
print ( f" Corrupted logit for '{ TARGET_TOKEN } ': { logit_corrupted :.3f}")
print ( f" -> Gap we are trying to recover : { logit_clean - logit_corrupted :.3f}")
print ()

#step 2 :Patch the residual stream layer by layer 
# for each layer L we run a new forward pass on the corrupted prompt but replace the redidual stream at layer L with CLEAN residual stream 
# 
# Implementaion:Transformerlens hooks lets us inject any activation 
# we use model.run_with_hooks() with a hook function that replaces the 

n_layers = model.cfg.n_layers
last_pos = tokens_corrupted.shape[1] - 1  # last token positions in corrupted prompt 
print("Residual stream patching : recovering 'Paris' logit by layers")
print("(higher recovered logit = this layer is more casually important)")
print()
print (f"{ ' Layer ': >7} | { ' Patched Logit ': >14} | { ' Recovery % ': >11} | Visual ")
print("-" * 70)

layer_results = []

for PATCH_LAYER in range(n_layers):
    # the clean activatiin we want to inject at PATCH_LAYER , last token 
    clean_resid_at_layer = cache_clean["resid_post",PATCH_LAYER][0,last_pos:]

    # ^ shape : ( d_model,) = (768,)
    #Hook function : replaces the residuals at the target layer with the clean one 
    # value has shape (batch , seq , d_model) ; we only replacce the last token
    def hook_fn(value, hook):
        value[0, last_pos:] = clean_resid_at_layer
        return value

    hook_name = f"blocks.{PATCH_LAYER}.hook_resid_post"
    #Run the corrupted prompt but with the patched residual at PATCH_LAYER 
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            tokens_corrupted,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    patched_logit = get_logit_for_token(patched_logits,TARGET_ID )
    
    # recovery : how much of the gap(clean - corrupted) did we recover?
    gap = logit_clean - logit_corrupted
    if abs(gap) > 1e-6:
        #Calculate recovery : 1.0  means full restoration to clean , 0.0 means no change from corrupted 
        recovery = (patched_logit - logit_corrupted) / gap 
    else :
        recovery = 0.0 
    
    # Store the results (Layer ID, the new logit value, and the recovery percentage)
    layer_results.append((PATCH_LAYER, patched_logit, recovery))
    
    # Prepare a visual progress bar (max 30 characters wide) based on the recovery %
    bar_len = max(0, int(recovery * 30))
    bar = "#" * bar_len + "." * max(0, 30 - bar_len)
    
    # Print a formatted row showing: Layer Index | Patched Logit Value | Recovery % | Visual Bar
    print(f" L={PATCH_LAYER:>2} | {patched_logit:>14.3f} | {recovery:>10.1%} | {bar}")    


print ()
best = max( layer_results , key = lambda x : x [2])
print ( f" Most causal layer : Layer { best [0]} ({ best [2]:.1%} recovery )")
print ()
print (" INTERPRETATION :")
print (" The layer with highest recovery % is where the factual association ")
print (" 'France -> Paris ' is stored in GPT -2. Meng et al. (2022) found that ")
print (" for factual recall , this is typically mid -to - late MLP layers .")
print ()
print (" WHAT TO TRY NEXT :")
print (" 1. Instead of patching the full residual stream , patch ONLY the MLP output :")
print (" hook_name = f'blocks .{L}. hook_mlp_out '")
print (" This tells you whether attention or MLP is more causally responsible .")
print (" 2. Patch ONLY specific attention heads by using the head output hook .")
print (" 3. Try a different factual pair : 'Rome is in ' / 'Madrid is in '.")

"""
You will see that most layers give low recovery (the Paris logit barely moves). One or two layers
will show 6090%+ recovery. Those are the layers that causally implement the France→Paris
association. This is precisely the method used in the ROME paper to nd the layers to edit
in order to implant new factual associations into a model.

"""