"""Linear probing
A probing classifier trains a amsll linear model (logitic regression )
to predict some property (eg "is this token a city name") from the residual 
stream at layer L . if a simpler linear model  can decode that property from those activations , then the concept is linearly representted that layer . If it cannot the concept is either absett or encoded nonlinearly . Probing is how you check whether  a concept exists in the model at a given layer before trying to manuplulate it 

why this matter 
Chris Olah's core hypothesis ) is the Linear Representation Hypothesis 
high level concepts are encoded as directions in activations space 
if true a linear probbe should decode them  . If the probe accuracy at layer L is 95 % that concept is strongly linearly present at layer L 


Method 
1. Build a dataset : prompts where the next token is a country and prompts where the next tokens is not a country 
2. Run all prompts through GPT 2 grab the residual stream at layer L for the last token position 
3. Train a logistic regression on (residual , label ) pairs 
4. Repeat for every layer and plot accuracy vs layer 
"""
import torch 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from setup import run,model 

# Dataset : country vs. non - country next tokens
# Positive examples : the next word is a country name


COUNTRY_PROMPTS = [
    "The official language of Germany is", # next : German / Germany
    "The president lives in", # -> America ( country context )
    "I want to visit ", # generic but we label manually 
    " Tourists often travel to",
    "The national dish of Japan is",
    "The GDP of China has grown ",
    "The flag of Brazil is",
    "The population of India is",
    "The rivers in France flow ",
    "The mountains of Switzerland are",
    " Immigration policy in Canada affects ",
    "The history of Egypt spans ",
    "The history of Egypt spans ",
]
# Negative examples : the next word is NOT a country - related concept
NOT_COUNTRY_PROMPTS = [
    "The cat sat on the",
    "She opened the door and ",
    " Mathematics is the language of",
    "The colour of the sky is",
    "He picked up the",
    " Yesterday I ate a",
    "The algorithm runs in",
    "She smiled and said ",
    "The book was written in",
    "The temperature dropped below ",
    "He turned off the",
    "The music played softly in",
]

ALL_PROMPTS = COUNTRY_PROMPTS + NOT_COUNTRY_PROMPTS
#Labels 1 = country  contentxt , 0 = not country related
ALL_LABELS = [1] * len(COUNTRY_PROMPTS) + [0] * len(NOT_COUNTRY_PROMPTS)

print(f"Dataset : {len(COUNTRY_PROMPTS)} country prompts, {len(NOT_COUNTRY_PROMPTS)} non-country prompts")
print()

def get_residuals_at_layer(prompts, layer): 
    """
    Run all prompts through GPT-2 extract the residual stream at 'layer' for the LAST token position 

    Returns a numpy array of shape(n_prompts, d_model)
    """
    residuals = []
    for prompt in prompts: 
        tokens, logits, cache = run(prompt)
        last_pos = tokens.shape[1] - 1 
        #residual stream at this layer for the last token : shape(d_model)
        resid = cache["resid_post", layer][0, last_pos, :]
        #detach from computation graph and convert to numpy for sklearn 
        residuals.append(resid.detach().cpu().numpy())
    return np.array(residuals) # shape(n_prompts, d_model)

# Train a probe at every layer , record accuracy 
print("Training linear probe at each layer (this takes ~30 seconds on CPU):")
print(f"{'Layer':>7} | {'Train Acc':>10} | {'Val Acc':>10} | Visual")
print("-" * 55)

layer_accuracies = []
for L in range(model.cfg.n_layers):
    # Get activations at this layer for all prompts 
    X = get_residuals_at_layer(ALL_PROMPTS, L) # (n_prompts, 768)
    y = np.array(ALL_LABELS) # (n_prompts,)

    # Train / val split. With only 24 examples this is rough but illustrative.
    # In a real experiment you would have hundreds or thousands of examples.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardise: zero mean, unit variance. VERY important for logistic regression
    # on high-dimensional activations -- otherwise features with large magnitude
    # dominate and the probe trains poorly.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Logistic regression with L2 regularisation.
    # max_iter=1000: sometimes needs more iterations on 768-dim input.
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train_sc, y_train)

    train_acc = probe.score(X_train_sc, y_train)
    val_acc = probe.score(X_val_sc, y_val)
    layer_accuracies.append((L, train_acc, val_acc))

    bar_len = int(val_acc * 30)
    bar = "#" * bar_len + "." * (30 - bar_len)
    print(f"L={L:>2} | {train_acc:>10.3f} | {val_acc:>10.3f} | {bar}")

print()
best_layer = max(layer_accuracies, key=lambda x: x[2])
print(f"Best validation accuracy: Layer {best_layer[0]} with {best_layer[2]:.3f}")
print()
print("INTERPRETATION:")
print("Layers where val_acc >> 0.5 (random): the concept is linearly encoded there.")
print("If val_acc peaks in layer 5 and stays high: the concept is computed by layer 5")
print("and is preserved in later layers. If it peaks and drops: it is transient.")
print()
print("WHAT TO TRY NEXT:")
print("1. Replace the country/not-country dataset with:")
print("   positive: prompts ending with a male pronoun context")
print("   negative: prompts ending with a female pronoun context")
print("   Do pronouns encode gender linearly?")
print("2. Build a dataset of sentiment (positive/negative reviews).")
print("3. After finding the best layer, inspect probe.coef_[0] -- this is the")
print("   768-dim direction vector. Project it onto the unembedding matrix W_U to")
print("   see which tokens live in the 'positive' direction.")

