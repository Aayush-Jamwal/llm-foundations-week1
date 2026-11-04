# nanoGPT From Scratch: Training & Loss Diagnostics

A lightweight implementation of an autoregressive, decoder-only Transformer language model trained on character-level data (Tiny Shakespeare) from first principles.

---

## Lab 1: nanoGPT Implementation & Training

Lab 1 focuses on building and training a decoder-only Transformer from scratch using character-level tokenization. The model architecture follows Andrej Karpathy's *nanoGPT* design and is scaled for efficient local execution and verification.

### Architectural Specifications & Hyperparameters

- **Architecture Type:** Autoregressive Causal Transformer Decoder (GPT-style)
- **Normalization:** Pre-Layer Normalization (Pre-LN) applied before Multi-Head Attention and MLP sub-layers
- **Vocabulary Size ($V$):** 65 unique characters (Tiny Shakespeare corpus)
- **Context Length ($T$):** 128 tokens
- **Embedding Dimension ($d_{\text{model}}$):** 128
- **Attention Heads ($h$):** 4 ($d_k = 32$ per head)
- **Transformer Layers ($L$):** 4 stacked blocks
- **Feed-Forward Multiplier:** $4\times$ ($128 \to 512 \to 128$) with ReLU activation
- **Regularization:** Dropout $p = 0.1$
- **Total Parameters:** $\sim 0.8\text{M}$
- **Optimizer:** AdamW ($\text{lr} = 3\times 10^{-4}$, $\text{zero\_grad(set\_to\_none=True)}$)
- **Batch Size:** 64
- **Training Iterations:** 3,000 steps

---

## Training Convergence & Loss Diagnostics

The model tracks train and validation loss every 100 iterations over 200 random batches:

- **Initial Loss:** $\sim 4.17$ (matches theoretical random initialization: $-\ln(1/65) \approx 4.17$)
- **Final Validation Loss:** $\sim 1.65 - 1.75$

### Loss Curve

![Loss Curve](assets/loss_curve.png)

---

## References

1. **Karpathy, A.** (2023). *Let's build GPT: from scratch, in code, spelled out.* nanoGPT repository: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
