import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
batch_size = 64          # Independent sequences processed in parallel
context_len = 128        # Maximum sequence length for context
train_steps = 3000       # Total iterations
eval_every = 100         # Evaluate every 100 steps
learning_rate = 3e-4     # Learning rate for AdamW
eval_batches = 200       # Batches to average for validation loss
embed_dim = 128          # Feature dimensions per token
num_heads = 4            # Attention heads (128 / 4 = 32 dim per head)
num_layers = 4           # Stacked Transformer blocks
dropout_prob = 0.1       # Dropout regularization

compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
print(f"Executing on: {compute_device}")

# ---------------------------------------------------------
# 1. Dataset & Tokenizer Setup
# ---------------------------------------------------------
data_path = 'data/input.txt' if os.path.exists('data/input.txt') else 'input.txt'
with open(data_path, 'r', encoding='utf-8') as f:
    raw_corpus = f.read()

unique_chars = sorted(list(set(raw_corpus)))
vocab_size = len(unique_chars)

char_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
id_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}

encode_text = lambda s: [char_to_id[c] for c in s]
decode_ids = lambda ids: ''.join([id_to_char[i] for i in ids])

all_tokens = torch.tensor(encode_text(raw_corpus), dtype=torch.long)
split_idx = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:split_idx]
val_tokens = all_tokens[split_idx:]

def fetch_batch(split_type):
    source_tokens = train_tokens if split_type == 'train' else val_tokens
    random_starts = torch.randint(len(source_tokens) - context_len, (batch_size,))
    x_batch = torch.stack([source_tokens[start:start + context_len] for start in random_starts])
    y_batch = torch.stack([source_tokens[start + 1:start + context_len + 1] for start in random_starts])
    return x_batch.to(compute_device), y_batch.to(compute_device)

@torch.no_grad()
def evaluate_splits(model):
    loss_summary = {}
    model.eval()
    for split_name in ['train', 'val']:
        batch_losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            x_eval, y_eval = fetch_batch(split_name)
            _, eval_loss = model(x_eval, y_eval)
            batch_losses[k] = eval_loss.item()
        loss_summary[split_name] = batch_losses.mean()
    model.train()
    return loss_summary

# ---------------------------------------------------------
# 2. Architecture
# ---------------------------------------------------------
class AttentionHead(nn.Module):

    # Computes Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k)) @ V
    #Masks out future tokens using a lower-triangular matrix
    def __init__(self, head_dim):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(context_len, context_len)))
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key_proj(x)   # (B, T, head_dim)
        q = self.query_proj(x) # (B, T, head_dim)
        scores = q @ k.transpose(-2, -1) * (C ** -0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        v = self.value_proj(x)
        return attention_weights @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.head_list = nn.ModuleList([AttentionHead(head_dim) for _ in range(n_heads)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.head_list], dim=-1)
        return self.dropout_layer(self.output_proj(combined))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        head_dim = dim // n_heads
        self.self_attn = MultiHeadSelfAttention(n_heads, head_dim)
        self.feed_forward = PositionWiseFeedForward(dim)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm_attn(x))
        x = x + self.feed_forward(self.norm_ffn(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(context_len, embed_dim)
        self.transformer_stack = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads=num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_indices, targets=None):
        B, T = token_indices.shape
        tok_vecs = self.token_embeddings(token_indices)
        pos_vecs = self.position_embeddings(torch.arange(T, device=compute_device))
        hidden_states = tok_vecs + pos_vecs
        hidden_states = self.transformer_stack(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.head_proj(hidden_states)

        if targets is None:
            total_loss = None
        else:
            B, T, C = logits.shape
            total_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, total_loss

    def generate(self, token_indices, max_new_tokens):
        for _ in range(max_new_tokens):
            context_crop = token_indices[:, -context_len:]
            logits, _ = self(context_crop)
            last_token_logits = logits[:, -1, :]
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_indices = torch.cat((token_indices, next_token), dim=1)
        return token_indices

# ---------------------------------------------------------
# 3. Execution, Checkpointing & Plot Export
# ---------------------------------------------------------
if __name__ == '__main__':
    model = GPTLanguageModel().to(compute_device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = {'step': [], 'train': [], 'val': []}

    print("Beginning training run...")
    for step in range(train_steps):
        if step % eval_every == 0 or step == train_steps - 1:
            losses = evaluate_splits(model)
            history['step'].append(step)
            history['train'].append(losses['train'].item())
            history['val'].append(losses['val'].item())
            print(f"Step {step:4d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")

        xb, yb = fetch_batch('train')
        _, batch_loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/ckpt.pt')
    print("Checkpoint saved to checkpoints/ckpt.pt")

    # Plot loss curve
    os.makedirs('assets', exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(history['step'], history['train'], label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(history['step'], history['val'], label='Val Loss', color='#ff7f0e', lw=2)
    plt.title('NanoGPT Training & Validation Loss Curve', fontsize=12, pad=10)
    plt.xlabel('Optimization Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/loss_curve.png', dpi=300)
    print("Loss curve saved to assets/loss_curve.png")

    # Test generation
    seed_context = torch.zeros((1, 1), dtype=torch.long, device=compute_device)
    print("\n--- SAMPLE GENERATION ---\n")
    print(decode_ids(model.generate(seed_context, max_new_tokens=400)[0].tolist()))
