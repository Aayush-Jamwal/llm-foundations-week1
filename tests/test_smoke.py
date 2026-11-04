import torch
from src.train_gpt import GPTLanguageModel

def test_model_initialization():
    model = GPTLanguageModel()
    dummy_input = torch.zeros((1, 8), dtype=torch.long)
    logits, loss = model(dummy_input)
    assert logits.shape == (1, 8, model.head_proj.out_features)
    print("Smoke test passed: Model initialized and generated logits.")
