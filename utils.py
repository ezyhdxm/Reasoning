import torch
from torchinfo import summary
import torch.nn.functional as F

def tabulate_model(model: torch.nn.Module, seq_len: int, batch_size: int, device: str) -> str:
    dummy_data = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    try:
        info = summary(model, 
                       input_data=dummy_data, 
                       depth=3, 
                       col_names=["input_size", "output_size", "num_params"])
        return str(info)
    except Exception as e:
        return f"Could not tabulate model: {e}"


def sample_from_transformer(model, batch, mask, tokenizer=None, temperature=1.0, top_k=None):
    """
    model: Transformer with output logits of shape (B, T, vocab_size)
    input_ids: LongTensor of shape (1, T) — starting token sequence
    k: how many tokens to sample
    tokenizer: (optional) for decoding results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = batch.to(device)
    k = mask.sum().item()
    model.eval()
    seq_len = batch.shape[1]
    batch = F.pad(batch, pad=(k, 0), value=15)
    print(batch.shape)
    generated = batch[:,:seq_len].clone()

    for t in range(k):
        # Get model output for current sequence
        with torch.no_grad():
            output, _ = model(generated[:,t:seq_len+t])  # logits: (1, T, vocab_size)

        logits = output[:, -1, :]  # get logits for the last token

        # Apply temperature
        logits = logits / temperature

        # (Optional) Top-k filtering
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            mask = logits < values[:, [-1]]
            logits[mask] = float('-inf')

        probs = F.softmax(logits, dim=-1)  # shape (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # shape (1, 1)

        # Append sampled token to sequence
        generated = torch.cat((generated, next_token), dim=1)

    if tokenizer:
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated