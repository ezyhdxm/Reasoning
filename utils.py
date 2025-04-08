import torch
from torchinfo import summary

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