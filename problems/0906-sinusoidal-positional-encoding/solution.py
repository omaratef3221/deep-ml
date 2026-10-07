import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    max_len = seq_len
    d_model = d_model

    pos_matrix = torch.zeros(max_len, d_model)
    i = torch.arange(0, d_model, step = 2).unsqueeze(0)
    pos = torch.arange(0, max_len).unsqueeze(1)

    division_term = 10000**((i)/d_model)

    pos_matrix[:, ::2] = torch.sin(pos/division_term)
    pos_matrix[:, 1::2] = torch.cos(pos/division_term)

    return pos_matrix