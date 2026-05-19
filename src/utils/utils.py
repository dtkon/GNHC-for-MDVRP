import math
import random
from typing import Any, Deque, Dict, Iterator, Union, cast
import torch
from torch import Tensor, nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


def clip_grad_norms(
    param_groups: list[dict], max_norm: float = math.inf
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            (
                max_norm if max_norm > 0 else math.inf
            ),  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [
            min(g_norm, torch.tensor(max_norm), key=lambda x: x.item())
            for g_norm in grad_norms
        ]
        if max_norm > 0
        else grad_norms
    )
    return grad_norms, grad_norms_clipped


def get_parameter_number(net: nn.Module) -> Dict[str, int]:
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def batch_slicer(total: int, parallel: int) -> Iterator[tuple[int, int]]:
    assert total >= 0 and parallel >= 1

    start = 0
    remain = total
    while (remain := remain - parallel) > (-parallel):
        if remain >= 0:
            pick_count = parallel
        else:
            pick_count = remain + parallel
        end = start + pick_count
        yield start, end
        start = end


def get_inner_model(model: Union[nn.Module, DDP, DataParallel]) -> nn.Module:
    return model.module if isinstance(model, (DDP, DataParallel)) else model


# https://github.com/pytorch/pytorch/issues/12160
def matrix_diag(diagonal: Tensor) -> Tensor:
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def masked_filter_and_pad(
    tensor: Tensor, mask: Tensor, fill_value: Union[int, float, Tensor] = 0
) -> tuple[Tensor, Tensor]:
    """
    Args:
        tensor: (B, M) Input tensor
        mask: (B, M) Boolean mask, True means to filter out
        fill_value: (B, 1) Value to fill invalid positions (default 0)

    Returns:
        filtered_padded: (B, N) Filtered and padded tensor (N is max valid count)
        valid_mask: (B, N) Boolean mask marking valid positions (True=invalid)
    """
    device = tensor.device
    B, M = tensor.shape

    # 1. Compute valid length per row and max length N
    lengths = (~mask).sum(dim=1)  # (B,)
    max_N = cast(int, lengths.max()) if B > 0 else 0

    # 2. Generate local column index for valid data per row (contiguous from 0)
    local_col_idx = torch.cumsum(~mask, dim=1) - 1  # (B, M)

    # 3. Extract all valid data (flattened)
    flat_data = tensor[~mask]  # (sum(lengths),)

    # 4. Generate row and column indices for target positions
    row_idx = torch.arange(B, device=device).repeat_interleave(
        lengths
    )  # (sum(lengths),)
    col_idx = local_col_idx[~mask]  # directly extract local column indices of valid positions

    # 5. Fill result
    if isinstance(fill_value, (int, float)):
        filtered_padded = torch.full((B, max_N), fill_value, device=device)
    else:
        filtered_padded = fill_value.to(device).repeat(1, max_N)
    filtered_padded[row_idx, col_idx] = flat_data

    # 6. Generate valid_mask
    valid_mask = torch.arange(max_N, device=device) >= lengths.unsqueeze(
        1
    )  # (B, max_N)

    return filtered_padded, valid_mask


def cumsum_with_clamp(
    x: Tensor, max_value: Union[float, Tensor], dim: int = -1
) -> Tensor:
    """
    Computes cumulative sum along `dim` with clamping at each step.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        max_value (float or torch.Tensor): Maximum allowed cumulative sum.
        dim (int, optional): Dimension to compute cumsum. Defaults to -1.

    Returns:
        torch.Tensor: Tensor with same shape as `x`, where cumsum is clamped at each step.

    Example:
        >>> x = torch.tensor([[1, -2, 3], [4, -5, 6]])
        >>> cumsum_with_clamp(x, max_value=3, dim=-1)
        tensor([[1, -1, 2], [3, -2, 1]])  # Clamped at each step
    """
    # Move target dim to the end for easier computation
    dim = dim if dim >= 0 else x.dim() + dim
    x = x.transpose(dim, -1)

    # Compute cumsum with clamping
    out = torch.zeros_like(x)
    current_sum = torch.zeros(*x.shape[:-1], device=x.device, dtype=x.dtype)

    for i in range(x.shape[-1]):
        current_sum += x[..., i]
        current_sum = torch.minimum(current_sum, torch.full_like(current_sum, max_value))  # type: ignore
        out[..., i] = current_sum

    # Restore original dim order
    out = out.transpose(dim, -1)
    return out


def pad_with_last_value(tensor: Tensor, dim: int, pad_size: int) -> Tensor:
    """
    Extends tensor using the last value along the specified dimension.

    Args:
        tensor: Input tensor
        dim: Dimension to extend
        pad_size: Length to extend

    Returns:
        Extended tensor
    """
    # Get the last element and repeat pad_size times along dim
    last = tensor.select(dim, -1).unsqueeze(dim)
    expanded_last = last.expand(
        *[-1 if i != dim else pad_size for i in range(tensor.dim())]
    )

    # Concatenate original tensor and expanded last value
    return torch.cat([tensor, expanded_last], dim=dim)


def distribute_evenly(N: int, K: int) -> list[int]:
    S, R = divmod(N, K)  # Compute S and R simultaneously, O(1) time
    distribution = [S + 1 if i < R else S for i in range(K)]  # Build list, O(K) time
    return distribution


def list_pick(
    list_length: int, start: int, pick_length: int
) -> tuple[int, int, int, int]:
    '''
    Returns: start1, end1, start2, end2
    '''
    end_index = start + pick_length
    if end_index <= list_length:
        return start, end_index, 0, 0
    else:
        exceed_len = end_index - list_length
        circle_end = min(start, exceed_len)
        return start, list_length, 0, circle_end


def merge_deques_random_popleft(deques: list[Deque[Any]]) -> list[Any]:
    result: list[Any] = []

    while True:
        # Get all non-empty deques
        non_empty_deques = [dq for dq in deques if dq]
        if not non_empty_deques:
            break  # All queues are empty, exit loop

        # Randomly select a non-empty deque
        chosen_deque = random.choice(non_empty_deques)

        # Pop left element and append to result
        result.append(chosen_deque.popleft())

    return result


def generate_random_booleans(n: int) -> list[bool]:
    assert n > 0
    while True:
        result = random.choices([True, False], k=n)
        if False in result and True in result:
            return result
