from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F


def zoom(problems: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    '''
    problem: [batch_size, problem_size, (x, y, ...)]

    mask: (batch_size, problem_size), mask[:, 0] must all be False.

    return: [problem_zoom, gap(batch_size,)]
    '''
    if mask is not None:
        depot = problems[:, 0:1, :].repeat(1, problems.size(1), 1)
        problems = torch.where(
            mask.unsqueeze(2).repeat(1, 1, problems.size(2)), depot, problems
        )

    problems_xy = problems[:, :, :2]
    max_coord = problems_xy.max(1)[0]  # (batch_size, 2)
    min_coord = problems_xy.min(1)[0]  # (batch_size, 2)
    x_gap = max_coord[:, 0] - min_coord[:, 0]  # (batch_size,)
    y_gap = max_coord[:, 1] - min_coord[:, 1]  # (batch_size,)
    xy_gap = torch.cat([x_gap[None, :], y_gap[None, :]])  # (2, batch_size)
    gap = xy_gap.max(0)[0]  # (batch_size,)
    problem_zoom = (problems_xy - min_coord[:, None, :]) / gap[:, None, None]

    if problems.size(2) > 2:
        problem_zoom = torch.cat((problem_zoom, problems[:, :, 2:]), dim=2)

    return problem_zoom, gap


def cal_distance(problems: Tensor, routes: Tensor, cycle: bool = False) -> Tensor:
    '''
    problems: [batch_size, problem_size, (x, y, ...)]

    routes: (batch_size, node_indexes)

    return: (batch_size,)
    '''
    routes = routes.unsqueeze(-1).expand(-1, -1, 2)
    seq_nodes = torch.gather(
        problems[:, :, :2], 1, routes
    )  # (batch_size, node_indexes, 2)

    if not cycle:
        next_nodes = seq_nodes[:, 1:, :]
        return torch.linalg.vector_norm(next_nodes - seq_nodes[:, :-1, :], dim=-1).sum(
            -1
        )
    else:
        next_nodes = torch.cat((seq_nodes[:, 1:, :], seq_nodes[:, 0:1, :]), 1)
        return torch.linalg.vector_norm(next_nodes - seq_nodes, dim=-1).sum(-1)


def action_select(prob_or_mask: Tensor, mode: str = 'sample') -> Tensor:
    '''
    return: (batch_size, 1) or (batch_size, 2)
    '''
    if prob_or_mask.dtype == torch.bool:
        mask = prob_or_mask
        compatibility = torch.ones(mask.size(), device=mask.device, dtype=torch.float)
        compatibility[mask] = -float('inf')
        prob = F.softmax(compatibility.view(mask.size(0), -1), dim=-1).view_as(mask)
    else:
        prob = prob_or_mask

    if prob.dim() == 2:
        if mode == 'sample':
            action = prob.multinomial(1)  # (batch_size, 1)
        elif mode == 'greedy':
            action = prob.max(dim=1, keepdim=True)[1]  # (batch_size, 1)
        else:
            raise NotImplementedError
    elif prob.dim() == 3:
        batch_size, x_num, y_num = prob.size()
        if mode == 'sample':
            select_result = prob.view(batch_size, -1).multinomial(1)  # (batch_size, 1)
        elif mode == 'greedy':
            select_result = prob.view(batch_size, -1).max(dim=1, keepdim=True)[
                1
            ]  # (batch_size, 1)
        else:
            raise NotImplementedError
        x_sel = torch.div(
            select_result, y_num, rounding_mode='trunc'
        )  # (batch_size, 1)
        y_sel = select_result % y_num  # (batch_size, 1)
        action = torch.cat((x_sel, y_sel), dim=1)  # (batch_size, 2)
    elif prob.dim() == 4:
        batch_size, x_num, y_num, z_num = prob.size()
        if mode == 'sample':
            select_result = prob.view(batch_size, -1).multinomial(1)  # (batch_size, 1)
        elif mode == 'greedy':
            select_result = prob.view(batch_size, -1).max(dim=1, keepdim=True)[
                1
            ]  # (batch_size, 1)
        else:
            raise NotImplementedError
        x_sel = torch.div(
            select_result, y_num * z_num, rounding_mode='trunc'
        )  # (batch_size, 1)
        rest = select_result % (y_num * z_num)
        y_sel = torch.div(rest, z_num, rounding_mode='trunc')  # (batch_size, 1)
        z_sel = rest % z_num  # (batch_size, 1)
        action = torch.cat((x_sel, y_sel, z_sel), dim=1)  # (batch_size, 3)
    else:
        raise NotImplementedError

    return action


def prob_select(prob: Tensor, action: Tensor, arange: Tensor) -> Tensor:
    if prob.dim() == 2:
        return prob[arange, action.view(-1)]
    elif prob.dim() == 3:
        return prob[arange, action[:, 0], action[:, 1]]
    elif prob.dim() == 4:
        return prob[arange, action[:, 0], action[:, 1], action[:, 2]]
    else:
        raise NotImplementedError


def knn_indices(
    problem: Tensor,
    center_index: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
    pad_value: int = -1,
) -> tuple[Tensor, Tensor]:
    """
    Vectorized KNN algorithm implementation.

    Args:
        problem: Coordinate matrix, shape (batch_size, node_num, 3)
        center_index: Center point index, shape (batch_size, 1)
        k: Number of neighbors to return
        mask: Mask matrix, shape (batch_size, node_num), True means masked point
        pad_value: Padding value for insufficient neighbors

    Returns:
        Neighbor index matrix, shape (batch_size, k), padded with -1
        Neighbor distance results
    """
    batch_size, node_num, _ = problem.shape
    device = problem.device

    # Extract valid coordinates (x, y)
    points = problem[..., :2]  # (batch_size, node_num, 2)

    # Get center point coordinates
    center_points = points[
        torch.arange(batch_size), center_index.squeeze(-1)
    ]  # (batch_size, 2)

    # Compute squared distances (avoid sqrt for performance)
    diff = points - center_points.unsqueeze(1)  # broadcast diff
    dists = (diff**2).sum(dim=-1)  # (batch_size, node_num)

    # Handle masked points: set their distance to infinity
    if mask is not None:
        dists = torch.where(mask, torch.tensor(float('inf'), device=device), dists)

    # Find k nearest neighbor indices (always select k closest points)
    if k > problem.size(1):
        k = problem.size(1)

    sorted_dists, sorted_indices = torch.topk(
        dists, k, dim=1, largest=False, sorted=True
    )

    # Handle cases with insufficient valid points
    # Create valid point mask (finite distance means valid)
    valid_mask = ~torch.isinf(dists)  # (batch_size, node_num)
    valid_count = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

    # Create k index range
    k_range = torch.arange(k, device=device).expand(batch_size, -1)

    # Mark invalid positions (beyond actual valid count)
    invalid_mask = k_range >= valid_count

    # Set invalid positions to -1
    result = torch.where(invalid_mask, pad_value, sorted_indices)

    return result, sorted_dists
