from typing import TYPE_CHECKING, Optional
import os, json, copy
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import itertools

if TYPE_CHECKING:
    from ..RL.agent import Agent


def log_eval(
    logger: Optional[SummaryWriter],
    agent: 'Agent',
    rewards: Tensor,
    action: Optional[Tensor] = None,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
    time: Optional[float] = None,
) -> None:
    _log_eval_SO(logger, rewards, epoch, save_dir, time)


def _log_eval_SO(
    logger: Optional[SummaryWriter],
    rewards: Tensor,
    epoch: Optional[int] = None,
    save_dir: Optional[str] = None,
    time: Optional[float] = None,
) -> None:
    '''
    rewards: (val_size, reward_dim)

    infeasible: (val_size, N_aug, sample_times)
    '''
    if logger is not None:
        logger.add_scalar('evaluating/avg_obj', (-rewards).mean().cpu().item(), epoch)

    if save_dir is not None:
        with open(os.path.join(save_dir, 'eval.txt'), 'a') as f:
            f.write(f'{(-rewards).mean().cpu().item()}\n')
            if time is not None:
                f.write(f'time: {str(time)}s\n')

        with open(os.path.join(save_dir, 'objv.json'), 'w') as f:
            json.dump((-rewards).cpu().tolist(), f)


def route_remove_duplicates(action: Tensor) -> list[list[list[list[int]]]]:
    '''
    action: (batch_size, need_num, vehicle_num, action_length)
    '''
    result = [
        [
            [[node for node, _ in itertools.groupby(a_vel.tolist())] for a_vel in a_sol]
            for a_sol in a_ins
        ]
        for a_ins in action
    ]
    return result


def create_action_array_from_list(
    action_list: list[list[list[list[int]]]],
) -> np.ndarray:
    action_list = copy.deepcopy(action_list)
    max_length = 0
    for a_ins in action_list:
        for a_sol in a_ins:
            for a_vel in a_sol:
                if len(a_vel) > max_length:
                    max_length = len(a_vel)
    for a_ins in action_list:
        for a_sol in a_ins:
            for a_vel in a_sol:
                a_vel.extend([0] * (max_length - len(a_vel)))
    return np.array(action_list)
