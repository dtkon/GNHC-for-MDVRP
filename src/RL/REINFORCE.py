from typing import TYPE_CHECKING, Optional
import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from ..problem.utils import action_select, prob_select

if TYPE_CHECKING:
    from .agent import Agent


def train_one_batch(
    rank: int,
    agent: 'Agent',
    feature: Tensor,
    other_feature: list[Tensor],
    log_step: int,
    logger: Optional[SummaryWriter] = None,
) -> None:
    batch_arange = torch.arange(feature.size(0), device=feature.device)

    agent.env.set_up(feature, *other_feature, depot_num=agent.option.depot_num)
    state, done = agent.env.step()

    log_p_list = []

    enc_problems = agent.encoder(feature, depot_num=agent.option.depot_num)
    while not done:
        prob: Tensor = agent.decoder(*enc_problems, *state)
        action = action_select(prob)

        log_p_list.append(torch.log(prob_select(prob, action, batch_arange)))

        state, done = agent.env.step(action)

    reward = agent.env.reward()
    log_p_sum = torch.stack(log_p_list, 0).sum(0)

    loss = -(reward.view(-1) * log_p_sum.view(-1)).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    # clip_grad_norms(agent.optimizer.param_groups, agent.option.max_grad_norm)
    agent.optimizer.step()

    if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
        logger.add_scalar('training/loss', loss.item(), log_step)
        logger.add_scalar('training/reward', reward.mean().item(), log_step)
