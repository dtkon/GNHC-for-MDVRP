# Neural Hybrid Construction for MDVRP

from typing import TYPE_CHECKING, Optional
import random, sys
import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ..problem.mdvrp import (
    local_reconstruct,
    cal_reward,
    solution_to_LEHD_direct_action,
    check_if_solutions_feasible,
)
from ..problem.utils import action_select, prob_select
from ..utils import augment, list_pick

if TYPE_CHECKING:
    from .agent import Agent


def train_one_batch(
    rank: int,
    agent: 'Agent',
    batch_feature: Tensor,
    other_feature: list[Tensor],
    log_step: int,
    logger: Optional[SummaryWriter] = None,
    enable_RL: bool = True,
    enable_LC: bool = False,
    enable_pomo: bool = False,
) -> None:
    batch_size, problem_size, _ = batch_feature.size()

    N_aug = agent.option.N_aug
    backward_len = agent.option.backward_len
    if backward_len < 0:
        backward_len = sys.maxsize
    step_len = agent.option.step_len

    feature = augment(
        batch_feature, N_aug
    )  # [N_aug*batch_size, problem_size, (x, y, ...)]
    other_feature = [x.repeat(N_aug, 1) for x in other_feature]

    if enable_pomo:
        pomo_size = problem_size - agent.option.depot_num
        feature_for_env = feature.repeat_interleave(pomo_size, 0)
        other_feature_for_env = [
            x.repeat_interleave(pomo_size, 0) for x in other_feature
        ]
    else:
        pomo_size = 1
        feature_for_env = feature
        other_feature_for_env = other_feature

    total_arange = torch.arange(feature_for_env.size(0), device=batch_feature.device)

    agent.env.set_up(
        feature_for_env,
        *other_feature_for_env,
        knn=agent.option.knn,
        depot_num=agent.option.depot_num
    )
    state, done = agent.env.step()

    # step first
    if enable_pomo:
        state, done = agent.env.step(agent.env.pomo_action())

    state_history: list[tuple[Tensor, ...]] = []
    action_history: list[Tensor] = []

    with torch.no_grad():
        enc_problems: list[Tensor] = agent.encoder(
            feature, depot_num=agent.option.depot_num
        )  # (N_aug*batch_size, problem_size, embed_dim), avg(N_aug*batch_size, embed_dim)

        if enable_pomo:
            enc_problems = [t.repeat_interleave(pomo_size, 0) for t in enc_problems]
            # (N_aug*batch_size*pomo_size, problem_size, embed_dim), avg(N_aug*batch_size*pomo_size, embed_dim)

        while not done:
            state_history.append(state)

            prob: Tensor = agent.decoder(*enc_problems, *state)
            action = action_select(prob)

            state, done = agent.env.step(action)

            action_history.append(action)

    sol = agent.env.solution()

    assert (
        check_if_solutions_feasible(sol, feature_for_env, other_feature_for_env[0]) == 0
    )

    reward = agent.env.reward()

    reward_logging = reward.clone()

    if enable_RL:
        reward = (
            reward.view(N_aug, batch_size, pomo_size)
            .permute(1, 0, 2)
            .reshape(batch_size, N_aug * pomo_size)
        )
        if N_aug * pomo_size > 1:
            advantage = reward - reward.mean(dim=1).view(batch_size, 1)
        else:
            advantage = reward

        agent.optimizer.zero_grad()

        if step_len > 0:
            rand_start = random.randint(0, len(state_history) - 1)
            s1, e1, s2, e2 = list_pick(len(state_history), rand_start, step_len)
            state_history = state_history[s1:e1] + state_history[s2:e2]
            action_history = action_history[s1:e1] + action_history[s2:e2]

        total_init_loss = 0.0

        i = 0
        while i < len(state_history):
            sub_state_history = state_history[i : i + backward_len]
            sub_action_history = action_history[i : i + backward_len]
            i += backward_len

            enc_problems = agent.encoder(feature, depot_num=agent.option.depot_num)

            if enable_pomo:
                enc_problems = [t.repeat_interleave(pomo_size, 0) for t in enc_problems]
                # (N_aug*batch_size*pomo_size, problem_size, embed_dim), avg(N_aug*batch_size*pomo_size, embed_dim)

            log_p_list = []

            for s, a in zip(sub_state_history, sub_action_history):
                prob = agent.decoder(*enc_problems, *s)
                log_p = torch.log(prob_select(prob, a, total_arange))
                log_p = (
                    log_p.view(N_aug, batch_size, pomo_size)
                    .permute(1, 0, 2)
                    .reshape(batch_size, N_aug * pomo_size)
                )
                log_p_list.append(log_p)

            log_p_sum = torch.stack(log_p_list, 0).sum(0)

            loss = -(advantage * log_p_sum).mean()
            total_init_loss += loss.item()
            loss.backward()

        # clip_grad_norms(agent.optimizer.param_groups, agent.option.max_grad_norm)
        agent.optimizer.step()

    if enable_LC:

        # local reconstruction RL
        reconstruction_bar = tqdm(
            total=agent.option.LC_iter,
            disable=agent.option.no_progress_bar or rank != 0,
            desc='reconstruction',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        )

        for _ in range(agent.option.LC_iter):
            agent.optimizer.zero_grad()

            seg_len = random.randint(agent.option.min_seg_len, agent.option.max_seg_len)

            # can not DDP, because returned list length may not same. However, can use step_len and pad to ensure same backward times, future work.
            sol, ori_state_histories, ori_action_histories, ori_reward_histories = (
                local_reconstruct(
                    feature_for_env,
                    other_feature_for_env[0],
                    sol,
                    seg_len,
                    agent.env,
                    agent.encoder,
                    agent.decoder,
                    agent.option.knn,
                )
            )

            assert (
                check_if_solutions_feasible(
                    sol, feature_for_env, other_feature_for_env[0]
                )
                == 0
            )

            if agent.option.max_LC_len > 0:
                state_num = 0
                state_histories, action_histories, reward_histories = [], [], []
                while (state_num < agent.option.max_LC_len) and len(
                    ori_state_histories
                ) > 0:
                    choose = random.randint(0, len(ori_state_histories) - 1)
                    state_num += len(ori_state_histories[choose])

                    state_histories.append(ori_state_histories[choose])
                    action_histories.append(ori_action_histories[choose])
                    reward_histories.append(ori_reward_histories[choose])

                    ori_state_histories.pop(choose)
                    ori_action_histories.pop(choose)
                    ori_reward_histories.pop(choose)
            else:
                state_histories, action_histories, reward_histories = (
                    ori_state_histories,
                    ori_action_histories,
                    ori_reward_histories,
                )

            flatten_state_history = []
            flatten_action_history = []
            flatten_reward_history: list[Tensor] = []
            for i in range(len(state_histories)):
                flatten_state_history.extend(state_histories[i])
                flatten_action_history.extend(action_histories[i])
                flatten_reward_history.extend(
                    [reward_histories[i]] * len(state_histories[i])
                )

            if step_len > 0:
                rand_start = random.randint(0, len(flatten_state_history) - 1)
                s1, e1, s2, e2 = list_pick(
                    len(flatten_state_history), rand_start, step_len
                )
                flatten_state_history = (
                    flatten_state_history[s1:e1] + flatten_state_history[s2:e2]
                )
                flatten_action_history = (
                    flatten_action_history[s1:e1] + flatten_action_history[s2:e2]
                )
                flatten_reward_history = (
                    flatten_reward_history[s1:e1] + flatten_reward_history[s2:e2]
                )

            total_final_loss = 0.0

            i = 0
            while i < len(flatten_state_history):
                sub_state_history = flatten_state_history[i : i + backward_len]
                sub_action_history = flatten_action_history[i : i + backward_len]
                sub_reward_history = flatten_reward_history[i : i + backward_len]
                i += backward_len

                enc_problems = agent.encoder(
                    feature_for_env, depot_num=agent.option.depot_num
                )
                r_times_log_p_list = []

                for s, a, r in zip(
                    sub_state_history, sub_action_history, sub_reward_history
                ):
                    prob = agent.decoder(*enc_problems, *s)
                    log_p = torch.log(prob_select(prob, a, total_arange))
                    r_times_log_p = r * log_p  # (N_aug*batch_size,)
                    r_times_log_p_list.append(r_times_log_p)

                r_times_log_p_sum = torch.stack(r_times_log_p_list, 0).sum(
                    0
                )  # (N_aug*batch_size,)

                loss = -r_times_log_p_sum.mean()
                total_final_loss += loss.item()
                loss.backward()

            agent.optimizer.step()
            reconstruction_bar.update()

        reconstruction_bar.close()

        # IL cost too high, and has bug, turn to todo
        if False and 'enable_IL' and agent.option.IL_num > 0:
            direct_action_lists = solution_to_LEHD_direct_action(
                sol, agent.option.IL_num
            )

            for direct_action_list in direct_action_lists:
                agent.optimizer.zero_grad()

                agent.env.set_up(
                    feature,
                    *other_feature,
                    knn=agent.option.knn,
                    depot_num=agent.option.depot_num
                )
                state, done = agent.env.step(
                    knn_must_contain=direct_action_list[0][:, 1:2]
                )

                for i, direct_action in enumerate(direct_action_list):
                    assert not done

                    available_index: Tensor = agent.env.available_index  # type: ignore
                    action_node = torch.nonzero(
                        direct_action[:, 1:2] == available_index, as_tuple=True
                    )[1]
                    action = direct_action.clone()
                    action[:, 1] = action_node

                    enc_problems = agent.encoder(
                        feature, depot_num=agent.option.depot_num
                    )
                    prob = agent.decoder(*enc_problems, *state)

                    if i == len(direct_action_list) - 1:
                        state, done = agent.env.step(action)
                    else:
                        state, done = agent.env.step(
                            action,
                            knn_must_contain=direct_action_list[i + 1][:, 1:2],
                        )

                    log_p = torch.log(prob_select(prob, action, total_arange))
                    loss = (-agent.option.IL_rate * log_p).mean()
                    loss.backward()

                agent.optimizer.step()

    if logger is not None and rank == 0 and log_step % agent.option.log_step == 0:
        if enable_RL:
            logger.add_scalar('training/init_loss', total_init_loss, log_step)
        logger.add_scalar(
            'training/init_reward', reward_logging.mean().item(), log_step
        )

        if enable_LC:
            final_reward = cal_reward(feature, sol)
            logger.add_scalar('training/final_loss', total_final_loss, log_step)
            logger.add_scalar(
                'training/final_reward', final_reward.mean().item(), log_step
            )
