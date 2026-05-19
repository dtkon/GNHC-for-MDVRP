import random
import time
import platform
from typing import TYPE_CHECKING, Callable, Optional, cast
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from ..problem import Env
from ..problem.utils import action_select
from ..problem.cvrp import (
    generate_dataset as generate_dataset_cvrp,
    load_dataset as load_dataset_cvrp,
    check_if_solutions_feasible as check_if_solutions_feasible_cvrp,
)
from ..problem.mdvrp import (
    generate_dataset as generate_dataset_mdvrp,
    load_dataset as load_dataset_mdvrp,
    local_reconstruct,
    cal_reward,
    check_if_solutions_feasible as check_if_solutions_feasible_mdvrp,
    solution_remove_dup,
)
from ..problem.tsp import (
    generate_dataset as generate_dataset_tsp,
    load_dataset as load_dataset_tsp,
    check_if_solutions_feasible as check_if_solutions_feasible_tsp,
)
from ..problem.mdvrp_d import (
    load_dataset as load_dataset_dmdvrp,
    DMDVRP_instance,
    events_to_device,
)
from ..utils import augment, batch_slicer

if TYPE_CHECKING:
    from .agent import Agent
    from ..NN.layers import Policy


def eval(
    rank: int, agent: 'Agent', mp_ret: Optional[mp.Queue] = None, init_dist: bool = True
) -> tuple[Tensor, Tensor, float]:
    '''
    return: rewards, action, time_used
    '''

    if rank == 0:
        print('\nEvaluating...', flush=True)

    option = agent.option

    # agent.eval()

    random_state_backup = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        random.getstate(),
    )

    torch.manual_seed(option.seed)
    random.seed(option.seed)

    if option.problem != 'dmdvrp':
        generate_dataset: Callable[..., list]
        load_dataset: Callable[..., list]
        if option.problem == 'cvrp':
            generate_dataset = generate_dataset_cvrp
            load_dataset = load_dataset_cvrp
        elif option.problem == 'mdvrp':
            generate_dataset = generate_dataset_mdvrp
            load_dataset = load_dataset_mdvrp
        elif option.problem == 'tsp':
            generate_dataset = generate_dataset_tsp
            load_dataset = load_dataset_tsp
        else:
            raise NotImplementedError

        if option.val_dataset is None:
            val_dataset = generate_dataset(
                option.val_range[1] - option.val_range[0],
                option.val_customer_num,
                depot_num=option.val_depot_num,
                vehicle_capacity=option.val_vehicle_capacity,
            )
        else:
            val_dataset = load_dataset(option.val_dataset)[
                option.val_range[0] : option.val_range[1]
            ]

        if option.distributed and init_dist:
            device = torch.device('cuda', rank)
            dist.init_process_group(
                backend='gloo' if platform.system() == 'Windows' else 'nccl',
                world_size=option.world_size,
                rank=rank,
            )
            torch.cuda.set_device(rank)
            agent.decoder.to(device)
            agent.encoder.to(device)

            if option.normalization == 'batch':
                agent.encoder = cast(
                    'Policy',
                    torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent.encoder).to(
                        device
                    ),
                )

            agent.decoder = cast(
                'Policy',
                torch.nn.parallel.DistributedDataParallel(
                    agent.decoder, device_ids=[rank], find_unused_parameters=False
                ),
            )
            agent.encoder = cast(
                'Policy',
                torch.nn.parallel.DistributedDataParallel(
                    agent.encoder, device_ids=[rank], find_unused_parameters=False
                ),
            )
        else:
            if option.use_cuda:
                device = torch.device('cuda', rank)
            else:
                device = option.device

        if option.distributed:
            val_sampler: torch.utils.data.distributed.DistributedSampler = (
                torch.utils.data.distributed.DistributedSampler(
                    cast(Dataset, val_dataset), shuffle=False
                )
            )
            val_dataloader: DataLoader = DataLoader(
                cast(Dataset, val_dataset),
                batch_size=option.val_batch_size // option.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler,
            )
        else:
            val_dataloader = DataLoader(
                cast(Dataset, val_dataset),
                batch_size=option.val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        start_time = time.time()

        rewards, solution = evaluate(rank, agent, val_dataloader, device)

        time_used = time.time() - start_time

        if option.distributed:
            dist.barrier()

            rewards = gather_tensor_and_concat(rewards.contiguous())

            dist.barrier()

            if mp_ret is not None and rank == 0:
                mp_ret.put((rewards.cpu(), solution.cpu(), time_used))

    else:
        start_time = time.time()

        rewards, solution = evaluate_d(agent)

        time_used = time.time() - start_time

    torch.set_rng_state(random_state_backup[0])
    torch.cuda.set_rng_state(random_state_backup[1])
    random.setstate(random_state_backup[2])

    return rewards, solution, time_used


def evaluate(
    rank: int, agent: 'Agent', val_dataloader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor]:
    '''
    return: rewards, action
    '''

    # calculate tqdm total
    val_size = (
        agent.option.val_range[1] - agent.option.val_range[0]
        if not agent.option.distributed
        else (agent.option.val_range[1] - agent.option.val_range[0])
        // agent.option.world_size
    )
    val_batch_size = (
        agent.option.val_batch_size
        if not agent.option.distributed
        else agent.option.val_batch_size // agent.option.world_size
    )
    batch_num = val_size // val_batch_size
    if agent.option.eval_type == 'greedy':
        total = batch_num
    elif agent.option.eval_type == 'sample':
        split_batch_num = (
            val_batch_size
            * agent.option.val_N_aug
            * agent.option.sample_times
            // agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num
    elif agent.option.eval_type == 'greedy_aug':
        split_batch_num = (
            val_batch_size * agent.option.val_N_aug // agent.option.max_parallel
        )
        if split_batch_num == 0:
            split_batch_num = 1
        total = batch_num * split_batch_num
    else:
        raise NotImplementedError

    eval_bar = tqdm(
        total=total,
        disable=agent.option.no_progress_bar or rank != 0,
        desc='evaluating',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

    reward_list = []

    with torch.no_grad():
        batch_dataset: list[Tensor]
        for batch_dataset in val_dataloader:
            feature, *other_feature = batch_dataset
            feature = feature.to(device)
            other_feature = [x.to(device) for x in other_feature]

            if agent.option.eval_type == 'greedy':
                N_aug = 1
                sample_times = 0
            elif agent.option.eval_type == 'sample':
                N_aug = agent.option.val_N_aug
                sample_times = agent.option.sample_times
            elif agent.option.eval_type == 'greedy_aug':
                N_aug = agent.option.val_N_aug
                sample_times = 0
            else:
                raise NotImplementedError

            if agent.option.enable_LC:
                val_LC_iter = agent.option.val_LC_iter
            else:
                val_LC_iter = 0

            if agent.option.problem == 'mdvrp':
                check_if_solutions_feasible: Callable[..., int] = (
                    check_if_solutions_feasible_mdvrp
                )
            elif agent.option.problem == 'cvrp':
                check_if_solutions_feasible = check_if_solutions_feasible_cvrp
            elif agent.option.problem == 'tsp':
                check_if_solutions_feasible = check_if_solutions_feasible_tsp
            else:
                raise NotImplementedError

            reward, solution = solve(
                feature,
                other_feature,
                agent.decoder,
                agent.encoder,
                agent.option.val_depot_num,
                agent.env,
                N_aug,
                sample_times,
                val_LC_iter,
                agent.option.min_seg_len,
                agent.option.max_seg_len,
                eval_bar,
                agent.option.knn,
                agent.option.max_parallel,
                check_if_solutions_feasible,
                enable_pomo=agent.option.enable_pomo,
            )

            reward_list.append(reward)

            # if action.size(-1) > max_action_length:
            #    max_action_length = action.size(-1)

    eval_bar.close()

    # for i, t in enumerate(action_list):
    #    action_list[i] = F.pad(t, (0, max_action_length - t.size(-1)), 'constant', -1)

    return (torch.cat(reward_list), torch.empty(1))  # (val_size,)


def evaluate_d(agent: 'Agent') -> tuple[Tensor, Tensor]:
    option = agent.option

    assert option.val_dataset is not None
    init_p, init_d, events = load_dataset_dmdvrp(option.val_dataset)
    last_step = events[-1]['step']

    init_p = init_p.to(option.device)
    init_d = init_d.to(option.device)
    events_to_device(events, option.device)

    d_ins = DMDVRP_instance(init_p, init_d, events)

    env = agent.env

    cus_nums = []

    while not d_ins.is_done():
        remain_problems = d_ins.remaining_instance()

        cus_num = remain_problems[0].size(0) - remain_problems[2]
        cus_nums.append(cus_num)

        remain_problems = [
            x.unsqueeze(0) if isinstance(x, torch.Tensor) else x
            for x in remain_problems
        ]  # type: ignore

        if option.val_N_aug > 1:
            N_aug = option.val_N_aug
            remain_problems[0] = augment(remain_problems[0], N_aug)  # type: ignore
            remain_problems[1] = remain_problems[1].repeat(N_aug, 1)  # type: ignore
            remain_problems[3] = remain_problems[3].repeat(N_aug, 1)  # type: ignore
            remain_problems[4] = remain_problems[4].repeat(N_aug, 1)  # type: ignore

        env.set_up(*remain_problems)  # type: ignore
        state, done = env.step()

        with torch.no_grad():
            enc_problems: list[Tensor] = agent.encoder(
                remain_problems[0], depot_num=remain_problems[2]
            )

            while not done:
                prob: Tensor = agent.decoder(*enc_problems, *state)
                action = action_select(prob, 'greedy')
                state, done = env.step(action)

        reward = env.reward()  # (N_aug,)
        solution = env.solution()  # (N_aug, length)

        if option.val_N_aug > 1:
            best_index = reward.max(0)[1]
            best_sol = solution[best_index : best_index + 1]
        else:
            best_sol = env.solution()

        assert (
            check_if_solutions_feasible_mdvrp(
                best_sol, remain_problems[0][0:1], remain_problems[1][0:1]
            )
            == 0
        )

        next_event_step = d_ins.submit_solution(
            solution_remove_dup(best_sol).squeeze(0)
        )
        print(f'{next_event_step}/{last_step}, cus: {cus_num}')

    print(f'mean cus: {sum(cus_nums)/len(cus_nums)}, len: {len(cus_nums)}')

    return -d_ins.objective(), d_ins.solution()


def solve(
    feature: Tensor,
    other_feature: list[Tensor],
    decoder: nn.Module,
    encoder: nn.Module,
    depot_num: int,
    env: Env,
    N_aug: int,
    sample_times: int,
    LC_iter: int,
    min_seg_len: int,
    max_seg_len: int,
    progress_bar: Optional[tqdm],
    knn: int = -1,
    max_parallel: Optional[int] = None,
    check_if_solutions_feasible: Optional[Callable[..., int]] = None,
    enable_pomo: bool = False,
) -> tuple[Tensor, Tensor]:
    '''
    sample_times: 0 for greedy

    return: rewards, solution
    '''
    batch_size, problem_size, _ = feature.size()

    feature = augment(feature, N_aug)  # [N_aug*batch_size, problem_size, 3]
    other_feature = [x.repeat(N_aug, 1) for x in other_feature]

    if sample_times > 1:
        feature = feature.repeat(
            sample_times, 1, 1
        )  # [sample_times*N_aug*batch_size, problem_size, 3]
        other_feature = [x.repeat(sample_times, 1) for x in other_feature]

    if sample_times > 0:
        sample_mode = 'sample'
    else:
        sample_mode = 'greedy'

    if max_parallel is None:
        max_parallel = feature.size(0)

    split_reward_list = []
    for split_start, split_end in batch_slicer(feature.size(0), max_parallel):
        split_problem = feature[split_start:split_end]
        split_other_feature = [x[split_start:split_end] for x in other_feature]

        if enable_pomo:
            pomo_size = problem_size - depot_num
            feature_for_env = split_problem.repeat_interleave(pomo_size, 0)
            other_feature_for_env = [
                x.repeat_interleave(pomo_size, 0) for x in split_other_feature
            ]
        else:
            pomo_size = 1
            feature_for_env = split_problem
            other_feature_for_env = split_other_feature

        env.set_up(
            feature_for_env, *other_feature_for_env, depot_num=depot_num, knn=knn
        )
        state, done = env.step()

        if enable_pomo:
            state, done = env.step(env.pomo_action())

        with torch.no_grad():
            enc_problems: list[Tensor] = encoder(split_problem, depot_num=depot_num)
            if enable_pomo:
                enc_problems = [t.repeat_interleave(pomo_size, 0) for t in enc_problems]
                # (sample_times*N_aug*batch_size*pomo_size, problem_size, embed_dim), avg(sample_times*N_aug*batch_size*pomo_size, embed_dim)

            while not done:
                prob: Tensor = decoder(*enc_problems, *state)
                action = action_select(prob, sample_mode)

                state, done = env.step(action)

        # local reconstruction
        if LC_iter > 0:
            reconstruction_bar = tqdm(
                total=LC_iter,
                disable=(progress_bar is None),
                desc='reconstruction',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            )

            sol = env.solution()

            if check_if_solutions_feasible is not None:
                assert (
                    check_if_solutions_feasible(
                        sol, feature_for_env, other_feature_for_env[0]
                    )
                    == 0
                )

            for _ in range(LC_iter):
                seg_len = random.randint(min_seg_len, max_seg_len)
                sol, *_ = local_reconstruct(
                    feature_for_env,
                    other_feature_for_env[0],
                    sol,
                    seg_len,
                    env,
                    encoder,
                    decoder,
                    knn,
                    sample_mode,
                )

                if check_if_solutions_feasible is not None:
                    assert (
                        check_if_solutions_feasible(
                            sol, feature_for_env, other_feature_for_env[0]
                        )
                        == 0
                    )

                reconstruction_bar.update()

            reconstruction_bar.close()
            reward = cal_reward(feature_for_env, sol)
        else:
            reward = env.reward()  # (max_paral,)

        if progress_bar is not None:
            progress_bar.update()

        split_reward_list.append(reward)

    all_reward = torch.cat(
        split_reward_list
    )  # (sample_times*N_aug*batch_size*pomo_size,)

    all_reward_best_index = (
        all_reward.view(-1, batch_size, pomo_size)
        .permute(0, 2, 1)
        .reshape(-1, batch_size)
        .max(dim=0)[1]
    )  # (batch_size,)

    best_reward = (
        all_reward.view(-1, batch_size, pomo_size)
        .permute(0, 2, 1)
        .reshape(-1, batch_size)[all_reward_best_index, torch.arange(batch_size)]
    )

    return (best_reward, torch.empty(1))  # (batch_size,)


def gather_tensor_and_concat(tensor: Tensor) -> Tensor:
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)
