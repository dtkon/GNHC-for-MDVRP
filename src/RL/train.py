import random
import platform
from typing import TYPE_CHECKING, Callable, cast
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm

from ..problem.cvrp import (
    generate_dataset as generate_dataset_cvrp,
    load_dataset as load_dataset_cvrp,
)
from ..problem.mdvrp import (
    generate_dataset as generate_dataset_mdvrp,
    load_dataset as load_dataset_mdvrp,
)
from ..problem.tsp import (
    generate_dataset as generate_dataset_tsp,
    load_dataset as load_dataset_tsp,
)
from ..utils import log_eval

from .REINFORCE import train_one_batch as train_one_batch_RL
from .REINFORCE_NHC import train_one_batch as train_one_batch_NHC
from .eval import eval


if TYPE_CHECKING:
    from .agent import Agent
    from ..NN.layers import Policy


def train(rank: int, agent: 'Agent') -> None:
    option = agent.option
    logger = None if option.no_log else SummaryWriter(option.log_dir)

    if option.distributed:
        device = torch.device('cuda', rank)
        dist.init_process_group(
            backend='gloo' if platform.system() == 'Windows' else 'nccl',
            world_size=option.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        agent.decoder.to(device)  # this will increase memory cost of original place.
        agent.encoder.to(device)

        if option.normalization == 'batch':
            agent.encoder = cast(
                'Policy',
                torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent.encoder).to(device),
            )

        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

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

        dist.barrier()
    else:
        device = option.device

        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    agent.train()

    # set or restore seed
    if option.load_path is None:
        torch.manual_seed(option.seed)
        random.seed(option.seed)
    else:
        agent.load()

    steps = option.epoch_size // option.batch_size

    training_bar = tqdm(
        total=(option.epoch_end - option.epoch_start) * steps,
        disable=option.no_progress_bar or rank != 0,
        desc='training',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    )

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

    if option.first_eval_once and not option.no_eval_in_train:
        # if rank == 0: all sub-process need to call eval, otherwise program will get stuck because of SyncBatchNorm
        agent.eval()
        eval_rewards, action, time_used = eval(rank, agent, init_dist=False)
        agent.train()

        if rank == 0:
            log_eval(logger, agent, eval_rewards, action, option.epoch_start - 1)

    if option.fine_tune:
        assert option.val_dataset is not None
        val_dataset = (
            load_dataset(option.val_dataset)[option.val_range[0] : option.val_range[1]]
            * option.epoch_size
        )
    for epoch in range(option.epoch_start, option.epoch_end):
        if not option.fine_tune:
            epoch_dataset = generate_dataset(
                option.epoch_size,
                option.customer_num,
                vehicle_capacity=option.vehicle_capacity,
                depot_num=option.depot_num,
            )
        else:
            epoch_dataset = val_dataset
        if option.distributed:
            train_sampler: torch.utils.data.distributed.DistributedSampler = (
                torch.utils.data.distributed.DistributedSampler(
                    cast(Dataset, epoch_dataset), shuffle=False
                )
            )
            train_dataloader = DataLoader(
                cast(Dataset, epoch_dataset),
                batch_size=option.batch_size // option.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler,
            )
        else:
            train_dataloader = DataLoader(
                cast(Dataset, epoch_dataset),
                batch_size=option.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        batch_dataset: list[Tensor]
        for step, batch_dataset in enumerate(train_dataloader):
            feature, *other_feature = batch_dataset
            feature = feature.to(device)
            other_feature = [x.to(device) for x in other_feature]

            if option.training_type == 'RL':
                train_one_batch_RL(
                    rank, agent, feature, other_feature, epoch * steps + step, logger
                )
            elif option.training_type == 'NHC':
                train_one_batch_NHC(
                    rank,
                    agent,
                    feature,
                    other_feature,
                    epoch * steps + step,
                    logger,
                    enable_RL=not option.disable_RL,
                    enable_LC=option.enable_LC,
                    enable_pomo=option.enable_pomo,
                )

            training_bar.update()

        agent.lr_scheduler.step()

        # save new model after one epoch
        if (
            rank == 0
            and not option.no_save
            and (
                (option.save_per_epochs != 0 and epoch % option.save_per_epochs == 0)
                or epoch == option.epoch_end - 1
            )
        ):
            agent.save(epoch)

        if not option.no_eval_in_train:

            torch.cuda.empty_cache()

            # if rank == 0: all sub-process need to call eval, otherwise program will get stuck because of SyncBatchNorm
            agent.eval()
            eval_rewards, action, time_used = eval(rank, agent, init_dist=False)
            agent.train()

            # if option.problem == 'cvrp':
            #    bl_vals = eval(rank, agent, baseline)
            #    baseline.update(eval_rewards, bl_vals, rank == 0)

            if rank == 0:
                log_eval(logger, agent, eval_rewards, action, epoch)

        torch.cuda.empty_cache()

        if option.distributed:
            dist.barrier()

    training_bar.close()
