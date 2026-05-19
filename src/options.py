import os
import sys
import time
import argparse
import re
from typing import Optional, Sequence
import torch

DEBUG_DDP = False


class Option(argparse.Namespace):
    # Problem
    problem: str
    customer_num: int
    depot_num: int
    vehicle_capacity: int

    # Model
    model: str
    embedding_dim: int
    feed_forward_dim: int
    n_heads_enc: int
    n_heads_dec: int
    n_blocks_enc: int
    n_blocks_dec: int
    normalization: str

    # Training
    training_type: str
    batch_size: int
    epoch_size: int
    epoch_start: int
    epoch_end: int
    N_aug: int
    knn: int
    step_len: int
    backward_len: int
    LC_iter: int
    min_seg_len: int
    max_seg_len: int
    IL_num: int
    IL_rate: float
    lr_model: float
    lr_decay: float
    weight_decay: float
    # max_grad_norm: float
    no_eval_in_train: bool
    fine_tune: bool
    enable_LC: bool
    max_LC_len: int
    enable_IL: bool
    enable_pomo: bool
    disable_RL: bool

    # Evaluate
    eval_only: bool
    eval_type: str
    val_range: list[int]
    val_batch_size: int
    val_dataset: Optional[str]
    val_customer_num: int
    val_depot_num: int
    val_vehicle_capacity: int
    val_N_aug: int
    val_LC_iter: int
    sample_times: int
    max_parallel: int

    # Misc
    seed: int
    no_cuda: bool
    enable_DDP: bool
    no_save: bool
    no_log: bool
    no_write: bool
    run_name: str
    model_save_dir: str
    log_dir: str
    save_per_epochs: int
    log_step: int
    load_path: Optional[str]
    resume: Optional[str]
    first_eval_once: bool
    no_progress_bar: bool
    DDP_port_offset: int
    zoom: str

    # Add later
    use_cuda: bool
    distributed: bool
    world_size: int
    save_dir: str
    device: torch.device
    zoom_on: bool


no_provide = object()


def get_options(args: Optional[Sequence[str]] = None) -> Option:
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Routing Problem with Reinforcement Learning"
    )

    # Problem
    parser.add_argument(
        '--problem',
        default='mdvrp',
        choices=('tsp', 'cvrp', 'mdvrp', 'dmdvrp'),
    )
    parser.add_argument('--customer_num', type=int, default=20)
    parser.add_argument('--depot_num', type=int, default=2)
    parser.add_argument('--vehicle_capacity', default=-1, type=int)

    # Model
    parser.add_argument(
        '--model',
        default='LEHDca_att_da',
        choices=(
            'AM',
            'AMre',
            'LEHD',
            'LEHDca',
            'LEHDca_att',
            'LEHDca_att_da',
            'greedy',
        ),
    )
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--feed_forward_dim', type=int, default=512)
    parser.add_argument('--n_heads_enc', type=int, default=8)
    parser.add_argument('--n_heads_dec', type=int, default=8)
    parser.add_argument('--n_blocks_enc', type=int, default=3)
    parser.add_argument('--n_blocks_dec', type=int, default=6)
    parser.add_argument(
        '--normalization',
        default='layer',
        choices=('layer', 'batch', 'instance', 'none'),
    )

    # Training
    parser.add_argument('--training_type', default='NHC', choices=('RL', 'NHC'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_size', type=int, default=16000)
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--epoch_end', type=int, default=100)
    parser.add_argument('--N_aug', type=int, default=8)
    parser.add_argument(
        '--knn',
        type=int,
        default=-1,
        help='k nearest nodes for each depot, -1 to unlimited',
    )
    parser.add_argument(
        '--step_len',
        type=int,
        default=-1,
        help='max RL trajectory length for one step, -1 to unlimited',
    )
    parser.add_argument(
        '--backward_len',
        type=int,
        default=10,
        help='max RL trajectory length for one backward, -1 to unlimited',
    )
    parser.add_argument(
        '--LC_iter',
        type=int,
        default=10,
        help='Reconstruction iteration during training',
    )
    parser.add_argument(
        '--min_seg_len',
        type=int,
        default=4,
        help='max segment length for reconstruction iteration during training',
    )
    parser.add_argument(
        '--max_seg_len',
        type=int,
        default=10,
        help='max segment length for reconstruction iteration during training',
    )
    parser.add_argument('--IL_num', type=int, default=1)
    parser.add_argument('--IL_rate', type=float, default=0.01)
    parser.add_argument('--lr_model', type=float, default=1e-4)
    parser.add_argument(
        '--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch'
    )
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument(
    #    '--max_grad_norm',
    #    type=float,
    #    default=1.0,
    #    help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)',
    # )
    parser.add_argument('--no_eval_in_train', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--enable_LC', action='store_true')
    parser.add_argument('--enable_IL', action='store_true')
    parser.add_argument('--enable_pomo', action='store_true')
    parser.add_argument('--disable_RL', action='store_true')
    parser.add_argument('--max_LC_len', type=int, default=-1)

    # Evaluate
    parser.add_argument(
        '--eval_only', action='store_true', help='Set this value to only evaluate model'
    )
    parser.add_argument(
        '--eval_type',
        type=str,
        default='greedy_aug',
        choices=('greedy', 'sample', 'greedy_aug'),
    )
    parser.add_argument(
        '--val_range',
        type=int,
        nargs=2,
        default=[0, 100],
        help='Range of instances used for reporting validation performance',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=100,
        help="Batch size to use during evaluation",
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        default=no_provide,
        help='Dataset file to use for validation',
    )
    parser.add_argument('--val_customer_num', type=int, default=no_provide)
    parser.add_argument('--val_depot_num', type=int, default=no_provide)
    parser.add_argument('--val_vehicle_capacity', type=int, default=no_provide)
    parser.add_argument('--val_N_aug', type=int, default=no_provide)
    parser.add_argument('--val_LC_iter', type=int, default=no_provide)
    parser.add_argument('--sample_times', type=int, default=8)
    parser.add_argument('--max_parallel', type=int, default=sys.maxsize)

    # Misc
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--enable_DDP', action='store_true', help='Enable DDP')
    parser.add_argument('--no_save', action='store_true', help='Disable model saving')
    parser.add_argument('--no_log', action='store_true', help='Disable tensorboard log')
    parser.add_argument(
        '--no_write',
        action='store_true',
        help='Disable model saving and tensorboard log',
    )
    parser.add_argument('--run_name', default='', help='Name to identify the run')
    parser.add_argument(
        '--model_save_dir',
        default='saved_model',
        help='Directory to write output models to',
    )
    parser.add_argument('--log_dir', default='log')
    parser.add_argument(
        '--save_per_epochs',
        type=int,
        default=1,
        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints',
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=10,
        help='log info every log_step gradient steps',
    )
    parser.add_argument(
        '--load_path', help='Path to load model parameters and optimizer state from'
    )
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--first_eval_once', action='store_true')
    parser.add_argument(
        '--no_progress_bar', action='store_true', help='Disable progress bar'
    )
    parser.add_argument('--DDP_port_offset', type=int, default=0)
    parser.add_argument(
        '--zoom',
        choices=('on', 'off', 'auto'),
        default='off',
        help='zoom on, off or auto when fine tune or eval instances',
    )

    opts = Option()
    parser.parse_args(args, namespace=opts)

    if opts.fine_tune:
        opts.val_batch_size = 1
        assert opts.val_range[1] - opts.val_range[0] == 1

    if opts.epoch_size < opts.batch_size:
        opts.batch_size = opts.epoch_size
    if (opts.val_range[1] - opts.val_range[0]) < opts.val_batch_size:
        opts.val_batch_size = opts.val_range[1] - opts.val_range[0]

    if not opts.eval_only:
        assert (
            opts.epoch_size % opts.batch_size == 0
        ), "Epoch size must be integer multiple of batch size!"

    assert (
        opts.val_range[1] - opts.val_range[0]
    ) % opts.val_batch_size == 0, (
        "Validation size must be integer multiple of validation batch size!"
    )

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.world_size = torch.cuda.device_count()
    opts.distributed = DEBUG_DDP or (
        opts.use_cuda and (opts.world_size > 1) and (opts.enable_DDP)
    )

    if opts.distributed:
        assert opts.batch_size % opts.world_size == 0
        assert opts.val_batch_size % opts.world_size == 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(4869 + opts.DDP_port_offset)

    if opts.run_name == '':
        opts.run_name = time.strftime("%Y%m%dT%H%M%S")
    else:
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))

    if opts.resume is not None:
        assert opts.load_path is None
        assert opts.epoch_start == 0
        opts.run_name = os.path.split(os.path.split(opts.resume)[0])[1]
        opts.epoch_start = (
            int(os.path.splitext(os.path.split(opts.resume)[1])[0].split("-")[1]) + 1
        )
        opts.load_path = opts.resume

    if opts.no_write:
        opts.no_save = True
        opts.no_log = True

    if opts.problem == 'tsp':
        opts.depot_num = 0
    elif opts.problem == 'cvrp':
        opts.depot_num = 1

    if opts.val_customer_num is no_provide:
        opts.val_customer_num = opts.customer_num
    if opts.val_depot_num is no_provide:
        opts.val_depot_num = opts.depot_num
    if opts.val_vehicle_capacity is no_provide:
        opts.val_vehicle_capacity = opts.vehicle_capacity
    if opts.val_N_aug is no_provide:
        opts.val_N_aug = opts.N_aug
    if opts.val_LC_iter is no_provide:
        opts.val_LC_iter = opts.LC_iter

    if opts.val_dataset is no_provide:
        opts.val_dataset = None
        if (opts.problem == 'cvrp') or (
            (opts.problem == 'mdvrp') and (opts.depot_num == 1)
        ):
            opts.val_dataset = f'datasets/cvrp-{opts.val_customer_num}.npz'
        elif opts.problem == 'mdvrp':
            opts.val_dataset = (
                f'datasets/mdvrp-{opts.val_customer_num}-{opts.depot_num}.npz'
            )
        elif 'tsp' in opts.problem:
            opts.val_dataset = f'datasets/tsp-{opts.val_customer_num}.npy'
    elif opts.val_dataset == 'random':
        opts.val_dataset = None

    if opts.zoom == 'on':
        opts.zoom_on = True
    elif opts.zoom == 'off':
        opts.zoom_on = False
    elif opts.zoom == 'auto':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if opts.problem != 'dmdvrp':
        opts.log_dir = os.path.join(
            opts.log_dir,
            "{}-{}-{}".format(opts.problem, opts.customer_num, opts.depot_num),
            opts.run_name,
        )

        opts.save_dir = os.path.join(
            opts.model_save_dir,
            "{}-{}-{}".format(opts.problem, opts.customer_num, opts.depot_num),
            opts.run_name,
        )
    else:
        assert opts.val_dataset is not None
        filename = os.path.splitext(os.path.split(opts.val_dataset)[1])[0]
        opts.log_dir = os.path.join(opts.log_dir, filename, opts.run_name)
        opts.save_dir = os.path.join(opts.model_save_dir, filename, opts.run_name)

    return opts
