import os
import random
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.multiprocessing as mp

from ..options import Option

from ..NN.layers import Policy
from ..NN.actor_AM import (
    AM_Encoder,
    AM_Decoder,
    AM_Decoder_MDVRP,
    AMre_Encoder,
    AMre_Decoder,
)
from ..NN.actor_LEHD import (
    LEHD_Encoder,
    LEHD_Decoder,
    LEHD_CrossAtt_Encoder,
    LEHD_CrossAtt_Decoder,
    LEHD_CrossAtt_Decoder_MDVRP_lpOut,
    LEHD_CrossAtt_Decoder_MDVRP_attOut,
    LEHD_CrossAtt_Decoder_MDVRP_attOut_directAction,
)
from ..NN.actor_Dummy import Greedy_Decoder_MDVRP, Greedy_Encoder
from ..problem import Env
from ..problem.tsp import TSP_Env, TSP_Env_LEHD, TSP_Env_AMre
from ..problem.cvrp import CVRP_Env, CVRP_Env_LEHD, CVRP_Env_AMre
from ..problem.mdvrp import MDVRP_Env, MDVRP_Env_LEHD, MDVRP_Env_LEHD_directAction

from ..utils import get_inner_model, log_eval

from .train import train
from .eval import eval


class Agent:
    encoder: Policy
    decoder: Policy
    env: Env

    def __init__(self, option: Option) -> None:
        self.option = option

        basic_problem = ''
        if 'vrp' in option.problem:
            basic_problem = 'vrp'
        elif 'tsp' in option.problem:
            basic_problem = 'tsp'

        if option.model == 'AM':
            self.encoder = AM_Encoder(
                option.embedding_dim,
                option.feed_forward_dim,
                option.n_heads_enc,
                option.n_blocks_enc,
                option.normalization,
                basic_problem,
            )
        elif option.model == 'LEHD':
            self.encoder = LEHD_Encoder(
                option.embedding_dim,
                option.feed_forward_dim,
                option.n_heads_enc,
                basic_problem,
            )
        elif option.model == 'AMre':
            self.encoder = AMre_Encoder(option.embedding_dim, basic_problem)
        elif 'LEHDca' in option.model:
            self.encoder = LEHD_CrossAtt_Encoder(option.embedding_dim, basic_problem)
        elif option.model == 'greedy':
            self.encoder = Greedy_Encoder()
        else:
            raise NotImplementedError

        if option.problem in ('tsp', 'cvrp'):
            if option.model == 'AM':
                self.decoder = AM_Decoder(
                    option.n_heads_dec, option.embedding_dim, basic_problem
                )
            elif option.model == 'LEHD':
                self.decoder = LEHD_Decoder(
                    option.n_heads_dec,
                    option.n_blocks_dec,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                    basic_problem,
                )
            elif option.model == 'AMre':
                self.decoder = AMre_Decoder(
                    option.n_heads_dec,
                    option.n_blocks_enc,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                    basic_problem,
                )
            elif option.model == 'LEHDca':
                self.decoder = LEHD_CrossAtt_Decoder(
                    option.n_heads_dec,
                    option.n_blocks_dec,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                    basic_problem,
                )
            else:
                raise NotImplementedError
        elif option.problem in ('mdvrp', 'dmdvrp'):
            if option.model == 'AM':
                self.decoder = AM_Decoder_MDVRP(
                    option.n_heads_dec, option.embedding_dim
                )
            elif option.model == 'LEHDca':
                self.decoder = LEHD_CrossAtt_Decoder_MDVRP_lpOut(
                    option.n_heads_dec,
                    option.n_blocks_dec,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                )
            elif option.model == 'LEHDca_att':
                self.decoder = LEHD_CrossAtt_Decoder_MDVRP_attOut(
                    option.n_heads_dec,
                    option.n_blocks_dec,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                )
            elif option.model == 'LEHDca_att_da':
                self.decoder = LEHD_CrossAtt_Decoder_MDVRP_attOut_directAction(
                    option.n_heads_dec,
                    option.n_blocks_dec,
                    option.embedding_dim,
                    option.feed_forward_dim,
                    option.normalization,
                    max_depot_num=100 if option.depot_num > 1 else 1,
                )
            elif option.model == 'greedy':
                self.decoder = Greedy_Decoder_MDVRP()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if (
            option.use_cuda and not option.distributed
        ):  # notice 'not option.distributed' condition: if not exist will cause DDP memory unbalance!!
            self.decoder.to(option.device)
            self.encoder.to(option.device)

        if option.problem == 'tsp':
            if option.model == 'AM':
                self.env = TSP_Env()
            elif option.model in ('LEHD', 'LEHDca'):
                self.env = TSP_Env_LEHD()
            elif option.model == 'AMre':
                self.env = TSP_Env_AMre()
            else:
                raise NotImplementedError
        elif option.problem == 'cvrp':
            if option.model == 'AM':
                self.env = CVRP_Env()
            elif option.model in ('LEHD', 'LEHDca'):
                self.env = CVRP_Env_LEHD()
            elif option.model == 'AMre':
                self.env = CVRP_Env_AMre()
            else:
                raise NotImplementedError
        elif option.problem in ('mdvrp', 'dmdvrp'):
            if option.model in ('AM', 'greedy'):
                self.env = MDVRP_Env()
            elif option.model in ('LEHDca', 'LEHDca_att'):
                self.env = MDVRP_Env_LEHD()
            elif option.model == 'LEHDca_att_da':
                self.env = MDVRP_Env_LEHD_directAction()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if not option.eval_only:
            self.optimizer = torch.optim.Adam(
                [
                    {
                        'params': self.decoder.parameters(),
                        'lr': option.lr_model,
                        'weight_decay': option.weight_decay,
                    },
                    {
                        'params': self.encoder.parameters(),
                        'lr': option.lr_model,
                        'weight_decay': option.weight_decay,
                    },
                ]
            )

            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, option.lr_decay
            )
            if option.epoch_start > 0:
                for _ in range(0, option.epoch_start):
                    self.lr_scheduler.step()

    def train(self) -> None:
        self.decoder.train()
        self.encoder.train()

    def eval(self) -> None:
        self.decoder.eval()
        self.encoder.eval()

    def save(self, epoch: int) -> None:
        print(' Saving model and state...', end='')
        torch.save(
            {
                'actor': get_inner_model(self.decoder).state_dict(),
                'pre_actor': get_inner_model(self.encoder).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'random_state': random.getstate(),
            },
            os.path.join(self.option.save_dir, 'epoch-{}.pt'.format(epoch)),
        )
        print('done.')

    def load(self) -> None:
        if self.option.load_path is not None:
            print(' [*] Loading data from {}...'.format(self.option.load_path), end='')
            load_data = torch.load(self.option.load_path, map_location='cpu')

            get_inner_model(self.decoder).load_state_dict(load_data['actor'])
            get_inner_model(self.encoder).load_state_dict(load_data['pre_actor'])

            if self.option.resume is not None:
                self.optimizer.load_state_dict(load_data['optimizer'])

                torch.set_rng_state(load_data['rng_state'])
                if self.option.use_cuda:
                    torch.cuda.set_rng_state(load_data['cuda_rng_state'])
                random.setstate(load_data['random_state'])

            torch.cuda.empty_cache()

            print('done.')

    def start_train(self) -> None:
        if self.option.distributed:
            mp.spawn(train, args=(self,), nprocs=self.option.world_size)  # type: ignore
        else:
            train(0, self)

    def start_eval(self) -> None:
        self.eval()
        self.load()

        if self.option.distributed:
            ret = mp.Manager().Queue()
            mp.spawn(eval, args=(self, ret), nprocs=self.option.world_size)  # type: ignore

            rewards, action, time_used = ret.get()
            del ret
        else:
            rewards, action, time_used = eval(0, self)

        logger = None if self.option.no_log else SummaryWriter(self.option.log_dir)
        log_eval(
            logger,
            self,
            rewards,
            action,
            None,
            self.option.save_dir if not self.option.no_save else None,
            time_used,
        )
