from typing import Optional
import torch
from torch import Tensor
import numpy as np

from . import Env
from .utils import cal_distance


class TSP_Env(Env):
    def set_up(self, problems: Tensor, **kwargs) -> 'TSP_Env':
        '''
        problems: (batch_size, problem_size, 2)
        '''
        self.problems = problems
        self.batch_size, self.problem_size, _ = problems.size()
        self.device = problems.device

        self.batch_arange = torch.arange(self.batch_size, device=self.device)

        # track solution
        self.actions = torch.tensor([], device=self.device, dtype=torch.long)

        self.current_node = self.actions.clone()
        self.end_node = self.actions.clone()

        # prepare mask
        self.mask = torch.zeros(
            (self.batch_size, self.problem_size), device=self.device, dtype=torch.bool
        )

        return self

    def step(
        self, action: Optional[Tensor] = None
    ) -> tuple[tuple[Tensor, Tensor, Tensor], bool]:
        '''
        action: (batch_size, 1)

        return: [current_node(batch_size, 1), end_node(batch_size, 1), mask(batch_size, problem_size)], is_all_batch_done
        '''
        if action is not None:
            assert not torch.any(self.mask[self.batch_arange, action.view(-1)])

            self.actions = torch.cat((self.actions, action), dim=1)

            if self.end_node.numel() == 0:
                self.end_node = action.clone()

            self.mask = (
                self.mask.clone()
            )  # avoid: RuntimeError: CUDA error: device-side assert triggered
            self.mask[self.batch_arange, action.view(-1)] = (
                True  # (batch_size, problem_size)
            )

            self.current_node = self.actions[:, -1:]

        return (
            (self.current_node, self.end_node, self.mask),
            self.__is_all_batch_done(),
        )

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        if self.__is_all_batch_done():
            return cal_reward(self.problems, self.actions)
        else:
            raise RuntimeError('state not done')

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        if self.__is_all_batch_done():
            return self.actions
        else:
            raise RuntimeError('state not done')

    def pomo_action(self) -> Tensor:
        raise NotImplementedError

    def __is_all_batch_done(self) -> bool:
        return bool(torch.all(self.mask))


class TSP_Env_LEHD(Env):
    def set_up(
        self, problems: Tensor, start_action: Optional[Tensor] = None, **kwargs
    ) -> 'TSP_Env_LEHD':
        '''
        problems: (batch_size, problem_size, 2)

        start_action: (batch_size, 1)
        '''
        self.tsp_env = TSP_Env().set_up(problems)

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = self.init_available_index.clone()

        if start_action is None:
            start_action = torch.randint(
                problems.size(1), (problems.size(0), 1), device=problems.device
            )
        self.tsp_env.step(start_action)

        return self

    def step(
        self, action: Optional[Tensor] = None
    ) -> tuple[tuple[Tensor, Tensor, Tensor], bool]:
        '''
        action: (batch_size, 1)

        return: [current_node(batch_size, 1), end_node(batch_size, 1), available_index(batch_size, node_num)], is_all_batch_done
        '''
        if action is not None:
            action = self.available_index[
                self.tsp_env.batch_arange, action.view(-1)
            ].view(-1, 1)

        (current_node, end_node, mask), is_all_batch_done = self.tsp_env.step(action)

        self.available_index = self.init_available_index[~mask].view(mask.size(0), -1)

        return (current_node, end_node, self.available_index), is_all_batch_done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.tsp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.tsp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


class TSP_Env_AMre(Env):
    def set_up(self, problems: Tensor, **kwargs) -> 'TSP_Env_AMre':
        '''
        problems: (batch_size, problem_size, 2)

        start_action: (batch_size, 1)
        '''
        self.tsp_env = TSP_Env().set_up(problems)

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = self.init_available_index.clone()

        return self

    def step(
        self, action: Optional[Tensor] = None
    ) -> tuple[tuple[Tensor, Tensor, None, Tensor, Tensor], bool]:
        '''
        action: (batch_size, 1)

        return: [current_node(batch_size, 1), available_index(batch_size, node_num), unavailable_mask(None), end_node(batch_size, 1), infeasible_mask(batch_size, problem_size)], is_all_batch_done
        '''
        if action is not None:
            action = self.available_index[
                self.tsp_env.batch_arange, action.view(-1)
            ].view(-1, 1)

        (current_node, end_node, mask), done = self.tsp_env.step(
            action
        )  # current_node and end_node are masked, need cancel.

        unavailable_mask = mask.clone()
        if current_node.numel() > 0:
            unavailable_mask[self.tsp_env.batch_arange, current_node.view(-1)] = False
            unavailable_mask[self.tsp_env.batch_arange, end_node.view(-1)] = False

        self.available_index = self.init_available_index[~unavailable_mask].view(
            unavailable_mask.size(0), -1
        )

        mask = mask.gather(1, self.available_index)

        return (current_node, self.available_index, None, end_node, mask), done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.tsp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.tsp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


def random_generate(size: int, customer_num: int) -> Tensor:
    '''
    return: (batch_size, customer_number, 2)
    '''
    return torch.rand((size, customer_num, 2))


def generate_dataset(
    size: int, customer_num: int, save_path: Optional[str] = None, **kwargs
) -> list[tuple[Tensor]]:
    feature = random_generate(size, customer_num)

    if save_path is not None:
        np.save(save_path, feature.cpu().numpy())

    return [(feature[i],) for i in range(feature.size(0))]


def load_dataset(path: str) -> list[tuple[Tensor]]:
    feature = torch.tensor(np.load(path))

    return [(feature[i],) for i in range(feature.size(0))]


def cal_reward(problems: Tensor, solutions: Tensor) -> Tensor:
    '''
    problems: [batch_size, problem_size, 2(x,y)]

    solutions: (batch_size, node_indexes), a row: 1-2-4-3, represents permutation

    return: (batch_size,)
    '''
    return -cal_distance(problems, solutions, True)


def check_if_solutions_feasible(solutions: Tensor, *args) -> int:
    '''
    solutions: (batch_size, node_indexes)

    return: 0 for good
    '''
    # check visit
    for i in range(solutions.size(1)):
        if not torch.all(torch.sum(solutions == i, 1) == 1):
            return 1
    return 0
