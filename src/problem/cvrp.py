from typing import Any, Generator, Optional, overload
import torch
from torch import Tensor
import numpy as np

from ..utils.utils import masked_filter_and_pad

from . import Env
from .utils import cal_distance

# https://github.com/ai4co/rl4co/blob/d557327b8e4c0cfa95286c06428cade849e55d3f/rl4co/envs/routing/cvrp/generator.py
CAPACITIES = {
    10: 20,
    15: 25,
    20: 30,
    30: 33,
    40: 37,
    50: 40,
    60: 43,
    75: 45,
    100: 50,
    125: 55,
    150: 60,
    200: 70,
    500: 100,
    1000: 150,
}


class CVRP_Env(Env):
    def set_up(self, problems: Tensor, demands: Tensor, **kwargs) -> 'CVRP_Env':
        '''
        problems: [batch_size, problem_size, 3(x,y,norm_d)]

        demands: (batch_size, problem_size)
        '''
        self.problems = problems
        self.demands = demands
        self.capacity = -demands[:, :1]
        self.batch_size, self.problem_size, _ = problems.size()
        self.device = problems.device

        self.batch_arange = torch.arange(self.batch_size, device=self.device)

        # track solution
        self.actions = torch.zeros(
            (self.batch_size, 1), device=self.device, dtype=torch.long
        )

        self.remain_capacity = self.capacity.clone()
        self.assign_count = torch.ones_like(self.actions)
        self.current_node = self.actions[:, -1:]

        # prepare mask
        self.mask, self.selected_mask, self.done_mask = self.__prepare_mask()

        return self

    def step(
        self, action: Optional[Tensor] = None, auto_backhaul: bool = True
    ) -> tuple[tuple[Tensor, Tensor, Tensor], bool]:
        '''
        action: (batch_size, 1), -1 means stand by

        return: [current_node(batch_size, 1), norm_remain_capacity(batch_size, 1), mask(batch_size, problem_size)], is_all_batch_done
        '''
        if action is not None:
            assert torch.any(self.assign_count < self.problem_size)

            actual_action = torch.where(action == -1, self.current_node, action)

            assert torch.all(
                (~self.mask[self.batch_arange, actual_action.view(-1)])
                | (action.view(-1) == -1)
            )

            self.actions = torch.cat((self.actions, actual_action), dim=1)
            self.assign_count += (actual_action != 0) & (
                actual_action != self.current_node
            )
            done = (self.assign_count == self.problem_size).view(-1)  # (batch_size,)

            self.remain_capacity[action.view(-1) != -1] -= self.demands[
                self.batch_arange, actual_action.view(-1)
            ].view(self.batch_size, -1)[action.view(-1) != -1]
            self.remain_capacity[actual_action.view(-1) == 0] = self.capacity[
                actual_action.view(-1) == 0
            ]

            # self.problems[self.batch_arange, action.view(-1), 2] = 0.0

            if self.__is_all_batch_done():
                self.actions = torch.cat(
                    (
                        self.actions,
                        torch.zeros(
                            (self.batch_size, 1), device=self.device, dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                return (
                    self.current_node,
                    self.remain_capacity / self.capacity,
                    self.mask,
                ), True

            ### update mask
            demand_too_large = (
                self.remain_capacity.expand(-1, self.problem_size) < self.demands
            )  # (batch_size, problem_size)

            if auto_backhaul:
                ### send no capacity vehicle directly to depot
                demand_too_large_or_have_visited = (
                    demand_too_large | self.selected_mask
                )  # (batch_size, problem_size)

                vehicle_encounter_unable_serve = (
                    torch.all(demand_too_large_or_have_visited, 1) & ~done
                )  # (batch_size,)
                if torch.any(vehicle_encounter_unable_serve):
                    # update selected_mask first
                    self.selected_mask[self.batch_arange, actual_action.view(-1)] = (
                        True  # (batch_size, problem_size)
                    )
                    actual_action[vehicle_encounter_unable_serve] = 0
                    self.actions = torch.cat((self.actions, actual_action), dim=1)
                    self.remain_capacity[vehicle_encounter_unable_serve] = (
                        self.capacity[vehicle_encounter_unable_serve]
                    )
                    demand_too_large = (
                        self.remain_capacity.expand(-1, self.problem_size)
                        < self.demands
                    )  # (batch_size, problem_size)

            self.mask = self.__update_mask(actual_action, demand_too_large, done)

            self.current_node = self.actions[:, -1:]

        return (
            (self.current_node, self.remain_capacity / self.capacity, self.mask),
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
        return bool(torch.all(self.assign_count == self.problem_size))

    def __prepare_mask(self) -> tuple[Tensor, Tensor, Tensor]:
        '''
        return: mask(used in decoder), selected_mask(used to be updated), done_mask(fixed for reference)
        '''
        selected_mask = torch.zeros(
            (self.batch_size, self.problem_size), device=self.device, dtype=torch.bool
        )
        selected_mask[:, 0] = True
        mask = selected_mask.clone()
        done_mask = torch.ones(
            (1, self.problem_size), device=self.device, dtype=torch.bool
        )
        done_mask[:, 0] = False

        return mask, selected_mask, done_mask

    def __update_mask(
        self, action: Tensor, demand_too_large: Tensor, done: Tensor
    ) -> Tensor:
        '''
        selected_mask will be changed in-place
        '''
        self.selected_mask[self.batch_arange, action.view(-1)] = (
            True  # (batch_size, problem_size)
        )

        return_to_depot = action == 0  # (batch, 1)

        mask = self.selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, 0] = False
        mask[:, 0][return_to_depot.view(-1)] = True

        mask[done, :] = self.done_mask

        return mask


class CVRP_Env_LEHD(Env):
    def set_up(self, problems: Tensor, demands: Tensor, **kwargs) -> 'CVRP_Env_LEHD':
        self.cvrp_env = CVRP_Env().set_up(problems, demands)
        self.end_index = self.cvrp_env.current_node.clone()

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = self.init_available_index.clone()

        return self

    def step(self, action: Optional[Tensor] = None) -> tuple[
        tuple[Tensor, Tensor, Tensor, None, Tensor, Tensor],
        bool,
    ]:
        '''
        action: (batch_size, 2), action[:, 1] == 1 means come from depot

        return: [start_index(batch_size, 1), end_index(batch_size, 1), available_index(batch_size, node_num), unavailable_mask(None), remain_capacity(batch_size, 1), infeasible_mask(batch_size, problem_size, 2)], is_all_batch_done
        '''
        if action is None:
            norm_action = None
        else:
            norm_action = action[:, :1]
            norm_action = self.available_index[
                self.cvrp_env.batch_arange, norm_action.view(-1)
            ].view(-1, 1)

            if torch.any(action[:, 1] == 1):
                first_action = torch.where(action[:, 1:] == 1, 0, -1)
                self.cvrp_env.step(first_action, auto_backhaul=False)

        (start_index, remain_capacity, mask), done = self.cvrp_env.step(
            norm_action, auto_backhaul=False
        )
        unavailable_mask = self.cvrp_env.selected_mask.clone()
        self.available_index = self.init_available_index[~unavailable_mask].view(
            unavailable_mask.size(0), -1
        )

        second_line_mask = unavailable_mask.clone()

        if self.cvrp_env.actions.size(1) == 1:
            second_line_mask[:] = True

        infeasible_mask = torch.stack((mask, second_line_mask), dim=2)
        infeasible_mask[:, 0, :] = True

        infeasible_mask = infeasible_mask.gather(
            1, self.available_index.unsqueeze(2).expand(-1, -1, 2)
        )

        return (
            start_index,
            self.end_index,
            self.available_index,
            None,
            remain_capacity,
            infeasible_mask,
        ), done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.cvrp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.cvrp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


class CVRP_Env_AMre(Env):
    def set_up(self, problems: Tensor, demands: Tensor, **kwargs) -> 'CVRP_Env_AMre':
        self.cvrp_env = CVRP_Env().set_up(problems, demands)

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = self.init_available_index.clone()

        return self

    def step(self, action: Optional[Tensor] = None) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        bool,
    ]:
        '''
        action: (batch_size, 1)

        return: [start_index(batch_size, 1), available_index(batch_size, node_num), unavailable_mask(batch_size, node_num), remain_capacity(batch_size, 1), infeasible_mask(batch_size, node_num)], is_all_batch_done
        '''
        if action is not None:
            action = self.available_index[
                self.cvrp_env.batch_arange, action.view(-1)
            ].view(-1, 1)

        (current_node, remain_capacity, infeasible_mask), done = self.cvrp_env.step(
            action
        )  # current_node and depot are masked, need cancel.
        unavailable_mask = self.cvrp_env.selected_mask.clone()
        unavailable_mask[self.cvrp_env.batch_arange, current_node.view(-1)] = False
        unavailable_mask[:, 0] = False

        self.available_index, trim_unavailable_mask = masked_filter_and_pad(
            self.init_available_index, unavailable_mask
        )

        infeasible_mask = infeasible_mask.gather(1, self.available_index)
        infeasible_mask[trim_unavailable_mask] = True

        return (
            current_node,
            self.available_index,
            trim_unavailable_mask,
            remain_capacity,
            infeasible_mask,
        ), done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.cvrp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.cvrp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


def random_generate(
    size: int, customer_num: int, max_demand: int = 9, vehicle_capacity: int = -1
) -> tuple[Tensor, Tensor]:
    '''
    return: feature(batch_size, customer_number+1, 3), ori_demand(batch_size, customer_number+1)

    demand range: 1 ~ max_demand, then divided by capacity

    depot demand = 0
    '''
    if vehicle_capacity < 0:
        capacity = CAPACITIES[customer_num]
    else:
        capacity = vehicle_capacity

    problems = torch.rand((size, customer_num + 1, 2))
    ori_demand = torch.randint(1, max_demand + 1, (size, customer_num + 1))
    ori_demand[:, 0] = -capacity
    norm_demand = ori_demand / capacity
    feature = torch.cat((problems, norm_demand.unsqueeze(2)), dim=2)
    return feature, ori_demand


def generate_dataset(
    size: int,
    customer_num: int,
    max_demand: int = 9,
    vehicle_capacity: int = -1,
    save_path: Optional[str] = None,
    **kwargs
) -> list[tuple[Tensor, Tensor]]:
    feature, demand = random_generate(size, customer_num, max_demand, vehicle_capacity)

    if save_path is not None:
        np.savez(save_path, feature=feature.cpu().numpy(), demand=demand.cpu().numpy())

    return [(feature[i], demand[i]) for i in range(size)]


def load_dataset(path: str) -> list[tuple[Tensor, Tensor]]:
    data = np.load(path)
    feature = torch.tensor(data['feature'])
    demand = torch.tensor(data['demand'])

    return [(feature[i], demand[i]) for i in range(feature.size(0))]


def cal_reward(problems: Tensor, solutions: Tensor) -> Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    solutions: (batch_size, node_indexes), a row: 0-2-5-0-1-4-7-0-0-0, represents permutation

    return: (batch_size,)
    '''
    return -cal_distance(problems, solutions)


def routes_number(solutions: Tensor, depot_index: Optional[Tensor] = None) -> Tensor:
    '''
    solutions: (batch_size, node_indexes), a row: 0-2-5-0-0-1-4-7-0-0-0, represents permutation

    depot_index: if 0,1,2 are depots, give tensor[0,1,2]

    return: (batch_size,), for above action, routes number is 2
    '''
    if depot_index is None:
        is_depot = solutions == 0
    else:
        is_depot = torch.isin(solutions, depot_index)
    duplicate = get_duplicate_mask(is_depot)
    is_depot[duplicate] = False
    return is_depot.sum(1) - 1


def check_if_solutions_feasible(
    solutions: Tensor, problems: Tensor, demands: Tensor, check_visit: bool = True
) -> int:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    demands: (batch_size, problem_size)

    solutions: (batch_size, node_indexes)

    return: 0 for good
    '''

    # check visit
    if check_visit:
        customer_num = problems.size(1) - 1
        dup_mask = get_duplicate_mask(solutions)
        depot_mask = solutions == 0
        sol_check = solutions.clone()
        sol_check[dup_mask | depot_mask] = -1
        for i in range(1, customer_num + 1):
            if not torch.all(torch.sum(sol_check == i, 1) == 1):
                return 1
        # real_customer = ~(dup_mask | depot_mask)
        # actual_cus_num = real_customer.sum(1)
        # if not torch.all(actual_cus_num == customer_num):
        #    return False

    # check demand
    actions_align = align_solutions_to_wait(solutions)

    capacity = -demands[:, 0]

    for (
        no_need_calculate,
        tsp_in_vrp,
        problems_of_tsp,
        mask,
    ) in split_solutions_with_wait(problems, actions_align):
        if not no_need_calculate:
            demands_of_tsp = demands.gather(1, tsp_in_vrp)
            demands_of_tsp[tsp_in_vrp == 0] = 0
            demands_of_tsp[mask] = 0
            if torch.any(demands_of_tsp.sum(1) > capacity):
                return 2

    return 0


def get_duplicate_mask(solutions: Tensor) -> Tensor:
    return torch.cat(
        (
            torch.zeros(
                solutions.size(0),
                1,
                device=solutions.device,
                dtype=torch.bool,
            ),
            solutions[:, :-1] == solutions[:, 1:],
        ),
        dim=1,
    )


@overload
def _split_solutions_no_wait(
    problems: None,
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
) -> Generator[tuple[Tensor, Tensor, None, Tensor], Any, None]: ...


@overload
def _split_solutions_no_wait(
    problems: Tensor,
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
) -> Generator[tuple[Tensor, Tensor, Tensor, Tensor], Any, None]: ...


def _split_solutions_no_wait(
    problems: Optional[Tensor],
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
):
    '''
    WARNING: this function is INEFFICIENT!
    SUGGESTION: first use align_actions_to_wait, then use split_action_with_wait.

    solutions:
        [0-2-5-0-0-1-4-7-0-0-0,
         0-2-0-1-3-4-0-0-0-0-0].
    no wait.
    represents permutation.

    yield: if is no_need_to_calculate, sub_solutions, sub_problems, mask
    '''
    solutions = solutions.clone()
    if depot_index is None:
        any_batch_in_depot = torch.nonzero(torch.any(solutions == 0, dim=0))
    else:
        any_batch_in_depot = torch.nonzero(
            torch.any(torch.isin(solutions, depot_index), dim=0)
        )

    for i in range(any_batch_in_depot.size(0) - 1):
        tsp_in_vrp = solutions[:, : any_batch_in_depot[i + 1]]
        problems_of_tsp = (
            problems.gather(1, tsp_in_vrp.unsqueeze(2).expand(-1, -1, 3))
            if problems is not None
            else None
        )

        if depot_index is None:
            mask = tsp_in_vrp == 0
        else:
            mask = torch.isin(tsp_in_vrp, depot_index)
        mask[:, 0] = False

        if have_duplicate:
            duplicate = get_duplicate_mask(tsp_in_vrp)
            mask[duplicate] = True

        not_finish_tsp = (solutions[:, any_batch_in_depot[i + 1]] != 0).view(-1)
        mask[:, 1:][not_finish_tsp] = True

        no_need_calculate = torch.all(
            mask[:, 1:] == True
        )  # if mask.size(1) == 1, will be True

        yield no_need_calculate, tsp_in_vrp, problems_of_tsp, mask

        tsp_in_vrp[~not_finish_tsp] = 0


@overload
def split_solutions_with_wait(
    problems: None,
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
) -> Generator[tuple[Tensor, Tensor, None, Tensor], Any, None]: ...


@overload
def split_solutions_with_wait(
    problems: Tensor,
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
) -> Generator[tuple[Tensor, Tensor, Tensor, Tensor], Any, None]: ...


def split_solutions_with_wait(
    problems: Optional[Tensor],
    solutions: Tensor,
    have_duplicate: bool = True,
    depot_index: Optional[Tensor] = None,
):
    '''
    solutions:
        [0-2-5-0-3-1-4-7-0-8-0,
         0-2-0-0-3-4-0-0-0-1-0].
    with wait.
    represents permutation.

    yield: yield: if is no_need_to_calculate, sub_solutions, sub_problems, mask
    '''
    if depot_index is None:
        all_batch_in_depot = torch.nonzero(solutions.sum(0) == 0)
    else:
        all_batch_in_depot = torch.nonzero(
            torch.all(torch.isin(solutions, depot_index), dim=0)
        )

    for i in range(all_batch_in_depot.size(0) - 1):
        tsp_in_vrp = solutions[:, all_batch_in_depot[i] : all_batch_in_depot[i + 1]]
        problems_of_tsp = (
            problems.gather(1, tsp_in_vrp.unsqueeze(2).expand(-1, -1, 3))
            if problems is not None
            else None
        )

        if depot_index is None:
            mask = tsp_in_vrp == 0
        else:
            mask = torch.isin(tsp_in_vrp, depot_index)
        mask[:, 0] = False

        if have_duplicate:
            duplicate = get_duplicate_mask(tsp_in_vrp)
            mask[duplicate] = True

        yield (
            torch.all(tsp_in_vrp == 0)
            if depot_index is None
            else torch.all(torch.isin(tsp_in_vrp, depot_index))
        ), tsp_in_vrp, problems_of_tsp, mask


def align_solutions_to_wait(
    solutions: Tensor, depot_index: Optional[Tensor] = None
) -> Tensor:
    '''
    solutions:
        [0-2-5-0-0-1-4-7-0-0-0,
         0-2-0-1-3-4-0-0-0-0-0].
    no wait.
    represents permutation.
    '''
    if depot_index is None:
        depot_column = torch.zeros(
            (solutions.size(0), 1), dtype=torch.long, device=solutions.device
        )  # solutions[:, 0:1]
    else:
        depot_column = depot_index.view(-1, 1).repeat(
            solutions.size(0) // depot_index.size(0), 1
        )

    solutions = _erase_duplicate_depot(solutions, depot_index)

    batch_size = solutions.size(0)

    i = 0
    while i < solutions.size(1) - 1:
        if depot_index is None:
            need_consider = not (
                torch.all(solutions[:, i] == 0) or torch.all(solutions[:, i] != 0)
            )
        else:
            need_consider = not (
                torch.all(torch.isin(solutions[:, i], depot_index))
                or torch.all(~torch.isin(solutions[:, i], depot_index))
            )

        if need_consider:

            if depot_index is None:
                need_push_back = (solutions[:, i] == 0) & (solutions[:, i + 1] != 0)
            else:
                need_push_back = (torch.isin(solutions[:, i], depot_index)) & (
                    ~torch.isin(solutions[:, i + 1], depot_index)
                )

            if torch.any(need_push_back):
                solutions = torch.cat((solutions, depot_column), 1)
                need_push_back_square = need_push_back.unsqueeze(1).expand(
                    -1, solutions.size(1) - (i + 2)
                )
                solutions[:, i + 2 :] = torch.where(
                    need_push_back_square,
                    solutions[:, i + 1 : -1],
                    solutions[:, i + 2 :],
                )
                solutions[:, i + 1][need_push_back] = depot_column[need_push_back].view(
                    -1
                )
        i += 1

    return solutions


def _erase_duplicate_depot(
    solutions: Tensor, depot_index: Optional[Tensor] = None
) -> Tensor:
    '''
    Replace duplicate depots in the sequence with the next visited node.
    '''
    solutions = solutions.clone()

    for i in range(solutions.size(1) - 2, 0, -1):
        copy_target = solutions[:, i + 1]
        keep_target = solutions[:, i]

        if depot_index is None:
            need_copy = (
                (solutions[:, i] == 0) & (solutions[:, i - 1] == 0) & (copy_target != 0)
            )
        else:
            need_copy = (
                (torch.isin(solutions[:, i], depot_index))
                & (torch.isin(solutions[:, i - 1], depot_index))
                & (~torch.isin(copy_target, depot_index))
            )
        if torch.any(need_copy):
            solutions[:, i] = torch.where(need_copy, copy_target, keep_target)

    return solutions
