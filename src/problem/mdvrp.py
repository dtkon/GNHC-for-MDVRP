from collections import deque
from typing import Optional, Union
from deprecated import deprecated
import torch
from torch import Tensor, nn
import numpy as np

from ..utils import (
    cumsum_with_clamp,
    masked_filter_and_pad,
    matrix_diag,
    pad_with_last_value,
    distribute_evenly,
    merge_deques_random_popleft,
)

from . import Env
from .utils import action_select, cal_distance, knn_indices
from .cvrp import (
    get_duplicate_mask,
    split_solutions_with_wait,
    align_solutions_to_wait,
    load_dataset,  # on purpose
)

# https://github.com/SaeedNB/DeepMDV/blob/main/problems/vrp/problem_vrp.py#L194
CAPACITIES = {
    5: 10,
    10: 20,
    20: 30,
    50: 40,
    100: 50,
    200: 100,
    400: 150,
    500: 160,
    700: 175,
    1000: 200,
    2000: 300,
    5000: 300,
}


class MDVRP_Env(Env):
    def set_up(
        self,
        problems: Tensor,
        demands: Tensor,
        depot_num: int,
        init_start_node: Optional[Tensor] = None,
        init_capacity: Optional[Tensor] = None,
        init_problem_mask: Optional[Tensor] = None,
        lock_depot: Optional[Tensor] = None,
        **kwargs,
    ) -> 'MDVRP_Env':
        '''
        problems: [batch_size, depot_num+customer_num, 3(x,y,norm_d)]

        demands: (batch_size, depot_num+customer_num)

        init_start_node, init_capacity: (batch_size, depot_num)

        init_problem_mask: (batch_size, depot_num+customer_num), init start and depot should not be masked.

        lock_depot: (batch_size, depot_num)
        '''
        self.problems = problems
        self.demands = demands
        self.capacity = -demands[:, :depot_num]
        self.batch_size, self.problem_size, _ = problems.size()
        self.depot_num = depot_num
        self.customer_num = self.problem_size - depot_num
        self.device = problems.device

        self.lock_depot = lock_depot
        self.lock_depot_mask = None
        if lock_depot is not None:
            # assert not torch.any(torch.all(lock_depot, dim=1))
            self.lock_depot_mask = lock_depot.unsqueeze(2).expand(
                -1, -1, self.problem_size
            )

        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.depot_arange = torch.arange(0, depot_num, device=self.device)
        self.depot_node = self.depot_arange.unsqueeze(0).repeat(
            self.batch_size, 1
        )  # (batch_size, depot_num)

        if init_start_node is None:
            # track solution
            self.actions = (
                self.depot_arange.unsqueeze(0)
                .unsqueeze(2)
                .repeat(self.batch_size, 1, 1)
            )
        else:
            self.actions = init_start_node.unsqueeze(2).clone()

        if lock_depot is not None and init_start_node is not None:
            self.end_node = torch.where(lock_depot, init_start_node, self.depot_node)
        else:
            self.end_node = self.depot_node

        self.current_node = self.actions[:, :, -1]  # (batch_size, depot_num)

        if init_capacity is None:
            self.remain_capacity = self.capacity.clone()
        else:
            self.remain_capacity = init_capacity.clone()

        self.assign_customer_count = torch.zeros(
            (self.batch_size, 1), dtype=torch.long, device=self.device
        )
        self.assign_customer_count += (
            ~torch.isin(self.current_node, self.depot_arange)
        ).sum(1, keepdim=True)

        if init_problem_mask is not None:
            self.assign_customer_count += init_problem_mask.sum(1, keepdim=True)

        self.done = (self.assign_customer_count == self.customer_num).view(
            -1
        )  # (batch,)

        if lock_depot is not None:
            self.done |= torch.all(lock_depot, dim=1)

        self.__cat_mask = torch.zeros(
            (self.batch_size, self.depot_num, self.customer_num),
            device=self.device,
            dtype=torch.bool,
        )

        # prepare mask
        self.selected_mask, self.done_mask = self.__prepare_mask(init_problem_mask)
        # update mask
        demand_too_large = self.remain_capacity.unsqueeze(2).expand(
            -1, -1, self.problem_size
        ) < self.demands.unsqueeze(
            1
        )  # (batch_size, depot_num, problem_size)
        self.mask = self.__update_mask(self.current_node, demand_too_large, self.done)

        return self

    def step(
        self, action: Optional[Tensor] = None, auto_backhaul: bool = True
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Tensor],
        Union[bool, Tensor],
    ]:
        '''
        Args:
            action: (batch_size, 2), (x, -1) means stand by

        Returns:
            state:
                [
                    current_node(batch_size, depot_num),
                    depot_node(batch_size, depot_num),
                    remain_capacity(batch_size, depot_num),
                    capacity(batch_size, depot_num),
                    done(batch_size,),
                    lock_depot(batch_size, depot_num),
                    mask(batch_size, depot_num, problem_size)
                ]

            is_all_batch_done
        '''
        if action is not None:
            assert torch.any(self.assign_customer_count < self.customer_num)

            vehicle_sel = action[:, 0].view(-1)
            direct_customer_sel = action[:, 1].view(-1).clone()
            direct_customer_sel[self.done] = -1

            customer_sel = torch.where(
                direct_customer_sel == -1,
                self.current_node[self.batch_arange, vehicle_sel],
                direct_customer_sel,
            )

            assert torch.all(
                (~self.mask[self.batch_arange, vehicle_sel, customer_sel])
                | (direct_customer_sel == -1)
            )

            current_visit = self.actions[:, :, -1].clone()
            current_visit[self.batch_arange, vehicle_sel] = customer_sel

            self.actions = torch.cat((self.actions, current_visit.unsqueeze(2)), dim=2)

            self.assign_customer_count += (
                ~torch.isin(customer_sel, self.depot_arange)
                & (direct_customer_sel != -1)
            ).view(self.batch_size, 1)

            self.done |= (self.assign_customer_count == self.customer_num).view(
                -1
            )  # (batch,)

            self.remain_capacity[self.batch_arange, vehicle_sel] -= self.demands[
                self.batch_arange, customer_sel
            ] * (direct_customer_sel != -1)
            self.remain_capacity[torch.isin(current_visit, self.depot_arange)] = (
                self.capacity[torch.isin(current_visit, self.depot_arange)]
            )
            # self.remain_capacity[self.batch_arange, vehicle_sel][
            #    torch.isin(customer_sel, self.depot_arange)
            # ] = 1.0 # fail to modify

            if torch.all(self.done):
                self.actions = torch.cat(
                    (self.actions, self.end_node.unsqueeze(2)), dim=2
                )

                return (
                    self.current_node,
                    self.depot_node,
                    self.remain_capacity.clone(),
                    self.capacity,
                    self.done,
                    self.lock_depot,
                    self.mask,
                ), True

            ### update mask
            demand_too_large = self.remain_capacity.unsqueeze(2).expand(
                -1, -1, self.problem_size
            ) < self.demands.unsqueeze(
                1
            )  # (batch_size, depot_num, problem_size)

            if auto_backhaul:
                ### send no capacity vehicle directly to depot
                demand_too_large_or_have_visited = (
                    demand_too_large | self.selected_mask
                )  # (batch_size, depot_num, problem_size)

                vehicle_encounter_unable_serve = torch.all(
                    demand_too_large_or_have_visited, 2
                ) & (
                    ~torch.isin(current_visit, self.depot_arange)
                )  # (batch_size, depot_num). Notice there may be multiple vehicles encounter at same time.

                if self.lock_depot is not None:
                    vehicle_encounter_unable_serve[self.lock_depot] = False

                if torch.any(vehicle_encounter_unable_serve) and not torch.all(
                    self.done
                ):
                    # update selected_mask first
                    self.selected_mask.scatter_(
                        2,
                        current_visit.unsqueeze(1).expand(-1, self.depot_num, -1),
                        True,
                    )

                    current_visit[vehicle_encounter_unable_serve] = self.depot_node[
                        vehicle_encounter_unable_serve
                    ]

                    self.actions = torch.cat(
                        (self.actions, current_visit.unsqueeze(2)), dim=2
                    )
                    self.remain_capacity[vehicle_encounter_unable_serve] = (
                        self.capacity[vehicle_encounter_unable_serve]
                    )
                    demand_too_large = self.remain_capacity.unsqueeze(2).expand(
                        -1, -1, self.problem_size
                    ) < self.demands.unsqueeze(
                        1
                    )  # (batch_size, depot_num, problem_size)

            self.mask = self.__update_mask(current_visit, demand_too_large, self.done)

            self.current_node = self.actions[:, :, -1]  # (batch_size, depot_num)

        return (
            (
                self.current_node,
                self.depot_node,
                self.remain_capacity.clone(),
                self.capacity,
                self.done,
                self.lock_depot,
                self.mask,
            ),
            torch.all(self.done),
        )

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        if torch.all(self.done):
            return cal_reward(self.problems, self.actions)
        else:
            raise RuntimeError('state not done')

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        if torch.all(self.done):
            return self.actions
        else:
            raise RuntimeError('state not done')

    def pomo_action(self) -> Tensor:
        pomo_action = torch.zeros(
            (self.batch_size, 2),
            device=self.device,
            dtype=torch.long,
        )
        pomo_action[:, 1] = torch.arange(self.depot_num, self.problem_size).repeat(
            self.batch_size // self.customer_num
        )
        return pomo_action

    def __prepare_mask(
        self, init_customer_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        '''
        end_node: (batch_size, depot_num)

        init_customer_mask: (batch_size, depot_num+customer_num)

        return: selected_mask(used to be updated), done_mask(fixed for reference)
        '''
        selected_mask = torch.zeros(
            (self.batch_size, self.depot_num, self.problem_size),
            device=self.device,
            dtype=torch.bool,
        )
        selected_mask[:, : self.depot_num, : self.depot_num] = True

        if init_customer_mask is not None:
            selected_mask |= init_customer_mask.unsqueeze(1).expand(
                -1, self.depot_num, -1
            )

        done_mask = torch.ones(
            (self.depot_num, self.problem_size),
            device=self.device,
            dtype=torch.bool,
        )
        done_mask[0, 0] = False

        return selected_mask, done_mask

    def __update_mask(
        self,
        # customer_sel: Tensor,
        current_visit: Tensor,
        demand_too_large: Tensor,
        done: Tensor,
    ) -> Tensor:
        '''
        current_visit: (batch_size, depot_num)

        selected_mask will be changed in-place
        '''
        # self.selected_mask[self.batch_arange, :, customer_sel] = (
        #    True  # (batch_size, depot_num, problem_size)
        # )
        ## above is not suitable in set_up for init start node.
        self.selected_mask.scatter_(
            2, current_visit.unsqueeze(1).expand(-1, self.depot_num, -1), True
        )

        return_to_depot = torch.isin(
            current_visit, self.depot_arange
        )  # (batch_size, depot_num)

        mask = self.selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, self.depot_arange, self.depot_arange] = False
        mask[torch.cat((matrix_diag(return_to_depot), self.__cat_mask), dim=2)] = True

        if self.lock_depot_mask is not None:
            mask[self.lock_depot_mask] = True

        mask[done] = self.done_mask

        return mask


class MDVRP_Env_LEHD(Env):
    def set_up(
        self,
        problems: Tensor,
        demands: Tensor,
        depot_num: int,
        init_start_node: Optional[Tensor] = None,
        init_capacity: Optional[Tensor] = None,
        init_problem_mask: Optional[Tensor] = None,
        lock_depot: Optional[Tensor] = None,
        knn: int = -1,
    ) -> 'MDVRP_Env_LEHD':
        '''
        problems: [batch_size, depot_num+customer_num, 3(x,y,norm_d)]

        demands: (batch_size, depot_num+customer_num)

        init_start_node, init_capacity: (batch_size, depot_num)

        init_problem_mask: (batch_size, depot_num+customer_num), init start and depot should not be masked.

        lock_depot: (batch_size, depot_num)
        '''
        self.mdvrp_env = MDVRP_Env().set_up(
            problems,
            demands,
            depot_num,
            init_start_node,
            init_capacity,
            init_problem_mask,
            lock_depot,
        )
        self.problem = problems
        self.depot_num = depot_num
        self.knn = knn

        if init_start_node is None:
            self.vehicle_launched = torch.zeros(
                (problems.size(0), depot_num, problems.size(1)),
                dtype=torch.bool,
                device=problems.device,
            )
        else:
            self.vehicle_launched = (
                (~torch.isin(init_start_node, self.mdvrp_env.depot_arange))
                .unsqueeze(2)
                .repeat(1, 1, problems.size(1))
            )

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = (
            self.init_available_index.clone()
        )  # could not be correct before step once

        return self

    def step(
        self, action: Optional[Tensor] = None, knn_must_contain: Optional[Tensor] = None
    ) -> tuple[
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Optional[Tensor],
            Tensor,
        ],
        Union[bool, Tensor],
    ]:
        '''
        Args:
            action: (batch_size, 3), action[:, 2] == 1 means come from depot
            knn_must_contain: (batch_size, 1)

        Returns:
            state:
                [
                    start_index(batch_size, depot_num),
                    depot_index(batch_size, depot_num),
                    available_index(batch_size, node_num),
                    unavailable_mask(batch_size, node_num),
                    remain_capacity(batch_size, depot_num),
                    capacity(batch_size, depot_num),
                    lock_depot(batch_size, depot_num),
                    infeasible_mask(batch_size, depot_num, node_num, 2)
                ]

            is_all_batch_done
        '''
        if action is None:
            norm_action = None
        else:
            norm_action = action[:, :2].clone()  # (batch_size, 2)
            norm_action[:, 1] = self.available_index[
                self.mdvrp_env.batch_arange, norm_action[:, 1]
            ]

            self.vehicle_launched[self.mdvrp_env.batch_arange, norm_action[:, 0]] = True

            if torch.any(action[:, 2] == 1):
                first_action = norm_action.clone()
                first_action[:, 1:] = torch.where(
                    action[:, 2:] == 1,
                    self.mdvrp_env.depot_node[
                        self.mdvrp_env.batch_arange, norm_action[:, 0]
                    ].view(-1, 1),
                    -1,
                )
                self.mdvrp_env.step(first_action, auto_backhaul=False)

        (
            start_index,
            depot_index,
            remain_capacity,
            capacity,
            batch_done,
            lock_depot,
            mask,
        ), done = self.mdvrp_env.step(norm_action, auto_backhaul=False)

        unavailable_mask = (
            self.mdvrp_env.selected_mask.clone()
        )  # (batch_size, depot_num, problem_size)

        if self.knn > 0:
            if knn_must_contain is None:
                knn = self.knn
            else:
                knn = self.knn - 1

            unavailable_num = unavailable_mask[:, 0, :].sum(1)
            available_num = unavailable_mask.size(2) - unavailable_num

            if torch.any(available_num > knn):
                knn_unavailable_mask = unavailable_mask[
                    :, 0, :
                ].clone()  # (batch_size, problem_size)

                if knn_must_contain is not None:
                    knn_unavailable_mask.scatter_(1, knn_must_contain, True)

                knn_split = distribute_evenly(knn, self.depot_num)
                knn_results = []

                for i in range(self.depot_num):
                    knn_result, _ = knn_indices(
                        self.problem,
                        start_index[:, i : i + 1],
                        knn_split[i],
                        knn_unavailable_mask,
                        0,
                    )  # (batch_size, k_split)
                    knn_results.append(knn_result)
                    knn_unavailable_mask.scatter_(1, knn_result, True)
                all_depot_knn = torch.cat(knn_results, 1)

                if knn_must_contain is not None:
                    all_depot_knn = torch.cat((all_depot_knn, knn_must_contain), 1)

                knn_choose = torch.zeros_like(
                    knn_unavailable_mask
                )  # (batch_size, problem_size)
                knn_choose.scatter_(1, all_depot_knn, True)

                unavailable_mask |= (
                    (~knn_choose).unsqueeze(1).expand(-1, self.depot_num, -1)
                )

        # self.available_index = self.init_available_index[
        #    ~unavailable_mask[:, 0, :]
        # ].view(
        #    unavailable_mask.size(0), -1
        # )  # (batch_size, node_num)

        self.available_index, trim_unavailable_mask = masked_filter_and_pad(
            self.init_available_index, unavailable_mask[:, 0, :]
        )

        second_line_mask = unavailable_mask.clone() | ~self.vehicle_launched

        if lock_depot is not None:
            second_line_mask[lock_depot] = True

        infeasible_mask = torch.stack(
            (mask, second_line_mask), dim=3
        )  # (batch_size, depot_num, problem_size, 2)
        infeasible_mask[:, :, : self.depot_num, :] = True

        if not done:
            infeasible_mask = infeasible_mask.gather(
                2,
                self.available_index.unsqueeze(1)
                .unsqueeze(3)
                .expand(-1, self.depot_num, -1, 2),
            )  # (batch_size, depot_num, node_num, 2)

            infeasible_mask[
                trim_unavailable_mask.unsqueeze(1)
                .unsqueeze(3)
                .expand(-1, self.depot_num, -1, 2)
            ] = True

            infeasible_mask[:, 0, 0, 0][batch_done] = False

        return (
            start_index,
            depot_index,
            self.available_index,
            trim_unavailable_mask,  # None,
            remain_capacity,
            capacity,
            lock_depot,
            infeasible_mask,
        ), done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.mdvrp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.mdvrp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


class MDVRP_Env_LEHD_directAction(Env):
    def set_up(
        self,
        problems: Tensor,
        demands: Tensor,
        depot_num: int,
        init_start_node: Optional[Tensor] = None,
        init_capacity: Optional[Tensor] = None,
        init_problem_mask: Optional[Tensor] = None,
        lock_depot: Optional[Tensor] = None,
        knn: int = -1,
    ) -> 'MDVRP_Env_LEHD_directAction':
        '''
        problems: [batch_size, depot_num+customer_num, 3(x,y,norm_d)]

        demands: (batch_size, depot_num+customer_num)

        init_start_node, init_capacity: (batch_size, depot_num)

        init_problem_mask: (batch_size, depot_num+customer_num), init start and depot should not be masked.

        lock_depot: (batch_size, depot_num)
        '''
        self.mdvrp_env = MDVRP_Env().set_up(
            problems,
            demands,
            depot_num,
            init_start_node,
            init_capacity,
            init_problem_mask,
            lock_depot,
        )
        self.problem = problems
        self.depot_num = depot_num
        self.knn = knn

        if init_start_node is None:
            self.vehicle_launched = torch.zeros(
                (problems.size(0), depot_num, problems.size(1)),
                dtype=torch.bool,
                device=problems.device,
            )
        else:
            self.vehicle_launched = (
                (~torch.isin(init_start_node, self.mdvrp_env.depot_arange))
                .unsqueeze(2)
                .repeat(1, 1, problems.size(1))
            )

        self.init_available_index = (
            torch.arange(0, problems.size(1), device=problems.device)
            .unsqueeze(0)
            .repeat(problems.size(0), 1)
        )

        self.available_index = (
            self.init_available_index.clone()
        )  # could not be correct before step once

        return self

    def step(
        self, action: Optional[Tensor] = None, knn_must_contain: Optional[Tensor] = None
    ) -> tuple[
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Optional[Tensor],
            Tensor,
        ],
        Union[bool, Tensor],
    ]:
        '''
        Args:
            action: (batch_size, 2)
            knn_must_contain: (batch_size, 1)

        Returns:
            state:
                [
                    start_index(batch_size, depot_num),
                    depot_index(batch_size, depot_num),
                    available_index(batch_size, node_num),
                    unavailable_mask(batch_size, node_num),
                    remain_capacity(batch_size, depot_num),
                    capacity(batch_size, depot_num),
                    lock_depot(batch_size, depot_num),
                    infeasible_mask(batch_size, depot_num, depot_num+node_num)
                ]

            is_all_batch_done
        '''
        if action is not None:
            norm_action = action.clone()  # (batch_size, 2)
            norm_action[:, 1] = self.available_index[
                self.mdvrp_env.batch_arange, action[:, 1]
            ]
        else:
            norm_action = None

        (
            start_index,
            depot_index,
            remain_capacity,
            capacity,
            batch_done,
            lock_depot,
            mask,
        ), done = self.mdvrp_env.step(norm_action, auto_backhaul=True)

        unavailable_mask = (
            self.mdvrp_env.selected_mask.clone()
        )  # (batch_size, depot_num, problem_size)

        # depot is masked, cancel
        unavailable_mask[:, :, : self.depot_num] = False

        if self.knn > 0:
            if knn_must_contain is None:
                knn = self.knn
            else:
                knn = self.knn - 1

            unavailable_num = (
                unavailable_mask[:, 0, :].sum(1) + self.depot_num
            )  # depot should be considered as unavailable
            available_num = unavailable_mask.size(2) - unavailable_num

            if torch.any(available_num > knn):
                knn_unavailable_mask = unavailable_mask[
                    :, 0, :
                ].clone()  # (batch_size, problem_size)

                knn_unavailable_mask[:, : self.depot_num] = True

                if knn_must_contain is not None:
                    knn_unavailable_mask.scatter_(1, knn_must_contain, True)

                knn_split = distribute_evenly(knn, self.depot_num)
                knn_results = []

                for i in range(self.depot_num):
                    knn_result, _ = knn_indices(
                        self.problem,
                        start_index[:, i : i + 1],
                        knn_split[i],
                        knn_unavailable_mask,
                        0,
                    )  # (batch_size, k_split)
                    knn_results.append(knn_result)
                    knn_unavailable_mask.scatter_(1, knn_result, True)
                all_depot_knn = torch.cat(knn_results, 1)

                if knn_must_contain is not None:
                    all_depot_knn = torch.cat((all_depot_knn, knn_must_contain), 1)

                knn_choose = torch.zeros_like(
                    knn_unavailable_mask
                )  # (batch_size, problem_size)
                knn_choose.scatter_(1, all_depot_knn, True)

                unavailable_mask |= (
                    (~knn_choose).unsqueeze(1).expand(-1, self.depot_num, -1)
                )

                unavailable_mask[:, :, : self.depot_num] = False

        self.available_index, trim_unavailable_mask = masked_filter_and_pad(
            self.init_available_index, unavailable_mask[:, 0, :]
        )

        if not done:
            infeasible_mask = mask.gather(
                2, self.available_index.unsqueeze(1).expand(-1, self.depot_num, -1)
            )  # (batch_size, depot_num, depot_num+node_num)

            infeasible_mask[
                trim_unavailable_mask.unsqueeze(1).expand(-1, self.depot_num, -1)
            ] = True
        else:
            infeasible_mask = mask

        return (
            start_index,
            depot_index,
            self.available_index,
            trim_unavailable_mask,  # None,
            remain_capacity,
            capacity,
            lock_depot,
            infeasible_mask,
        ), done

    def reward(self) -> Tensor:
        '''
        return: (batch_size,)
        '''
        return self.mdvrp_env.reward()

    def solution(self) -> Tensor:
        '''
        return: (batch_size, action_length)
        '''
        return self.mdvrp_env.solution()

    def pomo_action(self) -> Tensor:
        raise NotImplementedError


def random_generate(
    size: int,
    customer_num: int,
    depot_number: int,
    max_demand: int = 9,
    vehicle_capacity: int = -1,
) -> tuple[Tensor, Tensor]:
    '''
    demand range: 1 ~ max_demand, then divided by capacity

    depot demand = 0

    Returns:
        feature(batch_size, depot_number+customer_number, 3)

        ori_demand(batch_size, depot_number+customer_number)
    '''
    if vehicle_capacity < 0:
        capacity = CAPACITIES[customer_num]
    else:
        capacity = vehicle_capacity

    problems = torch.rand((size, depot_number + customer_num, 2))
    ori_demand = torch.randint(1, max_demand + 1, (size, depot_number + customer_num))
    ori_demand[:, :depot_number] = -capacity
    norm_demand = ori_demand / capacity
    feature = torch.cat((problems, norm_demand.unsqueeze(2)), dim=2)
    return feature, ori_demand


def generate_dataset(
    size: int,
    customer_num: int,
    depot_num: int,
    max_demand: int = 9,
    vehicle_capacity: int = -1,
    save_path: Optional[str] = None,
) -> list[tuple[Tensor, Tensor]]:
    feature, demand = random_generate(
        size, customer_num, depot_num, max_demand, vehicle_capacity
    )

    if save_path is not None:
        np.savez(save_path, feature=feature.cpu().numpy(), demand=demand.cpu().numpy())

    return [(feature[i], demand[i]) for i in range(size)]


def cal_reward(problems: Tensor, solutions: Tensor) -> Tensor:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    solutions: (batch_size, depot_number, node_indexes)

    return: (batch_size,)
    '''
    batch_size, depot_num, _ = solutions.size()
    problems = problems.repeat_interleave(depot_num, 0)

    route_length = cal_distance(
        problems, solutions.view(batch_size * depot_num, -1)
    ).view(batch_size, depot_num)

    return -route_length.sum(1)  # (batch_size,)


def depot_number(problems: Tensor) -> Tensor:
    return torch.sum(problems[:, :, 2] <= 0.0, dim=1)


def check_if_solutions_feasible(
    solutions: Tensor, problems: Tensor, demands: Tensor, check_visit: bool = True
) -> int:
    '''
    problems: [batch_size, problem_size, 3(x,y,d)]

    demands: (batch_size, problem_size)

    solutions: (batch_size, depot_number, node_indexes)

    return: 0 for good
    '''
    batch_size, depot_num, _ = solutions.size()
    customer_num = problems.size(1) - depot_num
    depot_index = torch.arange(depot_num, device=problems.device)

    problems_aggregate = problems.repeat_interleave(
        depot_num, 0
    )  # (batch_size*depot_number, problem_size, 3)

    solutions_aggregate = solutions.view(
        batch_size * depot_num, -1
    )  # (batch_size*depot_number, node_indexes)

    # check visit
    if check_visit:
        dup_mask = get_duplicate_mask(solutions_aggregate)
        depot_mask = torch.isin(solutions_aggregate, depot_index)
        sol_agg_check = solutions_aggregate.clone()
        sol_agg_check[dup_mask | depot_mask] = -1
        sol_check = sol_agg_check.view(batch_size, -1)
        for i in range(depot_num, depot_num + customer_num):
            if not torch.all(torch.sum(sol_check == i, 1) == 1):
                return 1

    # check demand
    solutions_aggregate_align = align_solutions_to_wait(
        solutions_aggregate, depot_index
    )

    capacity = (-demands[:, :depot_num]).view(-1)

    for (
        no_need_calculate,
        tsp_in_vrp,
        problems_of_tsp,
        mask,
    ) in split_solutions_with_wait(
        problems_aggregate, solutions_aggregate_align, depot_index=depot_index
    ):
        if not no_need_calculate:
            demands_of_tsp = demands.repeat_interleave(depot_num, 0).gather(
                1, tsp_in_vrp
            )
            demands_of_tsp[torch.isin(tsp_in_vrp, depot_index)] = 0
            demands_of_tsp[mask] = 0
            if torch.any(demands_of_tsp.sum(1) > capacity):
                return 2

    return 0


def solution_remove_dup(solution: Tensor, move_forward: bool = True) -> Tensor:
    '''
    solution: (batch_size, depot_num, node_num)
    '''
    depot_arange = torch.arange(0, solution.size(1), device=solution.device)

    if move_forward:
        dup_mask = get_duplicate_mask(
            solution.view(solution.size(0) * solution.size(1), -1)
        )
        masked_sol, mask = masked_filter_and_pad(
            solution.view(solution.size(0) * solution.size(1), -1), dup_mask, -721
        )
    else:
        is_depot = torch.isin(solution, depot_arange)
        is_new_cus = torch.cat(
            (
                ~is_depot[:, :, 0:1],
                ((solution[:, :, 1:] != solution[:, :, :-1]) & (~is_depot[:, :, 1:])),
            ),
            dim=2,
        )
        masked_sol, mask = masked_filter_and_pad(
            solution.view(solution.size(0) * solution.size(1), -1),
            (~(is_new_cus | is_depot)).view(solution.size(0) * solution.size(1), -1),
            -721,
        )

    masked_sol = masked_sol.view(solution.size(0), solution.size(1), -1)

    depot_fill = (
        depot_arange.view(-1, 1)
        .unsqueeze(0)
        .expand(solution.size(0), -1, masked_sol.size(2))
    )

    new_sol = torch.where(
        mask.view(solution.size(0), solution.size(1), -1), depot_fill, masked_sol
    )

    return new_sol


def solution_split(
    solution: Tensor, problem: Tensor, demands: Tensor, segment_length: int
) -> tuple[
    list[tuple[Tensor, Tensor, Tensor, Tensor]],
    list[Tensor],
    list[Tensor],
    list[Tensor],
]:
    '''
    solution: (batch_size, depot_num, node_num)

    problem: (batch_size, problem_size, 3)

    demands: (batch_size, problem_size)

    return: setup_infos, original_part_solutions, need_recons, remaining_solutions
    '''
    batch_size, depot_num, _ = solution.size()
    problem_size = problem.size(1)
    depot_arange = torch.arange(0, depot_num, device=solution.device)

    solution = solution_remove_dup(solution)

    demand_seq = -(demands.unsqueeze(1).expand(-1, depot_num, -1).gather(2, solution))
    remain_capacity = cumsum_with_clamp(demand_seq, -demands[0, 0])

    node_visit = torch.zeros(
        (batch_size, depot_num, problem_size), dtype=torch.long, device=problem.device
    )

    setup_infos = []
    ori_part_sols = []
    need_recons = []
    remaining_sols = []

    i = 0
    while i < solution.size(2) - 1:
        end = i + segment_length
        if end >= solution.size(2):
            end = solution.size(2) - 1

        init_start_node = solution[:, :, i]
        init_capacity = remain_capacity[:, :, i]

        ori_part_sol = solution[:, :, i : end + 1]

        is_recon_part = torch.zeros_like(ori_part_sol, dtype=torch.bool)
        in_recon_part = torch.zeros(
            (batch_size, depot_num), dtype=torch.bool, device=solution.device
        )

        for i in range(ori_part_sol.size(2) - 1, 0, -1):
            is_recon_part[:, :, i] = in_recon_part
            col_is_depot = torch.isin(ori_part_sol[:, :, i], depot_arange)
            in_recon_part |= col_is_depot
        is_recon_part[:, :, 0] = True

        lock_depot = ~torch.any(
            is_recon_part[:, :, 1:], dim=2
        )  # (batch_size, depot_num)
        need_recon = (ori_part_sol.size(2) > 2) & torch.any(~lock_depot)
        need_recons.append(need_recon)

        recon_part_sol = ori_part_sol.clone()
        recon_part_sol[~is_recon_part] = 0
        segment_visit = node_visit.scatter(2, recon_part_sol, 1)
        problem_mask = segment_visit.sum(1) == 0
        problem_mask[:, :depot_num] = False

        remaining_sol, _ = masked_filter_and_pad(
            ori_part_sol.view(batch_size * depot_num, -1),
            is_recon_part.view(batch_size * depot_num, -1),
            ori_part_sol[:, :, -1].view(batch_size * depot_num, 1),
        )
        remaining_sols.append(remaining_sol.view(batch_size, depot_num, -1))

        ori_part_sols.append(ori_part_sol)

        setup_infos.append((init_start_node, init_capacity, problem_mask, lock_depot))

        i = end

    return setup_infos, ori_part_sols, need_recons, remaining_sols


def local_reconstruct(
    problem: Tensor,
    demand: Tensor,
    solution: Tensor,
    seg_len: int,
    env: Env,
    encoder: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    knn: int = -1,
    sample_mode: str = 'sample',
    use_improved_reward: bool = True,
) -> tuple[Tensor, list[list[tuple]], list[list[Tensor]], list[Tensor]]:
    '''
    Returns:
        refined_solution: (batch_size, depot_num, node_num)

        state_histories

        action_histories

        reward_histories
    '''
    depot_num = solution.size(1)

    setup_infos, ori_part_sols, need_recons, remaining_sols = solution_split(
        solution, problem, demand, seg_len
    )

    state_histories = []
    action_histories = []
    reward_histories = []

    refined_part_sol_list = []

    with torch.no_grad():
        if encoder is not None:
            enc_problems = encoder(problem, depot_num)

        for setup_info, ori_part_sol, need_recon, remaining_sol in zip(
            setup_infos, ori_part_sols, need_recons, remaining_sols
        ):
            if need_recon:
                state_history = []
                action_history = []

                env.set_up(problem, demand, depot_num, *setup_info, knn=knn)
                state, done = env.step()

                while not done:
                    state_history.append(state)

                    if decoder is not None:
                        prob = decoder(*enc_problems, *state)
                        action = action_select(prob, sample_mode)
                    else:
                        mask = state[-1]
                        action = action_select(mask)
                    state, done = env.step(action)

                    action_history.append(action)

                state_histories.append(state_history)
                action_histories.append(action_history)

                new_part_sol = env.solution()

                if remaining_sol is not None:
                    new_part_sol = torch.cat((new_part_sol, remaining_sol), 2)

                # refine part solution
                new_sol_len = -cal_reward(problem, new_part_sol)
                ori_sol_len = -cal_reward(problem, ori_part_sol)

                if use_improved_reward:
                    reward_histories.append(ori_sol_len - new_sol_len)
                else:
                    reward_histories.append(env.reward())

                new_sol_nodes = new_part_sol.size(2)
                ori_sol_nodes = ori_part_sol.size(2)

                if ori_sol_nodes < new_sol_nodes:
                    ori_part_sol = pad_with_last_value(
                        ori_part_sol, 2, new_sol_nodes - ori_sol_nodes
                    )
                elif new_sol_nodes < ori_sol_nodes:
                    new_part_sol = pad_with_last_value(
                        new_part_sol, 2, ori_sol_nodes - new_sol_nodes
                    )

                refined_part_sol_list.append(
                    torch.where(
                        (new_sol_len < ori_sol_len)
                        .unsqueeze(1)
                        .unsqueeze(2)
                        .expand(-1, depot_num, new_part_sol.size(2)),
                        new_part_sol,
                        ori_part_sol,
                    )
                )

            else:
                refined_part_sol_list.append(ori_part_sol)

    refined_sol = torch.cat(refined_part_sol_list, dim=2)
    return refined_sol, state_histories, action_histories, reward_histories


@deprecated('low efficiency, buggy')
def solution_to_LEHD_direct_action(
    solution: Tensor, traj_num: int = 1
) -> list[list[Tensor]]:
    solution = solution_remove_dup(solution)

    batch_size, depot_num, seq_len = solution.size()

    depot_arange = torch.arange(0, depot_num, device=solution.device)
    first_false_col = torch.zeros(
        (batch_size, depot_num, 1), device=solution.device, dtype=torch.bool
    )

    is_new_customer = torch.cat(
        (first_false_col, solution[:, :, 1:] != solution[:, :, :-1]), dim=2
    ) & (~torch.isin(solution, depot_arange))

    last_is_depot = torch.cat(
        (first_false_col, torch.isin(solution[:, :, :-1], depot_arange)), dim=2
    )

    action_depot_deque: list[list[deque[Tensor]]] = [
        [deque() for _ in range(depot_num)] for _ in range(batch_size)
    ]

    for b in range(batch_size):
        for d in range(depot_num):
            for i in range(seq_len):
                if is_new_customer[b, d, i]:
                    if i > 1 and (last_is_depot[b, d, i]):
                        action_depot_deque[b][d].append(
                            torch.tensor(
                                [d, solution[b, d, i], 1], device=solution.device
                            )
                        )
                    else:
                        action_depot_deque[b][d].append(
                            torch.tensor(
                                [d, solution[b, d, i], 0], device=solution.device
                            )
                        )

    action_tensor_lists = []

    for _ in range(traj_num):
        action_list = []
        for b in range(batch_size):
            action_list.append(merge_deques_random_popleft(action_depot_deque[b]))

        action_tensor_list = []
        for i in range(len(action_list[0])):
            action_tensor = torch.stack([action[i] for action in action_list])
            action_tensor_list.append(action_tensor)

        action_tensor_lists.append(action_tensor_list)

    return action_tensor_lists
