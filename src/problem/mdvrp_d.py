import heapq
import random, json
from typing import Any, Optional, Union
import torch
from torch import Tensor

from ..utils import cumsum_with_clamp, generate_random_booleans

from .cvrp import get_duplicate_mask
from .mdvrp import cal_reward, random_generate as random_generate_mdvrp, CAPACITIES


class DMDVRP_instance:
    def __init__(
        self, init_problem: Tensor, init_demand: Tensor, events: list[dict[str, Any]]
    ) -> None:
        '''
        init_problem: (depot_num+customer_num, 3)

        init_demand: (depot_num+customer_num,)

        event: {step, add_feature, add_demand, lock_depot, lock_step}, step: after serve this step, then happen
        '''
        self.problem = init_problem
        self.demand = init_demand
        self.events: list[tuple[int, dict[str, Any]]] = []
        for e in events:
            heapq.heappush(self.events, (e['step'], e))

        self.depot_num = int((init_demand < 0).sum())
        self.capacity = int(-init_demand[0])

        self.cur_node = torch.arange(
            self.depot_num, dtype=torch.long, device=init_problem.device
        )
        self.remain_capacity = torch.full(
            (self.depot_num,), self.capacity, device=init_problem.device
        )
        self.cur_sol = self.cur_node.reshape(-1, 1)

        self.unavailable_mask = torch.zeros_like(init_demand, dtype=torch.bool)
        self.available_index = torch.arange(
            0, init_problem.size(0), device=init_problem.device
        )

        self.done = False

    def remaining_instance(self) -> tuple[Tensor, Tensor, int, Tensor, Tensor]:
        problem = self.problem[self.available_index]
        demand = self.demand[self.available_index]

        depot_num = int((demand < 0).sum())
        assert 0 < depot_num <= self.depot_num

        return problem, demand, depot_num, self.cur_node, self.remain_capacity

    def submit_solution(self, solution: Tensor) -> int:
        '''
        solution: ( <=depot_num, sol_len)
        '''
        if self.done:
            return -1

        actual_sol = (
            self.available_index.unsqueeze(0)
            .expand(solution.size(0), -1)
            .gather(1, solution[:, 1:])
        )

        if torch.any(self.unavailable_mask[: self.depot_num]):
            sol_back = self.cur_sol[:, -1:].repeat(1, actual_sol.size(1))
            sol_back[~self.unavailable_mask[: self.depot_num]] = actual_sol
        else:
            sol_back = actual_sol

        self.cur_sol = torch.cat([self.cur_sol, sol_back], dim=1)

        if len(self.events) == 0:
            self.done = True
            return -1
        else:
            next_event_step, next_event = heapq.heappop(self.events)
            # event: {step, add_feature, add_demand, lock_depot, lock_step}

            while len(self.events) > 0:
                peek_next_event_step, peek_next_event = heapq.heappop(self.events)
                if int(peek_next_event_step) != int(next_event_step):
                    heapq.heappush(self.events, (peek_next_event_step, peek_next_event))
                    break
                else:
                    next_event |= peek_next_event

            self.cur_sol = self.cur_sol[:, : int(next_event_step) + 1]
            self.cur_node = self.cur_sol[:, -1]

            demand_seq = -(
                self.demand.unsqueeze(0)
                .expand(self.depot_num, -1)
                .gather(1, self.cur_sol)
            )
            dup_mask = get_duplicate_mask(self.cur_sol)
            demand_seq[dup_mask] = 0
            remain_capacity = cumsum_with_clamp(demand_seq, self.capacity)
            self.remain_capacity = remain_capacity[:, -1]

            # process mask
            served_cus: Tensor = self.cur_sol[:, :-1].unique()
            served_cus = served_cus[self.depot_num :]
            self.unavailable_mask[served_cus] = True

            # recover because repeated stay
            self.unavailable_mask[self.cur_sol[:, -1]] = False

            add_feature = next_event.get('add_feature', None)
            add_demand = next_event.get('add_demand', None)
            lock_depot = next_event.get('lock_depot', None)
            lock_step = next_event.get('lock_step', None)

            if add_feature is not None:
                assert add_demand is not None
                self.problem = torch.cat([self.problem, add_feature])
                self.demand = torch.cat([self.demand, add_demand])
                self.unavailable_mask = torch.cat(
                    [
                        self.unavailable_mask,
                        torch.zeros_like(add_demand, dtype=torch.bool),
                    ]
                )

            if lock_depot is not None:
                self.unavailable_mask[: self.depot_num] = torch.tensor(
                    lock_depot, device=self.unavailable_mask.device
                )

                if lock_step is not None:
                    heapq.heappush(
                        self.events,
                        (
                            next_event_step + lock_step + 0.1,
                            {
                                'lock_depot': [False] * self.depot_num,
                            },
                        ),
                    )

            if torch.any(self.unavailable_mask[: self.depot_num]):
                extra_need_mask = self.cur_node[self.unavailable_mask[: self.depot_num]]
                self.unavailable_mask[extra_need_mask] = True

            self.cur_node = self.cur_node[~self.unavailable_mask[: self.depot_num]]
            self.remain_capacity = self.remain_capacity[
                ~self.unavailable_mask[: self.depot_num]
            ]
            self.available_index = torch.nonzero(~self.unavailable_mask).view(-1)

            self.cur_node = torch.nonzero(
                self.cur_node.view(-1, 1).expand(-1, self.available_index.size(0))
                == self.available_index,
                as_tuple=True,
            )[1].view(-1)

            return next_event_step

    def is_done(self) -> bool:
        return self.done

    def solution(self) -> Tensor:
        assert self.done
        return self.cur_sol

    def objective(self):
        assert self.done
        return -cal_reward(
            self.problem.unsqueeze(0), self.cur_sol.unsqueeze(0)
        ).squeeze(0)


def random_generate(
    init_customer_num: int,
    max_customer_num: int,
    depot_number: int,
    max_add_num: Optional[int] = None,
    with_veh_stop: bool = False,
    max_stop_step: Optional[int] = None,
    stop_prob: Optional[float] = None,
    max_demand: int = 9,
    vehicle_capacity: int = -1,
    save_path: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    # if init 50, max 100, 3 depot, then max_add=6, 50/3=16.67, max_stop_step=3, stop_prob=0.3

    if vehicle_capacity < 0:
        vehicle_capacity = CAPACITIES[init_customer_num]

    init_feature, init_demand = random_generate_mdvrp(
        1, init_customer_num, depot_number, max_demand, vehicle_capacity
    )
    init_feature.squeeze_(0)
    init_demand.squeeze_(0)

    max_add_num = 2 * depot_number if max_add_num is None else max_add_num
    stop_prob = 0.3 if stop_prob is None else stop_prob

    if max_stop_step is None:
        if init_customer_num < 100:
            max_stop_step = 3
        else:
            max_stop_step = 10

    stop_cd = max_stop_step

    # expected_steps = math.ceil((max_customer_num - init_customer_num) / depot_number)

    events: list[dict[str, Any]] = []

    cur_step = 1
    cur_cus_num = init_customer_num
    lock_cd = 0
    while cur_cus_num < max_customer_num:
        cur_event: dict[str, Any] = {'step': cur_step}
        add_cus_num = random.choice(range(max_add_num + 1))
        if add_cus_num > 0:
            add_fea, add_d = random_generate_mdvrp(
                1, add_cus_num, 0, vehicle_capacity=vehicle_capacity
            )
            add_fea.squeeze_(0)
            add_d.squeeze_(0)
            cur_event['add_feature'] = add_fea
            cur_event['add_demand'] = add_d
            cur_cus_num += add_cus_num
        if with_veh_stop and random.random() < stop_prob and lock_cd >= stop_cd:
            lock_depot = generate_random_booleans(depot_number)
            lock_step = random.choice(range(1, max_stop_step + 1))
            cur_event['lock_depot'] = lock_depot
            cur_event['lock_step'] = lock_step
            lock_cd = 0
        if len(cur_event) > 1:
            events.append(cur_event)

        lock_cd += 1
        cur_step += 1

    if save_path is not None:
        save_events = []
        for ev in events:
            save_ev = {}
            for k, v in ev.items():
                if isinstance(v, torch.Tensor):
                    v = v.tolist()
                save_ev[k] = v
            save_events.append(save_ev)
        with open(save_path, 'w') as f:
            json.dump([init_feature.tolist(), init_demand.tolist(), save_events], f)

    return init_feature, init_demand, events


def load_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    events_list: list[dict[str, Any]]
    with open(path) as f:
        init_feature_list, init_demand_list, events_list = json.load(f)
    tensor_events = []
    for ev in events_list:
        tensor_ev = {}
        for k, v in ev.items():
            if isinstance(v, list) and (not isinstance(v[0], bool)):
                v = torch.tensor(v)
            tensor_ev[k] = v
        tensor_events.append(tensor_ev)
    return (
        torch.tensor(init_feature_list),
        torch.tensor(init_demand_list),
        tensor_events,
    )


def events_to_device(
    events: list[dict[str, Any]], device: Union[str, int, torch.device]
) -> None:
    for ev in events:
        for k, v in ev.items():
            if isinstance(v, torch.Tensor):
                ev[k] = v.to(device)
