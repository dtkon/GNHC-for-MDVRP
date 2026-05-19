from abc import ABC, abstractmethod
from typing import Optional, Union
import torch
from torch import Tensor


class Env(ABC):
    @abstractmethod
    def set_up(self, *problems, **kwargs) -> 'Env':
        pass

    @abstractmethod
    def step(self, *action) -> tuple[tuple, Union[bool, Tensor]]:
        '''
        return: state, done_flag
        '''
        pass

    @abstractmethod
    def reward(self) -> Tensor:
        pass

    @abstractmethod
    def solution(self) -> Tensor:
        pass

    @abstractmethod
    def pomo_action(self) -> Tensor:
        pass
