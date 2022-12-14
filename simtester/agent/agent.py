# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 20:21 
# @Author : Shuyu Guo
# @File : agent.py 
# @contact : guoshuyu225@gmail.com

from typing import List, Union, Callable, Optional
from loguru import logger


def _start_fn():
    pass


class Agent(object):

    def __init__(self, interact_fn: Union[Callable[[List[str]], str], Callable[[str], str]], start_fn=_start_fn,
                 name: Optional[str] = None, only_sigle_turn=False):
        self.interact_fn = interact_fn
        self.start_fn = start_fn
        self.only_sigle_turn = only_sigle_turn
        self.context = []
        if name:
            self.name = name
        else:
            self.name = 'Agent-' + self.model_name if hasattr(self, 'model_name') else 'Agent-base'
        logger.info(f'[Agent] Agent {self.name} has been created.')

    def start_dialogue(self):
        self.context = []
        self.start_fn()

    def interact(self, context: Union[str, List[str]]) -> (str, dict):
        if self.only_sigle_turn:
            assert isinstance(context, str)
            self.context.append(context)
            response = self.interact_fn(context)
        else:
            if isinstance(context, str):
                self.context.append(context)
            else:
                self.context = context
            response = self.interact_fn(self.context)
        self.context.append(response)
        return response

    def get_context(self) -> List[str]:
        return self.context

    def get_name(self):
        return self.name
