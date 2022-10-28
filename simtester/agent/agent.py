# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 20:21 
# @Author : Shuyu Guo
# @File : agent.py 
# @contact : guoshuyu225@gmail.com

from typing import List, Union, Callable, Optional
from loguru import logger


class Agent(object):

    def __init__(self, interact_fn: Callable[[List[str]], str], name: Optional[str] = None):
        self.interact_fn = interact_fn
        self.context = []
        if name:
            self.name = name
        else:
            self.name = 'Agent-' + self.model_name if hasattr(self, 'model_name') else 'Agent-base'
        logger.info(f'[Agent] Agent {self.name} has been created.')

    def start_dialogue(self):
        self.context = []

    def interact(self, context: Union[str, List[str]]) -> (str, dict):
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
