# -*- coding: utf-8 -*- 
# @Time : 2022/10/27 20:48 
# @Author : Shuyu Guo
# @File : tester.py 
# @contact : guoshuyu225@gmail.com
import random
from typing import List, Union, Optional, Tuple

from simtester.agent import Agent
from simtester.config.multiwoz.config import *
from simtester.config.mapping import *
from simtester.utils import load_yaml_configs


class Tester(object):
    def __init__(self, agents: Union[List[Agent], str], rank: Optional[List[int]] = None, end_token: str = '[END]'):
        if isinstance(agents, str):
            if agents in tester_register_table:
                self.agents = []
                for agent in tester_register_table[agents]:
                    yaml_config = load_yaml_configs(agent)
                    agent = agent_mapping[yaml_config['model_name']](config=agent)
                    self.agents.append(agent)
            else:
                raise NotImplementedError(f'The tester [{agents}] has not been implemented')
        else:
            self.agents = agents
        self.rank = rank if rank else [(idx + 1) for idx in range(len(self.agents))]
        zip_list = list(zip(self.agents, self.rank))
        random.shuffle(zip_list)
        self.agents, self.rank = zip(*zip_list)
        self.agents = list(self.agents)
        self.rank = list(self.rank)
        self.end_token = end_token
        self.info = {
            'num_round': 0,
            'num_correct_round': 0,
            'exact_distinct_score': 0.0,
            'agents': [agent.get_name() for agent in self.agents],
            'rank': [r for r in self.rank]
        }
        self.start_new_round()

    def reset(self):
        zip_list = list(zip(self.agents, self.rank))
        random.shuffle(zip_list)
        self.agents, self.rank = zip(*zip_list)
        self.info = {
            'num_round': 0,
            'num_correct_round': 0,
            'exact_distinct_score': 0.0,
            'agents': [agent.get_name() for agent in self.agents],
            'rank': [r for r in self.rank]
        }
        self.start_new_round()

    def start_new_round(self):
        for agent in self.agents:
            agent.start_dialogue()
        self.dialog_state = [True] * len(self.agents)

    def interact(self, response_list: Optional[List[str]] = None, return_context: Optional[bool] = False,
                 rank: Optional[List[int]] = None) -> Union[
        Tuple[List[str], List[bool]], Tuple[List[List[str]], List[bool]], bool]:
        '''
        Interact with agents.Either response_list or rank parameters should be specified. If response_list is specified,
        tester will return the response list or context_list depending on whether return_context is set to ture, and the
        dialogue state list which denotes whether each agent has finish its current round of interaction.If rank is
        specified, tester will return rank result(true or false) and start a new round of interaction.

        Args:
            response_list ():
            return_context ():
            rank ():

        Returns:

        '''
        assert not all([response_list, rank]) or any([response_list, rank])
        if response_list:
            # TODO: parallel interaction with agents
            # TO_SOLOVE: Too many open and torch model in multiprocessing frozen
            # data_list = [(self.agents[idx], response_list[idx], return_context, self.end_token) for idx in range(len(self.agents))]
            # return_list, self.dialog_state = zip(*mp(interact_fn, data_list, processes=1))

            return_list = []
            for idx in range(len(response_list)):
                if not self.dialog_state[idx]:
                    return_list.append(self.agents[idx].get_context() if return_context else self.end_token)
                elif response_list[idx] == self.end_token:
                    self.dialog_state[idx] = False
                    return_list.append(self.agents[idx].get_context() if return_context else self.end_token)
                else:
                    res = self.agents[idx].interact(response_list[idx])
                    return_list.append(self.agents[idx].get_context() if return_context else res)
            return return_list, self.dialog_state
        if rank:
            self.info['num_round'] += 1
            is_correct = rank == self.rank
            if is_correct:
                self.info['num_correct_round'] += 1
            self.info['exact_distinct_score'] = self.info['num_correct_round'] / self.info['num_round']
            self.start_new_round()
            return is_correct

    def get_info(self):
        return self.info

    def get_score(self):
        return self.info['exact_distinct_score']

# multiprocessing target function
# def interact_fn(data):
#     print(data)
#     return_list = []
#     for d in data:
#         print(d)
#         agent, response, return_context, end_token = d
#         state = True
#         if response == end_token:
#             state = False
#             return_list.append((agent.get_context() if return_context else end_token, state))
#         else:
#             print('interact')
#             res = agent.interact(response)
#             print(res)
#             return_list.append((agent.get_context() if return_context else res, state))
#     print('done')
#     return return_list
