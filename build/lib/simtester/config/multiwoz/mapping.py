# -*- coding: utf-8 -*- 
# @Time : 2022/10/28 8:54 
# @Author : Shuyu Guo
# @File : mapping.py 
# @contact : guoshuyu225@gmail.com
from simtester.agent import MultiWOZAgent

# agent mapping
agent_mapping = {
    'soloist-base': MultiWOZAgent,
    'soloist-context-3': MultiWOZAgent,
    'soloist-context-1': MultiWOZAgent,
    'soloist-domain-01': MultiWOZAgent,
    'soloist-domain-001': MultiWOZAgent,
}