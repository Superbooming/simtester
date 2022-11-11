# -*- coding: utf-8 -*- 
# @Time : 2022/10/28 8:54 
# @Author : Shuyu Guo
# @File : mapping.py 
# @contact : guoshuyu225@gmail.com
from simtester.agent import SoloistAgent, DamdAgent

# agent mapping
agent_mapping = {
    'soloist-base': SoloistAgent,
    'soloist-context-3': SoloistAgent,
    'soloist-context-1': SoloistAgent,
    'soloist-domain-01': SoloistAgent,
    'soloist-domain-001': SoloistAgent,
    'damd-base': DamdAgent,
}