# -*- coding: utf-8 -*- 
# @Time : 2022/11/7 15:45 
# @Author : Shuyu Guo
# @File : register.py 
# @contact : guoshuyu225@gmail.com
from simtester.config.config import *

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from simtester.agent.damd.damd_net import DAMD
from simtester.agent.galaxy.model_base import ModelBase
from simtester.agent.galaxy.tokenizer import Tokenizer

# model trained in different configuration
model_register_table = {
    'soloist-base': GPT2LMHeadModel,
    'soloist-context-3': GPT2LMHeadModel,
    'soloist-context-1': GPT2LMHeadModel,
    'soloist-domain-01': GPT2LMHeadModel,
    'soloist-domain-001': GPT2LMHeadModel,
    'damd-base': DAMD,
    'galaxy-base': ModelBase,
}

# tokenizer in different configuration
tokenizer_register_table = {
    'soloist-base': GPT2Tokenizer,
    'soloist-context-3': GPT2Tokenizer,
    'soloist-context-1': GPT2Tokenizer,
    'soloist-domain-01': GPT2Tokenizer,
    'soloist-domain-001': GPT2Tokenizer,
    'galaxy-base': Tokenizer,
}