# -*- coding: utf-8 -*- 
# @Time : 2022/10/20 16:19 
# @Author : Shuyu Guo
# @File : config.py 
# @contact : guoshuyu225@gmail.com

import os
from os.path import dirname, realpath

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
DATA_PATH = os.path.join(ROOT_PATH, 'data/multiwoz/')
CONFIG_PATH = os.path.join(ROOT_PATH, 'config/multiwoz/')

# model trained in different configuration
model_register_table = {
    'soloist-base': GPT2LMHeadModel,
    'soloist-context-3': GPT2LMHeadModel,
    'soloist-context-1': GPT2LMHeadModel,
    'soloist-domain-01': GPT2LMHeadModel,
    'soloist-domain-001': GPT2LMHeadModel,
}

# tokenizer in different configuration
tokenizer_register_table = {
    'soloist-base': GPT2Tokenizer,
    'soloist-context-3': GPT2Tokenizer,
    'soloist-context-1': GPT2Tokenizer,
    'soloist-domain-01': GPT2Tokenizer,
    'soloist-domain-001': GPT2Tokenizer,
}

# tester configuration
tester_register_table = {
    'context-tester': [
        os.path.join(CONFIG_PATH, 'soloist-base.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-context-3.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-context-1.yaml')
    ],
    'recommender-tester': [
        os.path.join(CONFIG_PATH, 'soloist-base.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-recommend-04.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-recommend-01.yaml')
    ],
    'domain-tester': [
        os.path.join(CONFIG_PATH, 'soloist-base.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-domain-01.yaml'),
        os.path.join(CONFIG_PATH, 'soloist-domain-001.yaml')
    ]
}
