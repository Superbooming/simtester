# -*- coding: utf-8 -*- 
# @Time : 2022/11/6 18:37 
# @Author : Shuyu Guo
# @File : damd.py 
# @contact : guoshuyu225@gmail.com

"""
Modified from ConvLab-2 repo
repo_URL: https://github.com/thu-coai/ConvLab-2

@original_author: truthless
"""
import os

import torch
from loguru import logger
from typing import Optional
from tqdm import tqdm

from simtester.agent.damd.reader import MultiWozReader
from simtester.agent.damd.damd_net import cuda_, get_one_hot_input
from simtester.agent.damd.ontology import eos_tokens
from simtester.agent.damd.config import _Config
from simtester.agent import Agent
from simtester.utils.utils import load_yaml_configs, check_and_get_data
from simtester.config.register import *


class DamdAgent(Agent):
    def __init__(self, model: Optional[str] = None, model_file: Optional[os.PathLike] = None,
                 archive_file: Optional[os.PathLike] = None, device: Optional[str] = None,
                 config: Optional[os.PathLike] = None, name: Optional[str] = None):
        '''

        Args:
            model ():
            model_file ():
            archive_file ():
            device ():
            config ():
            name ():
        '''
        self._init_config(model, model_file, archive_file, device, config)
        if self.model_name in model_register_table:
            self._init_dataset()
            self.m = model_register_table[self.model_name]
            self._init_model()
        self.init_session()
        super(DamdAgent, self).__init__(self.response, self.init_session, name, True)

    def _init_config(self, model, model_file, archive_file, device, config):
        assert any([model, config])
        self.config = load_yaml_configs(config) if config else None
        if model:
            self.model_name = model
        else:
            assert 'model_name' in self.config
            self.model_name = self.config['model_name']
        self.dataset = self.model_name.split('-')[0]
        self.model_name = '-'.join(self.model_name.split('-')[1:])
        if not self.config:
            config_path = os.path.join(CONFIG_PATH, f'{self.dataset}/damd/{self.model_name}.yaml')
            self.config = load_yaml_configs(config_path)
        self.cfg = _Config(config_path)
        self.model_file = model_file if model_file else self.config.get('model_file', None)
        self.archive_file = archive_file if archive_file else self.config.get('archive_file', None)
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')
        logger.info(f'[Config] {self.dataset} -> {self.model_name} config has been loaded.')
        logger.info(f'{self.config}')

    def _init_model(self):
        if self.model_file:
            assert os.path.exists(self.model_file)
        else:
            DATA_PATH = os.path.join(MODEL_PATH, f'{self.dataset}/{self.cfg.eval_load_path}')
            self.model_file = DATA_PATH
            if not os.path.exists(DATA_PATH):
                check_and_get_data('model', self.dataset, self.model_name)
        self.reader = MultiWozReader(self.cfg)
        self.m = self.m(self.reader).to(self.device)
        if self.cfg.limit_bspn_vocab:
            self.reader.bspn_masks_tensor = {}
            for key, values in self.reader.bspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.bspn_masks_tensor[key] = v_
        if self.cfg.limit_aspn_vocab:
            self.reader.aspn_masks_tensor = {}
            for key, values in self.reader.aspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.aspn_masks_tensor[key] = v_
        self.cfg.model_path = os.path.join(self.model_file, 'model.pkl')
        self.load_model(self.cfg.model_path)
        self.m.eval()
        logger.info(f'[Model] {self.model_name} has been loaded.')

    def _init_dataset(self):
        if self.archive_file:
            assert os.path.exists(self.archive_file)
        else:
            DATA_PATH = os.path.join(ARCHIVE_PATH, f'{self.dataset}/data')
            if not os.path.exists(DATA_PATH):
                self.archive_file = check_and_get_data('archive', self.dataset, 'damd-archive')
                self.archive_file = DATA_PATH
        logger.info(f'[Dataset] Load MultiWOZ dataset.')

    def load_model(self, path=None):
        if not path:
            path = self.cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        logger.info('model loaded!')

    def add_torch_input(self, inputs, first_turn=False):
        need_onehot = ['user', 'usdx', 'bspn', 'aspn', 'pv_resp', 'pv_bspn', 'pv_aspn',
                       'dspn', 'pv_dspn', 'bsdx', 'pv_bsdx']
        if 'db_np' in inputs:
            inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())
        for item in ['user', 'usdx']:
            inputs[item] = cuda_(torch.from_numpy(inputs[item + '_unk_np']).long())
            if item in ['user', 'usdx']:
                inputs[item + '_nounk'] = cuda_(torch.from_numpy(inputs[item + '_np']).long())
            else:
                inputs[item + '_nounk'] = inputs[item]
            if item in need_onehot:
                inputs[item + '_onehot'] = get_one_hot_input(inputs[item + '_unk_np'])

        for item in ['resp', 'bspn', 'aspn', 'bsdx']:
            if 'pv_' + item + '_unk_np' not in inputs:
                continue
            inputs['pv_' + item] = cuda_(torch.from_numpy(inputs['pv_' + item + '_unk_np']).long())
            inputs['pv_' + item + '_nounk'] = inputs['pv_' + item]
            if 'pv_' + item in need_onehot:
                inputs['pv_' + item + '_onehot'] = get_one_hot_input(inputs['pv_' + item + '_unk_np'])

        return inputs

    def init_session(self):
        """Reset the class variables to prepare for a new session."""
        self.hidden_states = {}
        self.reader.reset()

    def response(self, usr, is_fill=True, return_bs=False, return_delex=False):
        """
        Generate agent response given user input.

        Args:
            observation (str):
                The input to the agent.
        Returns:
            response (str):
                The response generated by the agent.
        """

        u, u_delex = self.reader.preprocess_utterance(usr)
        # logger.info('usr:', usr)

        inputs = self.reader.prepare_input_np(u, u_delex)

        first_turn = self.reader.first_turn
        inputs = self.add_torch_input(inputs, first_turn=first_turn)
        decoded = self.m(inputs, self.hidden_states, first_turn, mode='test')

        decode_fn = self.reader.vocab.sentence_decode
        resp_delex = decode_fn(decoded['resp'][0], eos=eos_tokens['resp'], indicate_oov=True)

        if is_fill:
            response = self.reader.restore(resp_delex, self.reader.turn_domain, self.reader.constraint_dict).lower()
        else:
            response = resp_delex.lower()

        ret = []
        ret.append(response)
        if return_bs:
            ret.append(self.reader.constraint_dict)
        if return_delex:
            ret.append(resp_delex.lower())

        self.reader.py_prev['pv_resp'] = decoded['resp']
        if self.cfg.enable_bspn:
            self.reader.py_prev['pv_' + self.cfg.bspn_mode] = decoded[self.cfg.bspn_mode]
            # py_prev['pv_bspn'] = decoded['bspn']
        if self.cfg.enable_aspn:
            self.reader.py_prev['pv_aspn'] = decoded['aspn']
        # torch.cuda.empty_cache()
        self.reader.first_turn = False

        return tuple(ret)


if __name__ == '__main__':
    s = DamdAgent()
    logger.info(s.response("I want to find a cheap restaurant"))
    logger.info(s.response("ok, what is the address ?"))
