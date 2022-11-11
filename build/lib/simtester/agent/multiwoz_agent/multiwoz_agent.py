# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 17:14 
# @Author : Shuyu Guo
# @File : soloist.py
# @contact : guoshuyu225@gmail.com
import copy
import random
from typing import List, Optional, Tuple
from loguru import logger
import torch
import os
import pickle as pkl
from tqdm import tqdm

from simtester.agent import Agent
from simtester.utils.utils import download_model
from simtester.utils.multiwoz.utils import parse_decoding_results_direct
from simtester.utils.multiwoz.utils import get_talk, fill
from simtester.utils.utils import load_yaml_configs
from simtester.config.multiwoz import *

class MultiWOZAgent(Agent):
    def __init__(self, model: Optional[str] = None, model_dir: Optional[os.PathLike] = None,
                 device: Optional[str] = None, config: Optional[os.PathLike] = None, name: Optional[str] = None):
        '''
        Specify model and model_dir parameters to construct model's architecture and load trained model or
        config file contains model_name and model_dir parameters. Config file should be yaml format.

        Args:
            model ():
            model_dir ():
            device ():
            config (): Yaml format file. Should contain model, tokenizer, decode three parameters.
        '''
        self._init_config(model, model_dir, device, config)
        if self.model_name in model_register_table:
            self.model = model_register_table[self.model_name]
            self.tokenizer = tokenizer_register_table[self.model_name]
            assert os.path.isdir(self.model_dir)
            self._init_tokenizer()
            self._init_model()
            self._init_dataset()
            super(MultiWOZAgent, self).__init__(self._interact_fn, name)
        else:
            raise NotImplementedError(f'The model [{model}] has not been implemented')

    def _init_config(self, model, model_dir, device, config):
        assert any([model, config])
        if config:
            self.config = load_yaml_configs(config)
            assert 'model_name' in self.config
            self.model_name = self.config['model_name']
            assert 'model_dir' in self.config
            self.model_dir = self.config['model_dir']
        else:
            self.model_name = model
            config_path = os.path.join(CONFIG_PATH, f'{self.model_name}.yaml')
            self.config = load_yaml_configs(config_path)
            self.model_dir = model_dir if model_dir else self.config['model_dir']
        self.device = torch.device(device) if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')
        logger.info(f'[Config] {self.model_name} config has been loaded.')
        logger.info(f'{self.config}')

    def _init_model(self):
        self.model = self.model.from_pretrained(self.model_dir).to(self.device)
        if 'special_tokens' in self.config:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if 'eos_token' in self.config:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.model.config.eos_token_id
        logger.info(f'[Model] {self.model_name} has been loaded.')
        logger.info(f'{self.model.config.to_json_string()}')

    def _init_tokenizer(self):
        self.tokenizer = self.tokenizer.from_pretrained(self.model_dir)
        if 'special_tokens' in self.config:
            special_tokens_dir = os.path.join(ROOT_PATH, self.config['special_tokens'])
            for line in open(special_tokens_dir, 'r', encoding='utf-8'):
                line = line.strip('\n')
                self.tokenizer.add_tokens(line)
        if 'eos_token' in self.config:
            self.tokenizer.eos_token = self.config['eos_token']
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f'[Tokenizer] {self.model_name} tokenizer has been loaded.')

    def _init_dataset(self):
        self.system_token_id = self.tokenizer.convert_tokens_to_ids(['system'])
        self.user_token_id = self.tokenizer.convert_tokens_to_ids(['user'])
        self.induction_id = self.tokenizer.convert_tokens_to_ids(['=>'])
        slot_value_path = os.path.join(ROOT_PATH, 'data/multiwoz/dataset/db_slot_value.pkl')
        with open(slot_value_path, 'rb') as f:
            self.slot_value = pkl.load(f)
        logger.info(f'[Dataset] Load MultiWOZ slot values.')

    def _interact_fn(self, context: List[str], is_fill: bool = True, return_bs: bool = False) -> str:
        '''
        Decoding process of pretrained SOLOIST model in multiwoz dataset

        Args:
            context (List[str]): dialogue context

        Returns:
            response (str): response
        '''

        def embed_context(context: List[str], tokenizer, max_turn: int, max_context_length: int) -> Tuple[
            List[int], List[int], List[int]]:
            context = context[-max_turn:]
            context_ids = []
            token_ids_for_context = []
            attention_mask = []
            for cxt in context:
                ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
                context_ids += ids
                attention_mask += [1] * len(ids)
                if 'user :' in cxt:
                    token_ids_for_context += self.user_token_id * len(ids)
                else:
                    token_ids_for_context += self.system_token_id * len(ids)

            context_token = (context_ids + self.induction_id)[-max_context_length:]
            token_type_id = (token_ids_for_context + self.system_token_id)[-max_context_length:]
            attention_mask = (attention_mask + [1])[-max_context_length:]
            return context_token, token_type_id, attention_mask

        def generate_sequence(model, tokenizer, context, token_type_ids, attention_mask, max_length=512, num_samples=1,
                              temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, device=torch.device('cpu'),
                              num_mask=1):
            # First generate response
            context_ori = copy.deepcopy(context)
            context = torch.tensor(context, dtype=torch.long, device=device)
            token_type_ids_ori = copy.deepcopy(token_type_ids)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
            token_type_ids = token_type_ids.repeat(num_samples, 1)
            attention_mask_ori = copy.deepcopy(attention_mask)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
            attention_mask = attention_mask.repeat(num_samples, 1)
            generated = context
            generated = model.generate(input_ids=generated, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask, max_length=max_length, temperature=temperature,
                                       top_k=top_k, do_sample=True, num_return_sequences=num_samples,
                                       repetition_penalty=repetition_penalty, top_p=top_p)
            text_list = tokenizer.batch_decode(generated, skip_special_tokens=True)
            text_list = [txt.replace('!', '') for txt in text_list]
            res, res_bs = parse_decoding_results_direct(text_list)

            def get_droped_bs(bs, num_drop=0.5):
                remove_bs = {}
                replace_bs = copy.deepcopy(bs)
                for domain in bs.keys():
                    remove_bs[domain] = {}
                    num_slots = len(bs[domain].keys())
                    bs_list = list(bs[domain].keys())
                    random.shuffle(bs_list)
                    num_remain_slots = int(num_slots * num_drop + random.random())
                    for slot in bs_list[:num_remain_slots]:
                        remove_bs[domain][slot] = bs[domain][slot]
                    for slot in bs_list[num_remain_slots:]:
                        if slot not in self.slot_value[domain] or len(self.slot_value[domain][slot]) < 2:
                            continue
                        while replace_bs[domain][slot] == bs[domain][slot]:
                            replace_bs[domain][slot] = random.choice(self.slot_value[domain][slot])  # replace
                return replace_bs, remove_bs

            def convert_to_str(state):
                state_str = 'belief :'
                first_domain = True
                for domain in state.keys():
                    if first_domain:
                        state_str += ' ' + domain
                        first_domain = False
                    else:
                        state_str += '| ' + domain
                    for slot in state[domain].keys():
                        state_str = state_str + ' ' + str(slot) + ' = ' + str(state[domain][slot]) + ' ;'
                return state_str

            if num_mask < 1.0:
                # Analyze and change belief states
                remove_bs_list = []
                replace_bs, remove_bs = get_droped_bs(res_bs, num_mask)
                remove_bs_list.append(remove_bs)
                bs_str = convert_to_str(replace_bs)
                bs_str = bs_str.strip(' ;').strip(';').replace(';|', '|')
                bs_str += 'system :'
                bs_str_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_str))

                # Construct new context with changed belief states
                token_type_ids_ori[0] += self.system_token_id * len(bs_str_token)
                context_ori[0] += bs_str_token
                attention_mask_ori[0] += [1] * len(bs_str_token)
                token_type_ids_ori[0] = token_type_ids_ori[0][-self.config['max_context_length']:]
                context_ori[0] = context_ori[0][-self.config['max_context_length']:]
                attention_mask_ori[0] = attention_mask_ori[0][-self.config['max_context_length']:]

                # Second generate response
                context_ori = torch.tensor(context_ori, dtype=torch.long, device=device)
                token_type_ids_ori = torch.tensor(token_type_ids_ori, dtype=torch.long, device=device)
                token_type_ids_ori = token_type_ids_ori.repeat(num_samples, 1)
                attention_mask_ori = torch.tensor(attention_mask_ori, dtype=torch.long, device=device)
                attention_mask_ori = attention_mask_ori.repeat(num_samples, 1)
                generated = context_ori
                generated = model.generate(input_ids=generated, token_type_ids=token_type_ids_ori,
                                           attention_mask=attention_mask_ori, max_length=max_length,
                                           temperature=temperature, top_k=top_k, do_sample=True,
                                           num_return_sequences=num_samples, repetition_penalty=repetition_penalty,
                                           top_p=top_p)
                text_list = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                res, res_bs = parse_decoding_results_direct(text_list)
            return res, res_bs

        def add_prefix(context: List[str]) -> List[str]:
            prefix_context = []
            is_user = True
            for c in context:
                if is_user:
                    c = c.replace('user:', 'user :')
                    prefix_context.append(c if c.startswith('user :') else 'user :' + c)
                else:
                    c = c.replace('system:', 'system :')
                    prefix_context.append(c if c.startswith('system :') else 'system :' + c)
                is_user = not is_user
            return prefix_context

        context = add_prefix(context)
        context_tokens, token_type_ids, attention_masks = embed_context(context, self.tokenizer,
                                                                        self.config['max_turn'],
                                                                        self.config['max_context_length'])
        res, res_bs = generate_sequence(model=self.model, tokenizer=self.tokenizer, context=[context_tokens],
                                    token_type_ids=[token_type_ids], attention_mask=[attention_masks],
                                    num_samples=self.config['num_sample'],
                                    temperature=self.config['temperature'],
                                    top_k=self.config['top-k'], top_p=self.config['top-p'],
                                    repetition_penalty=self.config['repetition_penalty'],
                                    device=self.device, num_mask=self.config['recommend_coefficient'])

        if is_fill:
            talk = get_talk(res_bs, res)
            res_fill = fill(res, res_bs, talk)
            ret = res_fill
        else:
            ret = res
        if return_bs:
            ret = (ret, res_bs)
        return ret