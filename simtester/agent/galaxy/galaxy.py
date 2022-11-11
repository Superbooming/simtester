# -*- coding: utf-8 -*- 
# @Time : 2022/11/9 20:11 
# @Author : Shuyu Guo
# @File : galaxy.py 
# @contact : guoshuyu225@gmail.com
'''
Modified from GALAXY repo
repo_URL: https://github.com/siat-nlp/GALAXY
'''
import json
import os
import random
from typing import Optional
from loguru import logger
import torch
from tqdm import tqdm
import numpy as np
from itertools import chain
from collections import OrderedDict

from simtester.agent import Agent
from simtester.utils.utils import load_yaml_configs, check_and_get_data, yaml_to_parser
from simtester.agent.galaxy.args import parse_args
from simtester.config.register import *
from simtester.config.config import CONFIG_PATH
from simtester.config.multiwoz.config import ARCHIVE_PATH
from simtester.agent.galaxy.generator import Generator
from simtester.agent.galaxy import ontology
from simtester.agent.galaxy import utils
from simtester.agent.galaxy.db_ops import MultiWozDB
from simtester.agent.galaxy.ontology import all_domains
from simtester.utils.multiwoz.utils import get_talk, fill


class GalaxyAgent(Agent):
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"
    sos_u_token = "<sos_u>"
    eos_u_token = "<eos_u>"
    sos_b_token = "<sos_b>"
    eos_b_token = "<eos_b>"
    sos_d_token = "<sos_d>"
    eos_d_token = "<eos_d>"
    sos_a_token = "<sos_a>"
    eos_a_token = "<eos_a>"
    sos_db_token = "<sos_db>"
    eos_db_token = "<eos_db>"
    sos_r_token = "<sos_r>"
    eos_r_token = "<eos_r>"

    @property
    def bot_id(self):
        return 0

    @property
    def user_id(self):
        return 1

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.tokenizer.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    @property
    def sos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_u_token])[0]

    @property
    def eos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_u_token])[0]

    @property
    def sos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_b_token])[0]

    @property
    def eos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_b_token])[0]

    @property
    def sos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_db_token])[0]

    @property
    def eos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_db_token])[0]

    @property
    def sos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_a_token])[0]

    @property
    def eos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_a_token])[0]

    @property
    def sos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_r_token])[0]

    @property
    def eos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_r_token])[0]

    @property
    def sos_d_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_d_token])[0]

    @property
    def eos_d_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_d_token])[0]

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
            self.model = model_register_table[self.model_name]
            self.tokenizer = tokenizer_register_table[self.model_name]
            self._init_dataset()
            self._init_tokenizer()
            self._init_model()
            self.pv_context = {}
            self.first_turn = True
            super(GalaxyAgent, self).__init__(self._interact_fn, self._start_fn, name, True)
        else:
            raise NotImplementedError(f'The model [{model}] has not been implemented')

    def _init_config(self, model, model_file, archive_file, device, config):
        assert any([model, config])
        self.config = load_yaml_configs(config) if config else None
        self.hparams = parse_args(yaml_to_parser(config)) if config else None
        if model:
            self.model_name = model
        else:
            assert 'model_name' in self.config
            self.model_name = self.config['model_name']
        self.dataset = self.model_name.split('-')[0]
        self.model_name = '-'.join(self.model_name.split('-')[1:])
        if not self.config:
            config_path = os.path.join(CONFIG_PATH, f'{self.dataset}/galaxy/{self.model_name}.yaml')
            self.config = load_yaml_configs(config_path)
            self.hparams = parse_args(yaml_to_parser(config_path))
        self.model_file = model_file if model_file else self.config.get('model_file', None)
        self.archive_file = archive_file if archive_file else self.config.get('archive_file', None)
        self.device_name = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_name)
        self.hparams.use_gpu = False if 'cpu' in self.device_name else True
        self.max_ctx_turn = self.hparams.max_ctx_turn - 1
        self.max_len = self.hparams.max_len
        self.set_seed(seed=self.hparams.seed)
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')
        logger.info(f'[Config] {self.dataset} -> {self.model_name} config has been loaded.')
        logger.info(f'{json.dumps(self.hparams, indent=2)}')

    def _init_model(self):
        if self.model_file:
            assert os.path.exists(self.model_file)
        else:
            self.model_file = check_and_get_data('model', self.dataset, self.model_name)
        self.generator = Generator.create(self.hparams, tokenizer=self.tokenizer)
        self.model = self.model.create(self.hparams, generator=self.generator)
        self._load_model_state(self.model_file)
        logger.info(f'[Model] {self.model_name} has been loaded.')
        logger.info("Total number of parameters in networks is {}".format(sum(x.numel() for x in self.model.parameters())))

    def _load_model_state(self, model_file):
        state_path = os.path.join(MODEL_PATH, f'{model_file}/state_epoch_7.model')
        model_state_dict = torch.load(state_path, map_location=lambda storage, loc: storage)

        if 'module.' in list(model_state_dict.keys())[0]:
            new_model_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                assert k[:7] == 'module.'
                new_model_state_dict[k[7:]] = v
            model_state_dict = new_model_state_dict

        new_model_state_dict = OrderedDict()
        parameters = {name: param for name, param in self.model.named_parameters()}
        for name, param in model_state_dict.items():
            if name in parameters:
                if param.shape != parameters[name].shape:
                    assert hasattr(param, "numpy")
                    arr = param.numpy()
                    z = np.random.normal(scale=self.model.initializer_range,
                                         size=parameters[name].shape).astype("float32")
                    if name == 'embedder.token_embedding.weight':
                        z[-param.shape[0]:] = arr
                        logger.info(f"part of parameter({name}) random normlize initialize")
                    else:
                        if z.shape[0] < param.shape[0]:
                            z = arr[:z.shape[0]]
                            logger.info(f"part of parameter({name}) are dropped")
                        else:
                            z[:param.shape[0]] = arr
                            logger.info(f"part of parameter({name}) random normlize initialize")
                    dtype, device = param.dtype, param.device
                    z = torch.tensor(z, dtype=dtype, device=device)
                    new_model_state_dict[name] = z
                else:
                    new_model_state_dict[name] = param
            else:
                logger.info(f"parameter({name}) are dropped")
        model_state_dict = new_model_state_dict

        for name in parameters:
            if name not in model_state_dict:
                if parameters[name].requires_grad:
                    logger.info(f"parameter({name}) random normlize initialize")
                    z = np.random.normal(scale=self.model.initializer_range,
                                         size=parameters[name].shape).astype("float32")
                    dtype, device = parameters[name].dtype, parameters[name].device
                    model_state_dict[name] = torch.tensor(z, dtype=dtype, device=device)
                else:
                    model_state_dict[name] = parameters[name]

        self.model.load_state_dict(model_state_dict)

    def _init_tokenizer(self):
        if self.archive_file:
            assert os.path.exists(self.archive_file)
        else:
            self.archive_file = check_and_get_data('archive', self.dataset, 'galaxy-archive')
        self._build_vocab()
        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        special_tokens.extend(self._add_sepcial_tokens())
        self.tokenizer = self.tokenizer(vocab_path=os.path.join(self.archive_file, 'vocab.txt'),
                                        special_tokens=special_tokens,
                                        tokenizer_type=self.hparams.tokenizer_type)
        self.hparams.Model.num_token_embeddings = self.vocab_size
        self.hparams.Model.num_turn_embeddings = self.max_ctx_turn + 1
        logger.info(f'[Tokenizer] {self.model_name} tokenizer has been loaded.')

    def _init_dataset(self):
        check_and_get_data('archive', self.dataset, 'galaxy-dataset')
        self.data_root = os.path.join(ARCHIVE_PATH, 'galaxy-dataset')
        self.db = MultiWozDB({
            'attraction': os.path.join(ARCHIVE_PATH, 'db/attraction_db_processed.json'),
            'hospital': os.path.join(ARCHIVE_PATH, 'db/hospital_db_processed.json'),
            'hotel': os.path.join(ARCHIVE_PATH, 'db/hotel_db_processed.json'),
            'police': os.path.join(ARCHIVE_PATH, 'db/police_db_processed.json'),
            'restaurant': os.path.join(ARCHIVE_PATH, 'db/restaurant_db_processed.json'),
            'taxi': os.path.join(ARCHIVE_PATH, 'db/taxi_db_processed.json'),
            'train': os.path.join(ARCHIVE_PATH, 'db/train_db_processed.json'),
        })
        logger.info(f'[Dataset] Load MultiWOZ dataset.')

    def _to_tensor(self, array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if self.hparams.use_gpu else array

    def decode_generated_bspn(self, generated):
        eos_b_id = self.eos_b_id
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]

    def bspan_to_DBpointer(self, bspan):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # logger.info(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match = None
        match_dom = ''
        for domain in all_domains:
            if matnums.get(domain, None):
                match_dom = domain
                match = matnums[domain]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        """
        ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel'] -> {'hotel': {'pricerange': 'cheap', 'type': 'hotel'}}
        """
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx + 1]
                        ns = self.vocab.decode(ns) if type(
                            ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.eos_a_id
        eos_r_id = self.eos_r_id
        eos_b_id = self.eos_b_id

        # eos_r may not exists if galaxy generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated) - 1
            logger.info('eos_r not in generated: ' + self.tokenizer.decode(generated))

        eos_a_idx = generated.index(eos_a_id)
        decoded['aspn'] = generated[: eos_a_idx + 1]
        decoded['resp'] = generated[eos_a_idx + 1: eos_r_idx + 1]
        return decoded

    def _interact_fn(self, context: str, is_fill: bool = True, return_bs: bool = False) -> str:
        turn = {}
        turn['user'] = [self.sos_u_id] + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)) + [self.eos_u_id]
        inputs, prompt_id = self.convert_turn_eval(turn, self.pv_context, self.first_turn)
        self.first_turn = False
        batch, batch_size = self.collate_fn_multi_turn(samples=[inputs])
        batch = type(batch)(map(lambda kv: (kv[0], self._to_tensor(kv[1])), batch.items()))

        outputs = self.model.infer(inputs=batch, start_id=prompt_id,
                                   eos_id=self.eos_b_id, max_gen_len=60)
        generated_bs = outputs[0].cpu().numpy().tolist()
        bspn_gen = self.decode_generated_bspn(generated_bs)

        # check DB result
        db_result = self.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen))
        if 'night' in context or 'people' in context or ('book' in self.context[-1] and 'yes' in context):
            p = random.randint(0, 9)
            if p < 8:
                book_result = '[book_success]'
            else:
                book_result = '[book_fail]'
        else:
            book_result = '[book_nores]'
        book_result = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(book_result))[0]
        db = [self.sos_db_id] + \
             self.tokenizer.convert_tokens_to_ids([db_result]) + \
             [book_result] + \
             [self.eos_db_id]
        prompt_id = self.sos_a_id

        prev_input = torch.tensor(bspn_gen + db)
        if self.model.use_gpu:
            prev_input = prev_input.cuda()
        outputs_db = self.model.infer(inputs=batch, start_id=prompt_id,
                                      eos_id=self.eos_r_id, max_gen_len=80,
                                      prev_input=prev_input)
        generated_ar = outputs_db[0].cpu().numpy().tolist()
        try:
            decoded = self.decode_generated_act_resp(generated_ar)
            decoded['bspn'] = bspn_gen
        except ValueError as exception:
            logger.info(str(exception))
            logger.info(self.tokenizer.decode(generated_ar))
            decoded = {'resp': [], 'bspn': [], 'aspn': []}

        self.pv_context['labels'] = inputs['labels']  # all true previous context
        self.pv_context['resp'] = decoded['resp']
        self.pv_context['bspn'] = decoded['bspn']
        self.pv_context['db'] = db
        self.pv_context['aspn'] = decoded['aspn']
        res = self.tokenizer.decode(decoded['resp'])
        logger.info(f'res: {res}')
        res_bs = self.bspan_to_constraint_dict(self.tokenizer.decode(bspn_gen))
        logger.info(f'bs: {res_bs}')
        res = res.replace('-s', 's')
        res = res.replace('-ly', 'ly')
        res = res.replace('-er', 'er')
        if is_fill:
            talk = get_talk(res_bs, res)
            logger.info(f'talk {talk}')
            res_fill = fill(res, res_bs, talk)
            ret = res_fill
            logger.info(f'res_fill: {res_fill}')
        else:
            ret = res
        if return_bs:
            ret = (ret, res_bs)
        return ret

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
        firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        # predict bspn aspn resp. db are not predicted. this part tbd.
        context_list = ['user']
        prompt_id = self.sos_b_id

        if first_turn:
            context = []
            for c in context_list:
                context += turn[c]

            inputs['src'] = [context]
            inputs['labels'] = [context]
        else:
            context = []
            for c in context_list:
                context += turn[c]

            pv_context = pv_turn['labels'] + [pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn'] + pv_turn['resp']]

            # prompt response, add sos_r
            inputs['src'] = pv_context + [context]

            if self.hparams.use_all_previous_context:
                inputs['labels'] = pv_context + [context]  # use all previous ubar history
            else:
                inputs['labels'] = [context]  # use previous turn

        return inputs, prompt_id

    @staticmethod
    def max_lens(X):
        lens = [len(X)]
        while isinstance(X[0], list):
            lens.append(max(map(len, X)))
            X = [x for xs in X for x in xs]
        return lens

    def _list2np(self, X, padding=0, dtype="int64"):
        shape = self.max_lens(X)
        ret = np.full(shape, padding, dtype=np.int32)

        if len(shape) == 1:
            ret = np.array(X)
        elif len(shape) == 2:
            for i, x in enumerate(X):
                ret[i, :len(x)] = np.array(x)
        elif len(shape) == 3:
            for i, xs in enumerate(X):
                for j, x in enumerate(xs):
                    ret[i, j, :len(x)] = np.array(x)
        return ret.astype(dtype)

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)

        src = [sp["src"][-self.max_ctx_turn:] for sp in samples]
        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts in src:
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(l)) for l in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            role = [[self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id] * l
                    for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])

        src_token = self._list2np(src_token, padding=self.pad_id)
        src_pos = self._list2np(src_pos, padding=self.pad_id)
        src_turn = self._list2np(src_turn, padding=self.pad_id)
        src_role = self._list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        return batch, batch_size

    def _start_fn(self):
        self.pv_context = {}
        self.first_turn = True

    @staticmethod
    def set_seed(seed):
        """ fix random seed """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        # for word in ontology.all_slots:
        # to be determine whether slot should be [slot]
        # if slot, tokenizer having trouble decoding.
        # special_tokens.append(word)
        for word in self.vocab._word2idx.keys():
            if word.startswith('[value_') and word.endswith(']'):
                special_tokens.append(word)
        special_tokens.extend(ontology.get_special_tokens(data_name=self.hparams.data_name))

        return special_tokens

    def _build_vocab(self):
        self.vocab = utils.MultiWOZVocab(3000)
        vp = os.path.join(self.data_root, f'vocab')
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size
