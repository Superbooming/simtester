"""
Field class
"""
import os
import random
from collections import OrderedDict
from itertools import chain
import json
import numpy as np
import spacy
from tqdm import tqdm

from simtester.agent.galaxy.tokenizer import Tokenizer
from simtester.agent.galaxy import ontology, utils
from simtester.agent.galaxy.db_ops import MultiWozDB


def max_lens(X):
    lens = [len(X)]
    while isinstance(X[0], list):
        lens.append(max(map(len, X)))
        X = [x for xs in X for x in xs]
    return lens


def list2np(X, padding=0, dtype="int64"):
    shape = max_lens(X)
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


class BPETextField(object):

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

    def __init__(self, hparams):
        self.train, self.dev, self.test = [], [], []
        self.data_name = hparams.data_name
        self.version = hparams.version
        self.data_root = hparams.data_root
        self.data_processed = hparams.data_processed
        self.gpu = hparams.gpu
        self.tokenizer = None
        self.vocab = None
        self.db = None
        self.set_stats = {}

        self.batch_size = hparams.batch_size
        self.filtered = hparams.filtered
        self.max_len = hparams.max_len
        self.min_utt_len = hparams.min_utt_len
        self.max_utt_len = hparams.max_utt_len
        self.min_ctx_turn = hparams.min_ctx_turn
        self.max_ctx_turn = hparams.max_ctx_turn - 1  # subtract reply turn

        self.use_true_prev_bspn = hparams.use_true_prev_bspn
        self.use_true_prev_aspn = hparams.use_true_prev_aspn
        self.use_true_db_pointer = hparams.use_true_db_pointer
        self.use_true_prev_resp = hparams.use_true_prev_resp
        self.use_true_curr_bspn = hparams.use_true_curr_bspn
        self.use_true_curr_aspn = hparams.use_true_curr_aspn
        self.use_all_previous_context = hparams.use_all_previous_context
        self.use_true_bspn_for_ctr_eval = hparams.use_true_bspn_for_ctr_eval
        self.use_true_domain_for_ctr_eval = hparams.use_true_domain_for_ctr_eval

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == self.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        # if (len(batch) % len(cfg.cuda_device)) != 0:
        #     batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        if self.gpu <= 1:
            if len(batch) > 0.5 * self.batch_size:
                all_batches.append(batch)
            elif len(all_batches):
                all_batches[-1].extend(batch)
            else:
                all_batches.append(batch)

        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial

    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

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

        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn

        if "act" in samples[0]:
            act = [sp["act"] for sp in samples]
            batch["act_index"] = np.array(act)

        return batch, batch_size


class MultiWOZBPETextField(BPETextField):

    def __init__(self, hparams):
        super(MultiWOZBPETextField, self).__init__(hparams)
        self.nlp = spacy.load('en_core_web_sm')

        self.db = MultiWozDB({
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        })
        self._build_vocab()

        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type=hparams.tokenizer_type)

        test_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'data/multiwoz{self.version}/testListFile.json'), 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'data/multiwoz{self.version}/valListFile.json'), 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        self._load_data()

        return

    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key == 'dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_batches(self, set_name):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            try:
                log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                    k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            except:
                log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                    k, len(turn_bucket[k]), len(batches), 0.0)
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches

        # log stats
        # logging.info(log_str)
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps  # turn-level的steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        if set_name == 'train':
            random.shuffle(all_batches)
        return all_batches

    def add_sepcial_tokens(self):
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
        special_tokens.extend(ontology.get_special_tokens(data_name=self.data_name))

        return special_tokens

    def _build_vocab(self):
        self.vocab = utils.MultiWOZVocab(3000)
        vp = os.path.join(self.data_root, f'data/multiwoz{self.version}/vocab')
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size

    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if save_temp:  # save encoded data
            encoded_file = os.path.join(self.data_root, f'data/multiwoz{self.version}', self.data_processed)

            if os.path.exists(encoded_file):
                print('Reading encoded data from {}'.format(encoded_file))
                self.data = json.loads(
                    open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
                encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                self.train = encoded_data['train']
                self.dev = encoded_data['dev']
                self.test = encoded_data['test']
            else:
                print('Encoding data now and save the encoded data in {}'.format(encoded_file))
                # not exists, encode data and save
                self.data = json.loads(
                    open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
                self.train, self.dev, self.test = [], [], []
                for fn, dial in tqdm(self.data.items()):
                    if '.json' in fn:
                        fn = fn.replace('.json', '')
                    if self.dev_files.get(fn):
                        self.dev.append(self._get_encoded_data(fn, dial))
                    elif self.test_files.get(fn):
                        self.test.append(self._get_encoded_data(fn, dial))
                    else:
                        self.train.append(self._get_encoded_data(fn, dial))

                # save encoded data
                encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
        else:  # directly read processed data and encode
            self.data = json.loads(
                open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
            self.train, self.dev, self.test = [], [], []
            for fn, dial in self.data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if self.dev_files.get(fn):
                    self.dev.append(self._get_encoded_data(fn, dial))
                elif self.test_files.get(fn):
                    self.test.append(self._get_encoded_data(fn, dial))
                else:
                    self.train.append(self._get_encoded_data(fn, dial))

        random.shuffle(self.train)
        print('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn

            enc['user'] = [self.sos_u_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['user'])) + [self.eos_u_id]
            enc['usdx'] = [self.sos_u_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['user'])) + [self.eos_u_id]
            enc['resp'] = [self.sos_r_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['resp'])) + [self.eos_r_id]
            enc['bspn'] = [self.sos_b_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['constraint'])) + [self.eos_b_id]
            enc['bsdx'] = [self.sos_b_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['cons_delex'])) + [self.eos_b_id]
            enc['aspn'] = [self.sos_a_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['sys_act'])) + [self.eos_a_id]
            enc['dspn'] = [self.sos_d_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(t['turn_domain'])) + [self.eos_d_id]

            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']

            # add db results to enc, at every turn
            db_pointer = self.bspan_to_DBpointer(t['constraint'], t['turn_domain'].split())
            if enc['pointer'][-2:] == [0, 1]:
                book_pointer = '[book_success]'
            elif enc['pointer'][-2:] == [1, 0]:
                book_pointer = '[book_fail]'
            else:
                assert enc['pointer'][-2:] == [0, 0]
                book_pointer = '[book_nores]'
            db_book_pointer = ' '.join([db_pointer, book_pointer])

            enc['db'] = [self.sos_db_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(db_book_pointer)) + [self.eos_db_id]

            if 'unified_act' in t:
                enc['act'] = [int(a) for a in t['unified_act'].split(',')]

            encoded_dial.append(enc)
        return encoded_dial

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
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

    def convert_batch_turn(self, turn_batch, pv_batch, first_turn=False):
        """
        convert the current and the last turn
        concat [U_0,R_0,...,U_{t-1}, R_{t-1}, U_t, B_t, A_t, R_t]
        firts turn: [U_t, B_t, A_t, R_t]
        try: [user, bspn, db, aspn, resp]
        """
        inputs = []
        if first_turn:
            batch_zipped = zip(
                turn_batch['user'], turn_batch['bspn'], turn_batch['db'], turn_batch['aspn'],
                turn_batch['resp'], turn_batch['act'])
            for u, b, db, a, r, act in batch_zipped:
                if self.use_true_curr_bspn:
                    src = [u + b + db]
                    tgt = a + r
                else:
                    src = [u]
                    tgt = b + db + a + r
                inputs.append({'src': src, 'tgt': tgt, 'act': act})
                pv = [src[-1], tgt]
                pv_batch.append(pv)
        else:
            batch_zipped = zip(pv_batch,
                               turn_batch['user'], turn_batch['bspn'], turn_batch['db'], turn_batch['aspn'],
                               turn_batch['resp'], turn_batch['act'])
            for i, (pv, u, b, db, a, r, act) in enumerate(batch_zipped):
                if self.use_true_curr_bspn:
                    src = pv + [u + b + db]
                    tgt = a + r
                else:
                    src = pv + [u]
                    tgt = b + db + a + r
                inputs.append({'src': src, 'tgt': tgt, 'act': act})
                pv = [src[-1], tgt]
                pv_batch[i].extend(pv)

        return inputs, pv_batch

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                 'dspn_gen', 'dspn', 'bspn', 'pointer', 'qspn_gen', 'qspn']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'trun_num': len(turns)}
            for f in field[2:]:
                entry[f] = ''
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        v = " ".join(v)
                    else:
                        pass  # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

    def convert_turn_eval(self, turn, pv_turn, first_turn=False):
        """
        input: [all previous ubar, U_t, B_t, A_t] predict R_t
        firts turn: [U_t, B_t, A_t] predict R_t

        regarding the context, all previous ubar is too slow, try the previous ubar
        """
        inputs = {}

        context_list = []
        prompt_id = None
        if self.use_true_curr_bspn:
            if self.use_true_curr_aspn:  # only predict resp
                context_list = ['user', 'bspn', 'db', 'aspn']
                prompt_id = self.sos_r_id
            else:  # predicted aspn
                context_list = ['user', 'bspn', 'db']
                prompt_id = self.sos_a_id
        else:  # predict bspn aspn resp. db are not predicted. this part tbd.
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

            if self.use_true_curr_bspn:
                pv_context = pv_turn['labels'] + [pv_turn['aspn'] + pv_turn['resp']]
            else:
                pv_context = pv_turn['labels'] + [pv_turn['bspn'] + pv_turn['db'] + pv_turn['aspn'] + pv_turn['resp']]

            # prompt response, add sos_r
            inputs['src'] = pv_context + [context]

            if self.use_all_previous_context:
                inputs['labels'] = pv_context + [context]  # use all previous ubar history
            else:
                inputs['labels'] = [context]  # use previous turn

        if 'act' in turn:
            inputs['act'] = turn['act']
        return inputs, prompt_id
