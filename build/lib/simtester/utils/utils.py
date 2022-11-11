import yaml
from nltk.tokenize import word_tokenize
import copy
import random
import requests
import os
from loguru import logger
import time
import tqdm

# def parse_decoding_results_direct(predictions):
#     candidates = []
#     candidates_bs = []
#     for prediction in predictions:
#         prediction = prediction.strip()
#         prediction = prediction.split('=>')[-1]
#         if 'system :' in prediction:
#             system_response = prediction.split('system :')[-1]
#         else:
#             system_response = ''
#         system_response = ' '.join(word_tokenize(system_response))
#         system_response = system_response.replace('[ ', '[').replace(' ]', ']')
#         candidates.append(system_response)
#
#         prediction = prediction.strip().split('system :')[0]
#         prediction = ' '.join(prediction.split()[:])
#         try:
#             prediction = prediction.strip().split('belief :')[1]
#         except:
#             prediction = prediction.strip().split('belief :')[0]
#         domains = prediction.split('|')
#         belief_state = {}
#         for domain_ in domains:
#             if domain_ == '':
#                 continue
#             if len(domain_.split()) == 0:
#                 continue
#             domain = domain_.split()[0]
#             if domain == 'none':
#                 continue
#             belief_state[domain] = {}
#             svs = ' '.join(domain_.split()[1:]).split(';')
#             for sv in svs:
#                 if sv.strip() == '':
#                     continue
#                 sv = sv.split(' = ')
#                 if len(sv) != 2:
#                     continue
#                 else:
#                     s, v = sv
#                 s = s.strip()
#                 v = v.strip()
#                 if v == "" or v == "dontcare" or v == 'not mentioned' or v == "don't care" or v == "dont care" or v == "do n't care" or v == 'none':
#                     continue
#                 belief_state[domain][s] = v
#         candidates_bs.append(copy.copy(belief_state))
#
#     def compare(key1, key2):
#         key1 = key1[1]
#         key2 = key2[1]
#         if key1.count('[') > key2.count('['):
#             return 1
#         elif key1.count('[') == key2.count('['):
#             return 1 if len(key1.split()) > len(key2.split()) else -1
#         else:
#             return -1
#
#     import functools
#     candidates_w_idx = [(idx, v) for idx, v in enumerate(candidates)]
#     candidates = sorted(candidates_w_idx, key=functools.cmp_to_key(compare))
#     if len(candidates) != 0:
#         idx, value = candidates[-1]
#         candidates_bs = candidates_bs[idx]
#         candidates = value
#
#     return candidates, candidates_bs
#
#
# def get_droped_bs(bs, num_drop=0.5):
#     remove_bs = {}
#     replace_bs = copy.deepcopy(bs)
#     for domain in bs.keys():
#         remove_bs[domain] = {}
#         num_slots = len(bs[domain].keys())
#         bs_list = list(bs[domain].keys())
#         random.shuffle(bs_list)
#         num_remain_slots = int(num_slots * num_drop + random.random())
#         for slot in bs_list[:num_remain_slots]:
#             remove_bs[domain][slot] = bs[domain][slot]
#         for slot in bs_list[num_remain_slots:]:
#             if slot not in slot_value[domain] or len(slot_value[domain][slot]) < 2:
#                 continue
#             while replace_bs[domain][slot] == bs[domain][slot]:
#                 replace_bs[domain][slot] = random.choice(slot_value[domain][slot])  # replace
#         # if len(remove_bs[domain]) == 0:
#         #     remove_bs.pop(domain)
#     return replace_bs, remove_bs
#
#
# def convert_to_str(state):
#     state_str = 'belief :'
#     first_domain = True
#     for domain in state.keys():
#         if first_domain:
#             state_str += ' ' + domain
#             first_domain = False
#         else:
#             state_str += '| ' + domain
#         for slot in state[domain].keys():
#             state_str = state_str + ' ' + str(slot) + ' = ' + str(state[domain][slot]) + ' ;'
#     # print(state_str)
#     return state_str

# import os
# import subprocess
# from loguru import logger
# from typing import List
#
#
# def download_model_from_BaiduDisk(download_info: List[str]):
#     script_path = '../scripts/download_model_from_baidu.sh'
#     assert os.path.isfile(script_path)
#     # logger.info([script_path, *download_info])
#     process = subprocess.Popen([script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     process.wait()  # Wait for process to complete.
#
#     # iterate on the stdout line by line
#     for line in process.stdout.readlines():
#         logger.info(f'[Download Model] {line}')
#
#
# if __name__ == '__main__':
#     from config import model_download_url_dict
#
#     download_model_from_BaiduDisk(model_download_url_dict['soloist-base'])

import subprocess
import sys


def run_shell(shell):
    cmd = subprocess.Popen(shell, stdin=subprocess.PIPE, stderr=sys.stderr, close_fds=True,
                           stdout=sys.stdout, universal_newlines=True, shell=True, bufsize=1)

    cmd.communicate()
    return cmd.returncode

def download_model(model: str):
    pass

def load_yaml_configs(filename: os.PathLike):
        """This function reads ``yaml`` file to build config dictionary

        Args:
            filename (os.PathLike): path to ``yaml`` config

        Returns:
            dict: config

        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(f.read()))
        return config_dict

def mp(func, data, processes=20, **kwargs):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    pool = torch.multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        collect = data[ids * length:(ids + 1) * length]
        results.append(pool.apply_async(func, args=(collect,), kwds=kwargs))
    pool.close()
    pool.join()
    result_collect = []
    for j, res in enumerate(results):
        result = res.get()
        result_collect.extend(result)
    return result_collect

def add(data):
    return_list = []
    for d in data:
        return_list.append(d+1)
    return return_list


if __name__ == '__main__':
    # print(run_shell('scripts/download_model_from_baidu.sh'))
    result = mp(add, [1, 4, 6, 9, 2, 3], processes=3)
    print(result)

