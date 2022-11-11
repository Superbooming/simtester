import yaml
import os
from os.path import realpath, dirname
import subprocess
import sys
from loguru import logger
import zipfile

from simtester.utils.allennlp_file_utils import cached_path as allennlp_cached_path
from simtester.config.config import *


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

def cached_path(file_path, cached_dir=DATA_PATH):
    return allennlp_cached_path(file_path, cached_dir)

def check_and_get_data(data_type, dataset, data_name):
    '''

    Args:
        data_type ():  'model' or 'archive'
        dataset (): dataset_name
        data_name ():

    Returns:
        data_path

    '''
    assert data_type in ['archive', 'model']
    BASE_DIR = ARCHIVE_PATH if data_type == 'archive' else MODEL_PATH
    DATA_PATH = os.path.join(BASE_DIR, f'{dataset}/{data_name}')
    if not os.path.exists(DATA_PATH):
        logger.info(f"Did not find {data_type} data {dataset} -> {data_name} from cache.")
        logger.info(f'Download {data_type} data {dataset} -> {data_name} from {DEFAULT_DOWNLOAD_URL[dataset][data_name]}')
        data_file = cached_path(DEFAULT_DOWNLOAD_URL[dataset][data_name], os.path.join(BASE_DIR, f'{dataset}/'))
        archive = zipfile.ZipFile(data_file, 'r')
        logger.info(f'Unzip to ' + str(os.path.join(BASE_DIR, f'{dataset}/')))
        archive.extractall(os.path.join(BASE_DIR, f'{dataset}/'))
        archive.close()
        os.remove(data_file)
        logger.info(f'Load {data_type} data {dataset} -> {data_name}.')
    else:
        logger.info(f'Load {data_type} data {dataset} -> {data_name} from cache.')
    return DATA_PATH

def yaml_to_parser(yaml_config_path):
    import argparse

    parser = argparse.ArgumentParser()
    config = load_yaml_configs(yaml_config_path)
    for k, v in config.items():
        if isinstance(v, dict):
            group = parser.add_argument_group(k)
            for _k, _v in config[k].items():
                group.add_argument(f"--{_k}", default=_v)
        else:
            parser.add_argument(f"--{k}", default=v)
    return parser

if __name__ == '__main__':
    from simtester.agent.galaxy.args import parse_args
    from simtester.config.multiwoz.config import CONFIG_PATH
    import json

    parser = yaml_to_parser(os.path.join(CONFIG_PATH, 'galaxy/galaxy-base.yaml'))
    hparams = parse_args(parser)
    # hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 1

    print(json.dumps(hparams, indent=2))
