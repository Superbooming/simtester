# -*- coding: utf-8 -*- 
# @Time : 2022/11/6 21:19 
# @Author : Shuyu Guo
# @File : config.py 
# @contact : guoshuyu225@gmail.com
from os.path import realpath, dirname
import os

ROOT_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data/')
MODEL_PATH = os.path.join(DATA_PATH, 'model/')
ARCHIVE_PATH = os.path.join(DATA_PATH, 'archive/')
CONFIG_PATH = os.path.join(ROOT_PATH, 'config/')

DEFAULT_DOWNLOAD_URL = {
    'multiwoz': {
        'db': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-db.zip',
        'soloist-archive': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-archive.zip',
        'soloist-base': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-base.zip',
        'soloist-context-3': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-context-3.zip',
        'soloist-context-1': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-context-1.zip',
        'soloist-domain-01': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-domain-01.zip',
        'soloist-domain-001': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-soloist-domain-001.zip',
        'damd-archive': 'https://huggingface.co/superbooming/Simtester/resolve/main/damd_multiwoz_data.zip',
        'damd-base': 'https://huggingface.co/superbooming/Simtester/resolve/main/damd_multiwoz.zip',
        'galaxy-dataset': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-galaxy-dataset.zip',
        'galaxy-archive': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-galaxy-archive.zip',
        'galaxy-base': 'https://huggingface.co/superbooming/Simtester/resolve/main/multiwoz-galaxy-base.zip',
    },
    'redial': {

    },
    'JDDC': {

    }
}

# tester configuration
tester_register_table = {
    'multiwoz': {
        'context-tester': [
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-base.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-context-3.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-context-1.yaml')
        ],
        'recommender-tester': [
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-base.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-recommend-04.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-recommend-01.yaml')
        ],
        'domain-tester': [
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-base.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-domain-01.yaml'),
            os.path.join(CONFIG_PATH, 'multiwoz/soloist/soloist-domain-001.yaml')
        ]
    },
    'redial': {

    },
    'JDDC': {

    }
}
