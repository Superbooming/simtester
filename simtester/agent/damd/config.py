"""
Modified from ConvLab-2 repo
repo_URL: https://github.com/thu-coai/ConvLab-2

@original_author: truthless
"""
import logging, os
from simtester.utils.utils import load_yaml_configs
from simtester.config.multiwoz.config import CONFIG_PATH

class _Config:
    def __init__(self, yaml_config):
        cfg = load_yaml_configs(yaml_config)
        self.__dict__.update(cfg)

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler('./log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config(os.path.join(CONFIG_PATH, 'damd/damd-base.yaml'))

