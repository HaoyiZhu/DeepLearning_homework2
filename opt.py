import argparse
import logging
import os
import yaml

import paddle

from types import MethodType
from easydict import EasyDict as edict

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

parser = argparse.ArgumentParser(description='MNIST Training')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='config/resnet50.yaml',
                    type=str)
parser.add_argument('--exp-id', default='default', type=str,
                    help='Experiment ID')

"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=80, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=1, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

"----------------------------- Training options -----------------------------"
parser.add_argument('--sync', default=False, dest='sync',
                    help='Use Sync Batchnorm', action='store_true')

"----------------------------- Homework task options -----------------------------"
parser.add_argument('--reduce_dataset', default=False, action='store_true',
                    help='Reduce training set labeled 0-4 to 10%')
parser.add_argument('--label_smooth', default=False,
                    help='Use label smoothing', action='store_true')
parser.add_argument('--label_weighted', default=False,
                    help='Use label smoothing', action='store_true')
parser.add_argument('--ssp', default=False, action='store_true',
                    help='Use self-supervised pretraining')


opt = parser.parse_args()
cfg_file_name = os.path.basename(opt.cfg)
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name
opt.work_dir = './exp/{}-{}/'.format(opt.exp_id, cfg_file_name)
opt.device = paddle.device.get_device()

if not os.path.exists("./exp/{}-{}".format(opt.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(opt.exp_id, cfg_file_name))

if not opt.label_smooth:
    cfg.LOSS['EPSILON'] = None
cfg.LOSS['WEIGHTED'] = opt.label_weighted

filehandler = logging.FileHandler(
    './exp/{}-{}/training.log'.format(opt.exp_id, cfg_file_name))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def epochInfo(self, set, idx, loss, acc):
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
        set=set,
        idx=idx,
        loss=loss,
        acc=acc
    ))


logger.epochInfo = MethodType(epochInfo, logger)
