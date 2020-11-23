from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
import pprint

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from misc import datasets
from stageI import model
from stageI import trainer 
from misc import utils
from misc import config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        config.cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        config.cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(config.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % config.cfg.DATASET_NAME
    dataset = datasets.TextDataset(datadir, config.cfg.EMBEDDING_TYPE, 1)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if config.cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)

        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (config.cfg.DATASET_NAME, config.cfg.CONFIG_NAME, timestamp)
        utils.mkdir_p(ckt_logs_dir)
    else:
        s_tmp = config.cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = model.CondGAN(
        image_shape=dataset.image_shape
    )

    algo = trainer.CondGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    if config.cfg.TRAIN.FLAG:
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        algo.evaluate()
