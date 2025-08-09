#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import logging

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths

from config import config, update_config
from core.function import test
from dataset.build import build_dataloader
from models.build import build_model
from core.loss import build_criterion
from utils.comm import comm


def parse_args():
    parser = argparse.ArgumentParser(description='Test CvT on final test set')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='path to trained model checkpoint',
                        required=True,
                        type=str)
    parser.add_argument('--dataset-type',
                        help='which dataset to test on: val or test',
                        default='test',
                        choices=['val', 'test'],
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = build_model(config)

    logging.info(f'=> loading model from {args.model_file}')
    model.load_state_dict(torch.load(args.model_file, map_location='cpu'))
    model = model.cuda()

    # Data loading code
    logging.info(f'=> creating {args.dataset_type} dataset')
    test_loader = build_dataloader(
        config, is_train=False, distributed=False, dataset_type=args.dataset_type
    )

    criterion = build_criterion(config, train=False)
    criterion = criterion.cuda()

    logging.info(f'=> start testing on {args.dataset_type} set')
    
    # Test the model
    test_acc = test(
        config, test_loader, model, criterion,
        config.OUTPUT_DIR, config.OUTPUT_DIR,  # Use OUTPUT_DIR for both
        distributed=False
    )

    logging.info(f'=> Final {args.dataset_type} accuracy: {test_acc:.3f}%')
    print(f'\n🎉 Final {args.dataset_type} accuracy: {test_acc:.3f}%')


if __name__ == '__main__':
    main()
