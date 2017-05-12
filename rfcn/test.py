# -*- coding: UTF-8 -*-

# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger

# 设置环境参数
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'


def parse_args():
    """定义脚本参数"""
    parser = argparse.ArgumentParser(description='Test a R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    # 从传入的yaml文件中读取设置写入config对象
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--show',
                        help='show test results immediately (only if \'--vis\' is specified)',
                        action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

# 传入脚本参数
args = parse_args()
if not args.vis and args.show:
    print 'Warning: visualization is not turned on, argument \'--show\' would be dumped.'
    args.show = False

# 将$DCN_ROOT/external/mxnet加入系统PATH
# 若使用其他版本的mxnet，在yaml中设置MXNET_VERSION参数（对应config.MXNET_VERSION）。
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


def main():
    # 统计GPU个数
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    # 生成日志和输出目录，用于保存测试过程的输出
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    print '\n\n------------------------------------------'
    print 'Notice:'
    print 'Testing through model \'{}\''.format(os.path.join('_'.join([iset for iset in config.dataset.image_set.split('+')]),
                                                             config.TRAIN.model_prefix))
    print '------------------------------------------'

    test_rcnn(config,  # cfg
              config.dataset.dataset,  # dataset
              config.dataset.test_image_set,  # image_set
              config.dataset.root_path,  # root_path
              config.dataset.dataset_path,  # dataset_path
              ctx,  # ctx
              os.path.join(final_output_path,
                           '..',
                           '_'.join([iset for iset in config.dataset.image_set.split('+')]),
                           config.TRAIN.model_prefix),  # prefix
              config.TEST.test_epoch,  # epoch
              args.vis,  # vis
              args.show,  # show
              args.ignore_cache,  # ignore_cache
              args.shuffle,  # shuffle
              config.TEST.HAS_RPN,  # has_rpn
              config.dataset.proposal,  # proposal
              args.thresh,  # thresh
              logger=logger,  # logger=None
              output_path=final_output_path)  # output_path=None

if __name__ == '__main__':
    main()
