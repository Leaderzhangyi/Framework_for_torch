# -*- coding: utf-8 -*-
# @File    : base_options.py
# @Software: PyCharm

import argparse
import os
import torch
import data


class BaseOptions():
    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='newdata.csv', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # models parameters
        parser.add_argument('--input_size',type=int,default=6)
        parser.add_argument('--seq_len',type=int,default=30)
        parser.add_argument('--output_size',type=int,default=1)
        parser.add_argument('--num_layer',type=int,default=3,help="隐藏层层数")
        parser.add_argument('--hidden_size',type=int,default=100,help="隐藏层上神经元个数")
        parser.add_argument('--models', type=str, default='lstm', help='chooses which models to use.')
        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--target',type=str,default="燃料实际配比")

       # additional parameters
        parser.add_argument('--epochs', type=int, default='10')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional models-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in models and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test


        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

if __name__ == '__main__':
    opt = BaseOptions().parse()
    print(opt)
